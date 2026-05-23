"""
EUDRComplianceSystem – end-to-end orchestrator.

Full analysis pipeline:
  1. Satellite acquisition  (Sentinel-2 optical + Sentinel-1 SAR + Hansen + WRI)
  2. Preprocessing          (normalise, NDVI, SAR diff, 224×224 tensors)
  3. Change detection       (Prithvi-100M Siamese network)
  4. Buffer-zone analysis   (5 km leakage risk)
  5. Compliance engine      (RED / YELLOW / GREEN)
  6. Council deliberation   (Gemma 3-agent: technical + legal + report)
  7. PDF generation         (Due Diligence Statement)

Usage
-----
    from eudr.pipeline import EUDRComplianceSystem
    from eudr.schemas import GeoJSONPolygon

    system = EUDRComplianceSystem()
    polygon = GeoJSONPolygon(coordinates=[[[-5.567,7.234], ...]])
    result = system.analyze(polygon, generate_pdf=True)
    print(result.compliance.status)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from eudr.acquisition import SatelliteDataAcquisition
from eudr.compliance import EUDRComplianceEngine, ReportGenerator
from eudr.models import ComplianceCouncil, PrithviChangeDetector
from eudr.preprocessing import GeospatialPreprocessor
from eudr.schemas import AnalysisResult, GeoJSONPolygon

logger = logging.getLogger(__name__)


class EUDRComplianceSystem:
    """
    Façade that coordinates all layers for a single farm-polygon analysis.

    Models are instantiated lazily on the first call to ``analyze()``
    to avoid GPU allocation at import time (important for test environments).

    Parameters
    ----------
    lazy_load:
        Defer heavy model loading until the first analysis call.
        Set to False to pre-warm at startup (long-running API).
    lora_checkpoint:
        Optional path to a LoRA fine-tuned checkpoint.  When supplied, the
        change-detector is loaded with the fine-tuned weights for higher
        accuracy on domain-specific data.
    """

    def __init__(
        self,
        lazy_load: bool = True,
        lora_checkpoint: Optional[str] = None,
    ) -> None:
        self._lazy = lazy_load
        self._lora_checkpoint = lora_checkpoint

        # Stateless helpers — always ready
        self._engine = EUDRComplianceEngine()
        self._reporter = ReportGenerator()
        self._preprocessor = GeospatialPreprocessor()

        # Lazy-loaded heavy components
        self._acquirer: Optional[SatelliteDataAcquisition] = None
        self._detector: Optional[PrithviChangeDetector] = None
        self._council: Optional[ComplianceCouncil] = None

        if not lazy_load:
            self._load_models()

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        polygon: GeoJSONPolygon,
        generate_pdf: bool = True,
        analyze_buffer: bool = True,
    ) -> AnalysisResult:
        """
        Run the complete EUDR compliance pipeline on a single polygon.

        Parameters
        ----------
        polygon:
            GeoJSON polygon for the production plot.
        generate_pdf:
            Generate a PDF Due Diligence Statement.
        analyze_buffer:
            Also run change detection on the 5 km buffer zone.

        Returns
        -------
        AnalysisResult with compliance determination, council analysis, and optional PDF path.
        """
        self._ensure_loaded()

        # ── 1. Satellite data ──────────────────────────────────────────────────
        logger.info("[1/6] Fetching satellite data …")
        raw = self._acquirer.fetch_baseline_and_current(polygon)
        baseline = raw["baseline_2020"]
        current = raw["current"]
        sar_current = raw.get("sar_current")
        wri_driver  = raw.get("wri_driver")   # full driver dict
        tree_cover  = raw.get("tree_cover")   # forest baseline dict
        crop_map    = raw.get("crop_map")     # JRC EUCROPMAP V1 crop type dict
        metadata = raw["metadata"]

        # ── 2. Preprocessing ───────────────────────────────────────────────────
        logger.info("[2/6] Preprocessing imagery (NDVI + SAR fusion) …")
        processed = self._preprocessor.prepare(
            baseline,
            current,
            sar_current=sar_current,
        )
        _log_ndvi_summary(processed.ndvi_change)

        # ── 3. Change detection ────────────────────────────────────────────────
        logger.info("[3/6] Running Prithvi-100M change detection …")
        self._detector.eval()
        with torch.no_grad():
            output = self._detector(
                processed.t1_tensor,
                processed.t2_tensor,
                ndvi_change=processed.ndvi_change,
            )
        metrics = self._engine.compute_metrics(output.change_map)
        logger.info(
            "  Change: %.2f%%  (%.2f ha, confidence=%.3f)",
            metrics.change_percentage,
            metrics.loss_area_hectares,
            metrics.change_probability,
        )

        # ── 4. Buffer analysis ─────────────────────────────────────────────────
        buffer_metrics = None
        if analyze_buffer:
            logger.info("[4/6] Analysing 5 km buffer zone …")
            buffer_metrics = self._analyze_buffer(polygon)

        # ── 5. Compliance engine ───────────────────────────────────────────────
        logger.info("[5/6] Computing EUDR compliance …")
        compliance = self._engine.classify(
            metrics,
            buffer_metrics,
            wri_driver=wri_driver,
            tree_cover=tree_cover,
            crop_map=crop_map,
        )
        evidence_hash = self._engine.generate_evidence_hash(polygon, metadata)
        logger.info("  Result: %s (%s)", compliance.status.value, compliance.color.value)

        # ── 6. Council deliberation ────────────────────────────────────────────
        logger.info("[6/6] Council deliberation (Gemma) …")
        council = self._council.deliberate(
            metrics=metrics,
            coordinates=polygon.coordinates[0][0],
            wri_driver=wri_driver,
            tree_cover=tree_cover,
        )

        result = AnalysisResult(
            polygon=polygon,
            metrics=metrics,
            buffer_metrics=buffer_metrics,
            compliance=compliance,
            council=council,
            metadata=metadata,
            evidence_hash=evidence_hash,
        )

        # ── PDF generation ─────────────────────────────────────────────────────
        if generate_pdf:
            logger.info("Generating PDF Due Diligence Statement …")
            pdf_path = self._reporter.generate(result, processed.diff_visualization)
            result.pdf_path = str(pdf_path)

        return result

    def _ensure_loaded(self) -> None:
        if self._detector is None:
            self._load_models()

    def _load_models(self) -> None:
        logger.info("Loading models …")
        self._acquirer = SatelliteDataAcquisition()
        self._detector = PrithviChangeDetector(freeze_backbone=True)

        # Apply LoRA fine-tuning checkpoint if provided
        if self._lora_checkpoint:
            from eudr.models.fine_tuning import FineTuner
            self._detector.apply_lora()
            FineTuner.load_checkpoint(self._detector, self._lora_checkpoint)
            logger.info("Fine-tuned LoRA checkpoint loaded: %s", self._lora_checkpoint)

        self._council = ComplianceCouncil()
        logger.info("All models ready.")

    def _analyze_buffer(self, polygon: GeoJSONPolygon):
        """Run the change-detection pipeline on the 5 km buffer zone."""
        try:
            buffer_poly = self._engine.build_buffer_polygon(polygon)
            raw_buf = self._acquirer.fetch_baseline_and_current(buffer_poly)
            proc_buf = self._preprocessor.prepare(
                raw_buf["baseline_2020"],
                raw_buf["current"],
                sar_current=raw_buf.get("sar_current"),
            )
            with torch.no_grad():
                out_buf = self._detector(
                    proc_buf.t1_tensor,
                    proc_buf.t2_tensor,
                    ndvi_change=proc_buf.ndvi_change,
                )
            return self._engine.compute_metrics(out_buf.change_map)
        except Exception as exc:
            logger.warning("Buffer analysis skipped: %s", exc)
            return None


# ── Helper ────────────────────────────────────────────────────────────────────

def _log_ndvi_summary(ndvi_change: "np.ndarray") -> None:  # type: ignore[name-defined]
    """Log NDVI change statistics for diagnostics."""
    try:
        import numpy as np  # noqa: PLC0415
        loss_pct = float(np.mean(ndvi_change < -0.1) * 100)
        gain_pct = float(np.mean(ndvi_change > 0.1) * 100)
        logger.info(
            "  NDVI summary — loss pixels: %.1f%%  gain pixels: %.1f%%",
            loss_pct, gain_pct,
        )
    except Exception:
        pass
