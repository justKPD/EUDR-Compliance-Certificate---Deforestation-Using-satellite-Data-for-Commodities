"""
EUDR Compliance Engine

Implements the three-tier risk classification defined by EUDR Article 3:

  GREEN  (COMPLIANT)     – No detectable forest loss
  YELLOW (MEDIUM RISK)   – Loss in buffer zone; or natural-cause loss; or non-forest
                           baseline; or non-commodity land-use change (mining, infra)
  RED    (NON-COMPLIANT) – >5% forest loss from a commodity/logging driver,
                           or unknown driver (precautionary)

Driver classification uses WRI GDM (1 km v1.2, 2001-2024) with Hansen GFC 2024
forest-baseline corroboration.  Only WRI driver classes 1 (Permanent agriculture),
3 (Shifting cultivation), and 4 (Logging) are EUDR Article 3 commodity drivers.
Classes 2 (Mining), 5 (Wildfire), 6 (Settlements/infrastructure), and 7 (Other
natural disturbances) are explicitly outside EUDR scope and route to YELLOW.

Source: EUDR Regulation EU 2023/1115 Article 2(4), Article 3, Annex I.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
import torch
from typing import Optional

from eudr.config import settings
from eudr.exceptions import ComplianceEngineError
from eudr.regulations import (
    EUCROPMAP_EUDR_CLASSES,
    EUDR_NATURAL_DRIVERS,
    EUDR_NON_COMMODITY_DRIVERS,
    driver_legal_rationale,
)
from eudr.schemas import (
    ChangeMetrics,
    ComplianceColor,
    ComplianceResult,
    ComplianceStatus,
    GeoJSONPolygon,
)

logger = logging.getLogger(__name__)

# Pixel area at 10 m resolution
PIXEL_AREA_M2: float = 10.0 * 10.0  # 100 m²

# EUDR Article 2(4): forest = trees able to reach >5 m, ≥10% canopy cover
FOREST_CANOPY_MIN_PCT: int = 10


class EUDRComplianceEngine:
    """
    Stateless compliance logic.  All thresholds come from application settings
    so they can be tuned without code changes.
    """

    # ── Metrics ────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        change_map: torch.Tensor,
        threshold: float = 0.35,
    ) -> ChangeMetrics:
        """
        Derive area-based deforestation statistics from the model output.

        Parameters
        ----------
        change_map:
            Tensor [B, 1, H, W] with change probabilities in [0, 1].
        threshold:
            Pixel-level probability threshold for binary classification.
        """
        if change_map.ndim != 4:
            raise ComplianceEngineError(
                f"Expected 4-D change_map [B,1,H,W], got shape {tuple(change_map.shape)}"
            )
        change_map_cpu = change_map.float().cpu()
        binary = (change_map_cpu > threshold).float()

        change_pct = binary.mean().item() * 100.0
        loss_area_m2 = binary.sum().item() * PIXEL_AREA_M2

        return ChangeMetrics(
            change_probability=change_map_cpu.mean().item(),
            change_percentage=change_pct,
            loss_area_m2=loss_area_m2,
            loss_area_hectares=loss_area_m2 / 10_000.0,
            threshold=threshold,
        )

    # ── Determination confidence ───────────────────────────────────────────────

    @staticmethod
    def compute_determination_confidence(
        metrics: "ChangeMetrics",
        tree_cover: Optional[dict] = None,
        wri_driver: Optional[dict] = None,
    ) -> float:
        """
        Return a 0–100 score expressing how certain the RED/YELLOW/GREEN
        determination is, based on three independent signals:

          1. Distance from the 5% EUDR threshold (dominant factor, 45%)
             → farther from threshold = more certain
          2. NDVI signal clarity (25%)
             → very low mean prob = clean no-change signal (COMPLIANT confidence)
             → high mean prob = clear change signal (NON-COMPLIANT confidence)
          3. Hansen GFC 2024 post-EUDR corroboration (20%)
             → Hansen lossyear 21-24 agreeing with NDVI boosts confidence;
               disagreement lowers it
          4. WRI driver classification (10%)
             → confirmed EUDR commodity driver + loss → stronger RED confidence
             → natural/no driver + no loss → stronger GREEN confidence

        Scores typically look like:
          - Germany farmland (0% loss)         → ~80–90%  (very certain GREEN)
          - Kazakhstan steppe (4.1% loss)      → ~40–50%  (borderline, close to 5%)
          - Kalimantan palm oil (15% loss)     → ~85–95%  (very certain RED)
        """
        threshold = 5.0
        change_pct  = metrics.change_percentage
        mean_prob   = metrics.change_probability

        # 1. Distance from threshold
        margin = abs(change_pct - threshold)
        distance_score = min(1.0, margin / threshold)   # saturates at 2× threshold away

        # 2. NDVI signal clarity
        if change_pct < threshold:       # determination = COMPLIANT
            # mean_prob baseline for intact forest is ~0.18 (sigmoid offset).
            # Values significantly below 0.18 = very clean; near 0.30 = borderline.
            ndvi_score = max(0.0, 1.0 - (mean_prob / 0.30))
        else:                            # determination = NON-COMPLIANT / YELLOW
            ndvi_score = min(1.0, mean_prob / 0.35)

        # 3. Hansen GFC post-EUDR corroboration
        post_eudr_loss = (tree_cover or {}).get("post_eudr_loss_pct")
        if post_eudr_loss is None:
            hansen_score = 0.5           # unknown → neutral
        elif change_pct < threshold and post_eudr_loss < threshold:
            hansen_score = 0.85          # both NDVI and Hansen say no post-2020 loss
        elif change_pct >= threshold and post_eudr_loss >= threshold:
            hansen_score = 0.90          # both confirm post-2020 loss
        else:
            # NDVI and Hansen disagree (one above, one below 5% threshold)
            hansen_score = 0.25

        # 4. WRI driver
        is_eudr_drv    = (wri_driver or {}).get("is_eudr_driver")
        driver_class   = (wri_driver or {}).get("driver_class", 0)
        is_natural     = driver_class in EUDR_NATURAL_DRIVERS         # classes 5, 7
        is_non_eudr_lu = driver_class in EUDR_NON_COMMODITY_DRIVERS   # classes 2, 6
        if is_eudr_drv is True and change_pct >= threshold:
            wri_score = 0.90   # confirmed commodity driver + loss → strong RED
        elif (is_natural or is_non_eudr_lu) and change_pct >= threshold:
            wri_score = 0.80   # known non-EUDR cause → strong YELLOW determination
        elif is_eudr_drv is False and change_pct < threshold:
            wri_score = 0.75   # no EUDR driver + no loss → cleaner GREEN
        else:
            wri_score = 0.50   # unknown or mixed

        confidence = (
            0.45 * distance_score +
            0.25 * ndvi_score +
            0.20 * hansen_score +
            0.10 * wri_score
        ) * 100.0

        return round(min(99.0, max(5.0, confidence)), 1)

    # ── Classification ─────────────────────────────────────────────────────────

    def classify(
        self,
        metrics: ChangeMetrics,
        buffer_metrics: Optional[ChangeMetrics] = None,
        wri_driver: Optional[dict] = None,
        tree_cover: Optional[dict] = None,
        crop_map: Optional[dict] = None,
    ) -> ComplianceResult:
        """
        Apply EUDR Article 3 thresholds with WRI driver and forest-baseline context.

        Classification logic
        --------------------
        1.  If detected loss > 5% AND area was forested (EUDR Article 2):
              - Commodity/logging driver (WRI class 1, 3, 4) → RED
              - Natural driver (wildfire, class 5/7)         → YELLOW (not EUDR)
              - Unknown driver                               → RED (precautionary)

        2.  If detected loss > 5% but area was NOT forested baseline:
              → YELLOW with note (land-cover change, not deforestation per EUDR)

        3.  If buffer zone shows > 20% loss:
              → YELLOW (leakage risk)

        4.  Otherwise → GREEN

        Parameters
        ----------
        metrics:       Change metrics for the production polygon.
        buffer_metrics: Change metrics for the 5 km buffer zone (optional).
        wri_driver:    Dict from SatelliteDataAcquisition._get_wri_driver().
        tree_cover:    Dict from SatelliteDataAcquisition._get_tree_cover().
        """
        direct_pct       = metrics.change_percentage
        direct_threshold = settings.direct_loss_threshold
        buffer_threshold = settings.buffer_loss_threshold

        # ── Extract driver context early — needed to guard the Hansen override ──
        # Driver must be known BEFORE the Hansen GFC override so we don't
        # escalate natural events or infrastructure clearing to RED.
        driver_class   = (wri_driver or {}).get("driver_class", 0)
        driver_label   = (wri_driver or {}).get("driver_label", "Unknown")
        driver_source  = (wri_driver or {}).get("source", "unknown")
        is_eudr_drv    = (wri_driver or {}).get("is_eudr_driver")   # True/False/None
        is_natural     = driver_class in EUDR_NATURAL_DRIVERS        # wildfire / other natural (5, 7)
        is_non_eudr_lu = driver_class in EUDR_NON_COMMODITY_DRIVERS  # mining / settlements (2, 6)

        # ── Hansen GFC 2024 independent corroboration ──────────────────────────
        # If the Hansen lossyear band (2021–2024) shows confirmed post-EUDR forest
        # loss above the threshold AND the area was forested AND the driver is
        # commodity-relevant (or unknown), upgrade direct_pct toward RED.
        #
        # Guard: do NOT apply the override when the driver is already known to be
        # a non-EUDR cause (natural disturbance or infrastructure/mining).
        # Bark beetle die-off and hospital construction both produce valid Hansen
        # loss signals that must NOT escalate to RED simply because the signal is
        # strong — the CAUSE is what determines EUDR applicability.
        post_eudr_loss_pct = (tree_cover or {}).get("post_eudr_loss_pct")
        is_eudr_forest_pre = (tree_cover or {}).get("is_eudr_forest")
        if (
            post_eudr_loss_pct is not None
            and post_eudr_loss_pct > direct_threshold
            and is_eudr_forest_pre is not False
            and direct_pct > direct_threshold * 0.5   # NDVI must show at least half the threshold
            and not is_natural       # never escalate wildfire / bark beetle (classes 5, 7)
            and not is_non_eudr_lu   # never escalate infra / mining (classes 2, 6)
        ):
            logger.info(
                "Hansen GFC override: %.2f%% confirmed post-2020 loss (lossyear 21-24), "
                "driver=%s — escalating direct_pct",
                post_eudr_loss_pct, driver_label,
            )
            # Upgrade to RED if NDVI was borderline; if NDVI already says RED this is no-op
            direct_pct = max(direct_pct, post_eudr_loss_pct)
        elif post_eudr_loss_pct is not None and post_eudr_loss_pct > direct_threshold:
            if is_natural or is_non_eudr_lu:
                logger.info(
                    "Hansen GFC override suppressed: %.2f%% loss confirmed but driver is "
                    "'%s' (non-EUDR) — will route to YELLOW, not RED",
                    post_eudr_loss_pct, driver_label,
                )

        is_eudr_forest = (tree_cover or {}).get("is_eudr_forest")   # True/False/None
        canopy_pct     = (tree_cover or {}).get("mean_canopy_pct")
        mature_pct     = (tree_cover or {}).get("mature_forest_pct")

        # ── JRC EUCROPMAP context ──────────────────────────────────────────────
        # Confirms whether the land was already under agricultural use before the
        # EUDR reference date (31 December 2020).  EUDR Article 2(13) defines
        # "deforestation-free" as not subject to deforestation AFTER that date.
        # If EUCROPMAP 2018 shows the polygon was arable cropland, any NDVI signal
        # detected comparing 2020 and 2024 imagery is crop rotation — not forest loss.
        _crop             = crop_map or {}
        crop_pre_existing = _crop.get("is_pre_existing_agricultural")   # True / False / None
        crop_continuous   = _crop.get("is_continuous_agricultural")     # True / False / None
        crop_is_eudr      = _crop.get("is_eudr_commodity")              # True (soya 233) / False / None
        crop_class_2022   = _crop.get("class_2022")
        crop_label_2022   = _crop.get("label_2022", "Unknown")
        crop_class_2018   = _crop.get("class_2018")
        crop_label_2018   = _crop.get("label_2018", "Unknown")

        # When WRI data is used, flag its 1 km resolution so operators know this
        # single pixel may cover a wider area than the production polygon itself.
        wri_resolution_caveat = (
            " Note: WRI GDM driver data has 1 km resolution — the driver "
            "classification reflects the dominant land-use within a 1 km² cell "
            "and may not precisely represent a small production polygon."
            if driver_source == "wri" else ""
        )
        driver_note = (
            f" Deforestation driver: {driver_label}"
            f" (WRI GDM class {driver_class}"
            + (f", source: {driver_source}" if driver_source not in ("wri", "unknown") else "")
            + ")."
            + wri_resolution_caveat
        ) if driver_class else ""

        forest_note = ""
        if canopy_pct is not None:
            forest_note = (
                f" Baseline tree cover: {canopy_pct:.1f}% canopy"
                f" ({mature_pct:.1f}% at ≥30% density, proxy for mature canopy ≥15 m)."
            )

        # ── Direct loss detected ───────────────────────────────────────────────
        if direct_pct > direct_threshold:

            # Area was not forested at baseline → land-cover change, not deforestation
            if is_eudr_forest is False:
                return ComplianceResult(
                    status=ComplianceStatus.MEDIUM_RISK,
                    color=ComplianceColor.YELLOW,
                    reason=(
                        f"Land-cover change of {direct_pct:.2f}% detected, but the baseline "
                        f"tree cover ({canopy_pct:.1f}%) is below the EUDR forest threshold "
                        f"of {FOREST_CANOPY_MIN_PCT}% canopy cover (Article 2(4)). "
                        f"This may not constitute deforestation under EUDR."
                        f"{driver_note}{forest_note}"
                    ),
                )

            # Natural disturbance (wildfire / other natural) → not an EUDR violation
            if is_natural:
                return ComplianceResult(
                    status=ComplianceStatus.MEDIUM_RISK,
                    color=ComplianceColor.YELLOW,
                    reason=(
                        f"Forest loss of {direct_pct:.2f}% detected inside the polygon, "
                        f"but attributed to natural disturbance ({driver_label}). "
                        f"EUDR Article 3 applies to commodity-driven deforestation — "
                        f"natural events are not a direct violation. "
                        f"Enhanced due diligence documentation is still required."
                        f"{forest_note}"
                    ),
                )

            # Non-commodity land-use change (mining class 2; settlements/infra class 6)
            # → NOT regulated under EUDR Annex I, no Article 3 violation
            if is_non_eudr_lu:
                eudr_rationale = driver_legal_rationale(driver_class)
                return ComplianceResult(
                    status=ComplianceStatus.MEDIUM_RISK,
                    color=ComplianceColor.YELLOW,
                    reason=(
                        f"Land-cover change of {direct_pct:.2f}% detected inside the polygon "
                        f"and attributed to '{driver_label}' (WRI GDM class {driver_class}). "
                        f"{eudr_rationale} "
                        f"No EUDR Article 3 non-compliance. "
                        f"Operator should retain documentation of this land-use change "
                        f"for due diligence records."
                        f"{forest_note}"
                    ),
                )

            # ── JRC EUCROPMAP pre-existing agricultural land check ─────────────
            # EUDR Article 2(13): "deforestation-free" means no deforestation AFTER
            # 31 December 2020.  EUCROPMAP 2018 is a pre-EUDR baseline snapshot.
            # If the polygon was already classified as arable cropland in 2018, the
            # land was unambiguously agricultural BEFORE the reference date.
            #
            # Under these conditions any spectral NDVI change detected by comparing
            # 2020-baseline imagery with 2024 current imagery is crop rotation
            # (e.g. winter wheat → maize → fallow → rapeseed), not post-2020
            # forest conversion.
            #
            # Exception: if the 2022 layer shows soya (class 233, the only EUDR
            # Annex I crop in EUCROPMAP) and the 2018 layer did NOT show soya,
            # the area may have expanded into soya post-EUDR — this specific case
            # passes through to the RED path for further scrutiny.
            if crop_pre_existing is True and not (crop_is_eudr and crop_class_2018 not in EUCROPMAP_EUDR_CLASSES):
                crop_note = (
                    f"2018: {crop_label_2018} → 2022: {crop_label_2022}"
                    if crop_class_2022 is not None
                    else "2018 cropland confirmed"
                )
                logger.info(
                    "EUCROPMAP: pre-existing agricultural land (%s) — NDVI change %.2f%% "
                    "is crop rotation, not post-2020 deforestation → YELLOW",
                    crop_note, direct_pct,
                )
                return ComplianceResult(
                    status=ComplianceStatus.MEDIUM_RISK,
                    color=ComplianceColor.YELLOW,
                    reason=(
                        f"Spectral change of {direct_pct:.2f}% detected inside the polygon "
                        f"(above the {direct_threshold}% threshold), but JRC EU CropMap V1 "
                        f"(EUCROPMAP) confirms this land was already under agricultural use "
                        f"in 2018 — BEFORE the EUDR reference date of 31 December 2020. "
                        f"Crop map: {crop_note}. "
                        f"Under EUDR Article 2(13), deforestation-free means no forest was cleared "
                        f"AFTER 31 December 2020; pre-existing arable land does not constitute "
                        f"deforestation. The detected spectral change is consistent with "
                        f"seasonal crop rotation on established farmland."
                        f"{forest_note}{driver_note}"
                    ),
                )

            # ── Hansen GFC contradiction check ─────────────────────────────────
            # If Hansen GFC 2024 explicitly confirms near-zero post-EUDR loss
            # (<2% of pixels show loss 2021-2024) while NDVI shows >5% change,
            # the spectral signal is almost certainly noise — seasonal NDVI
            # variation, crop rotation, phenological timing differences between
            # the 2020 and 2024 imagery acquisition dates.
            #
            # This is the decisive tell for temperate-Europe false positives:
            # if the UMD/Hansen dataset (Landsat 30 m, peer-reviewed) says
            # no forest was cleared post-2020, we cannot in good conscience
            # declare NON-COMPLIANT based on a borderline NDVI signal alone.
            #
            # Guard: only applies when the driver is NOT confirmed as a commodity
            # supply-chain driver (is_eudr_drv is not True). If WRI confirms
            # genuine agriculture/logging AND Hansen also shows near-zero loss
            # (unusual but possible in marginal cases), we stay RED.
            _HANSEN_NEAR_ZERO = direct_threshold * 0.4   # <2.0% = "effectively no loss"
            hansen_contradicts_ndvi = (
                post_eudr_loss_pct is not None
                and post_eudr_loss_pct < _HANSEN_NEAR_ZERO
                and is_eudr_drv is not True   # not a confirmed commodity driver
            )
            if hansen_contradicts_ndvi:
                logger.info(
                    "Hansen GFC contradicts NDVI: NDVI=%.2f%% but Hansen post-2020 loss=%.2f%% "
                    "(<%.1f%% threshold) — downgrading to YELLOW (conflicting spectral signals)",
                    direct_pct, post_eudr_loss_pct, _HANSEN_NEAR_ZERO,
                )
                return ComplianceResult(
                    status=ComplianceStatus.MEDIUM_RISK,
                    color=ComplianceColor.YELLOW,
                    reason=(
                        f"NDVI spectral change of {direct_pct:.2f}% detected inside the polygon "
                        f"(above the {direct_threshold}% threshold), but this signal is NOT "
                        f"corroborated by Hansen GFC 2024: only {post_eudr_loss_pct:.2f}% of "
                        f"pixels show confirmed post-2020 forest loss in the satellite ground-truth "
                        f"dataset. The spectral difference likely reflects seasonal vegetation "
                        f"variation, crop rotation, or phenological timing differences between "
                        f"the 2020 baseline and 2024 current imagery — not actual deforestation. "
                        f"Non-compliance cannot be confirmed without corroborating evidence. "
                        f"Enhanced due diligence documentation is recommended under EUDR Article 9."
                        f"{forest_note}{driver_note}"
                    ),
                )

            # Commodity-driven or unknown driver → RED (precautionary for unknown)
            eudr_context = (
                f"Driver identified as '{driver_label}' — a commodity supply-chain "
                f"activity directly regulated under EUDR Regulation EU 2023/1115."
                if is_eudr_drv
                else (
                    f"Driver: {driver_label}. Unable to confirm commodity link — "
                    f"applying precautionary principle."
                    if driver_class
                    else "Deforestation driver could not be determined — precautionary RED applied."
                )
            )
            return ComplianceResult(
                status=ComplianceStatus.NON_COMPLIANT,
                color=ComplianceColor.RED,
                reason=(
                    f"Forest loss of {direct_pct:.2f}% detected inside the production polygon "
                    f"(EUDR threshold: {direct_threshold}%). "
                    f"{eudr_context} "
                    f"This constitutes a violation of EUDR Article 3 with reference "
                    f"date {settings.eudr_cutoff_date}. "
                    f"Operator must suspend placing this commodity on the EU market."
                    f"{forest_note}"
                ),
            )

        # ── Buffer zone leakage risk ───────────────────────────────────────────
        if buffer_metrics and buffer_metrics.change_percentage > buffer_threshold:
            return ComplianceResult(
                status=ComplianceStatus.MEDIUM_RISK,
                color=ComplianceColor.YELLOW,
                reason=(
                    f"No direct deforestation inside polygon ({direct_pct:.2f}%), but "
                    f"{buffer_metrics.change_percentage:.2f}% forest loss detected in the "
                    f"{settings.buffer_radius_km} km buffer zone "
                    f"(threshold: {buffer_threshold}%). "
                    f"Leakage risk — enhanced due diligence required under EUDR Article 9."
                    f"{driver_note}"
                ),
            )

        # ── Compliant ──────────────────────────────────────────────────────────
        return ComplianceResult(
            status=ComplianceStatus.COMPLIANT,
            color=ComplianceColor.GREEN,
            reason=(
                f"No significant deforestation detected inside the production polygon "
                f"({direct_pct:.2f}% < {direct_threshold}% threshold). "
                f"Commodity appears deforestation-free as required by EUDR Article 3 "
                f"(reference date {settings.eudr_cutoff_date})."
                f"{driver_note}{forest_note}"
            ),
        )

    # ── Evidence hash ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_evidence_hash(
        polygon: GeoJSONPolygon,
        metadata: dict,
        system_version: str = "1.0.0",
    ) -> str:
        """
        Produce a short SHA-256 fingerprint of the polygon + analysis metadata
        for an immutable audit trail.
        """
        payload = json.dumps(
            {
                "coordinates": polygon.coordinates,
                "baseline_date": metadata.get("baseline_date"),
                "current_date": metadata.get("current_date"),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "system_version": system_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    # ── Buffer-zone helper ─────────────────────────────────────────────────────

    @staticmethod
    def build_buffer_polygon(polygon: GeoJSONPolygon) -> GeoJSONPolygon:
        """
        Create a GeoJSON polygon representing the 5 km buffer zone around the
        input polygon using Shapely.

        The approximation of 1° ≈ 111 km is adequate at tropical latitudes.
        """
        try:
            from shapely.geometry import mapping, shape  # local import – optional dep
        except ImportError as exc:
            raise ComplianceEngineError(
                "Shapely is required for buffer analysis – install it with `pip install shapely`."
            ) from exc

        radius_deg = settings.buffer_radius_km / 111.0  # rough degrees
        geom = shape({"type": "Polygon", "coordinates": polygon.coordinates})
        buffered = geom.buffer(radius_deg)
        geo = mapping(buffered)
        return GeoJSONPolygon(coordinates=list(geo["coordinates"]))
