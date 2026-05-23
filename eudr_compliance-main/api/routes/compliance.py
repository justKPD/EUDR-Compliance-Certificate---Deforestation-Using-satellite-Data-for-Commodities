"""
/verify  – EUDR compliance verification endpoint.

POST /verify
  Body : VerifyRequest  (GeoJSON polygon + options)
  Returns: VerifyResponse (compliance status + evidence hash)

GET /reports/{filename}
  Streams the generated PDF report back to the client.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from eudr.config import settings
from eudr.exceptions import (
    DataAcquisitionError,
    ModelInferenceError,
    ReportGenerationError,
)
from eudr.compliance.engine import EUDRComplianceEngine
from eudr.schemas import VerifyRequest, VerifyResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["compliance"])


@router.post("/verify", response_model=VerifyResponse, summary="Verify EUDR compliance for a polygon")
async def verify(request: VerifyRequest, background_tasks: BackgroundTasks) -> VerifyResponse:
    """
    Submit a GeoJSON polygon representing a farm / production plot and receive
    an automated EUDR compliance determination.

    The system will:
    1. Fetch Sentinel-2 imagery for 2020 (baseline) and the most recent year.
    2. Run the Prithvi-100M change-detection model.
    3. Classify compliance according to EUDR Article 3.
    4. Optionally generate a PDF Due Diligence Statement.
    """
    # Import here to avoid heavy GPU allocation at module load time
    from api.main import get_system

    system = get_system()

    try:
        result = system.analyze(
            polygon=request.polygon,
            generate_pdf=request.generate_pdf,
        )
    except DataAcquisitionError as exc:
        raise HTTPException(status_code=422, detail=f"Satellite data unavailable: {exc}") from exc
    except ModelInferenceError as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    except ReportGenerationError as exc:
        logger.warning("PDF generation failed (non-fatal): %s", exc)
        result.pdf_path = None  # degrade gracefully
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pdf_url: str | None = None
    if result.pdf_path:
        filename = Path(result.pdf_path).name
        pdf_url = f"/api/v1/reports/{filename}"

    # Extract WRI driver context from metadata (populated by SatelliteDataAcquisition)
    # metadata["wri_driver"] is a string label (e.g. "Permanent agriculture")
    meta = result.metadata or {}
    wri_driver_label  = meta.get("wri_driver") if isinstance(meta.get("wri_driver"), str) else None
    wri_driver_class  = meta.get("wri_driver_class")
    is_eudr_driver    = meta.get("is_eudr_driver")
    wri_driver_source = meta.get("wri_driver_source")
    mean_canopy_pct        = meta.get("mean_canopy_pct")
    mature_forest_pct      = meta.get("mature_forest_pct")
    is_eudr_forest         = meta.get("is_eudr_forest")
    hansen_post_eudr_loss  = meta.get("post_eudr_loss_pct")

    # JRC EUCROPMAP V1 fields
    crop_class_2018               = meta.get("crop_class_2018")
    crop_label_2018               = meta.get("crop_label_2018")
    crop_class_2022               = meta.get("crop_class_2022")
    crop_label_2022               = meta.get("crop_label_2022")
    crop_is_eudr_commodity        = meta.get("crop_is_eudr_commodity")
    crop_is_pre_existing          = meta.get("crop_is_pre_existing_agricultural")
    crop_is_continuous            = meta.get("crop_is_continuous_agricultural")

    buffer_change_pct: float | None = None
    if result.buffer_metrics:
        buffer_change_pct = result.buffer_metrics.change_percentage

    # Compute determination confidence using the three-signal composite
    tree_cover_dict = {
        "post_eudr_loss_pct": hansen_post_eudr_loss,
        "is_eudr_forest":     is_eudr_forest,
    }
    wri_dict = {
        "is_eudr_driver": meta.get("is_eudr_driver"),
        "driver_class":   meta.get("wri_driver_class"),
    }
    det_conf = EUDRComplianceEngine.compute_determination_confidence(
        result.metrics, tree_cover=tree_cover_dict, wri_driver=wri_dict
    )

    return VerifyResponse(
        compliance_status=result.compliance.status,
        color=result.compliance.color,
        change_percentage=result.metrics.change_percentage,
        loss_area_hectares=result.metrics.loss_area_hectares,
        change_probability=result.metrics.change_probability,
        reason=result.compliance.reason,
        evidence_hash=result.evidence_hash,
        timestamp=result.timestamp,
        # WRI driver
        wri_driver_label=wri_driver_label,
        wri_driver_class=wri_driver_class,
        is_eudr_driver=is_eudr_driver,
        wri_driver_source=wri_driver_source,
        # Forest baseline
        mean_canopy_pct=mean_canopy_pct,
        mature_forest_pct=mature_forest_pct,
        is_eudr_forest=is_eudr_forest,
        hansen_post_eudr_loss_pct=hansen_post_eudr_loss,
        determination_confidence=det_conf,
        # Buffer zone
        buffer_change_percentage=buffer_change_pct,
        # Council analysis
        technical_note=result.council.technical if result.council else None,
        legal_note=result.council.legal if result.council else None,
        council_narrative=result.council.narrative if result.council else None,
        # Commodity context
        commodity=request.commodity,
        # JRC EUCROPMAP V1
        crop_class_2018=crop_class_2018,
        crop_label_2018=crop_label_2018,
        crop_class_2022=crop_class_2022,
        crop_label_2022=crop_label_2022,
        crop_is_eudr_commodity=crop_is_eudr_commodity,
        crop_is_pre_existing_agricultural=crop_is_pre_existing,
        crop_is_continuous_agricultural=crop_is_continuous,
        # Report
        pdf_url=pdf_url,
    )


@router.get(
    "/reports/{filename}",
    summary="Download a generated PDF compliance report",
    response_class=FileResponse,
)
async def get_report(filename: str) -> FileResponse:
    """Return a previously generated PDF Due Diligence Statement."""
    # Sanitise filename to prevent path traversal
    safe_name = Path(filename).name
    report_path = settings.report_output_dir / safe_name

    if not report_path.exists() or report_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@router.get("/health", summary="System health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}
