"""
Pydantic data-transfer objects shared across all layers.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Geometry ──────────────────────────────────────────────────────────────────

class GeoJSONPolygon(BaseModel):
    """Minimal GeoJSON Polygon representation."""

    type: str = Field(default="Polygon", pattern="^Polygon$")
    coordinates: list[list[list[float]]] = Field(
        ...,
        description="Nested list: [ring[point[lon, lat]]]",
        min_length=1,
    )


# ── Satellite imagery ─────────────────────────────────────────────────────────

class SatelliteImagery(BaseModel):
    """Raw numpy arrays fetched from GEE (kept as bytes to cross process boundaries)."""

    baseline_date: str
    current_date: str
    resolution_m: int = 10
    bands: list[str]
    # Serialised as base64 or kept in-memory; heavy data stays out of the API response
    baseline_shape: tuple[int, int, int]
    current_shape: tuple[int, int, int]


# ── Change-detection metrics ──────────────────────────────────────────────────

class ChangeMetrics(BaseModel):
    """Pixel-level deforestation statistics from the change-detection model."""

    change_probability: float = Field(..., ge=0.0, le=1.0)
    change_percentage: float = Field(..., ge=0.0, le=100.0)
    loss_area_m2: float = Field(..., ge=0.0)
    loss_area_hectares: float = Field(..., ge=0.0)
    threshold: float = Field(default=0.5)


# ── Compliance ────────────────────────────────────────────────────────────────

class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    MEDIUM_RISK = "MEDIUM RISK"
    NON_COMPLIANT = "NON-COMPLIANT"


class ComplianceColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ComplianceResult(BaseModel):
    status: ComplianceStatus
    color: ComplianceColor
    reason: str


# ── Council analysis ──────────────────────────────────────────────────────────

class CouncilAnalysis(BaseModel):
    technical: str
    legal: str
    narrative: str


# ── Full analysis result ──────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    polygon: GeoJSONPolygon
    metrics: ChangeMetrics
    buffer_metrics: Optional[ChangeMetrics] = None
    compliance: ComplianceResult
    council: CouncilAnalysis
    metadata: dict[str, Any]
    evidence_hash: str
    pdf_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── API request / response ────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    polygon: GeoJSONPolygon
    generate_pdf: bool = True
    commodity: Optional[str] = Field(
        default=None,
        description=(
            "EUDR-regulated commodity produced in the polygon "
            "(e.g. cocoa, coffee, palm_oil, soy, cattle, wood, rubber). "
            "Included in the PDF report and council narrative."
        ),
    )


class VerifyResponse(BaseModel):
    # ── Core compliance ────────────────────────────────────────────────────────
    compliance_status: ComplianceStatus
    color: ComplianceColor
    change_percentage: float
    loss_area_hectares: float
    change_probability: float
    reason: str
    evidence_hash: str
    timestamp: datetime

    # ── WRI deforestation driver ───────────────────────────────────────────────
    wri_driver_label: Optional[str] = None
    wri_driver_class: Optional[int] = None
    is_eudr_driver: Optional[bool] = None
    wri_driver_source: Optional[str] = None

    # ── Forest baseline (Hansen GFC 2024) ─────────────────────────────────────
    mean_canopy_pct: Optional[float] = None
    mature_forest_pct: Optional[float] = None
    is_eudr_forest: Optional[bool] = None
    hansen_post_eudr_loss_pct: Optional[float] = None   # confirmed loss 2021-2024

    # ── Determination confidence (0-100) ──────────────────────────────────────
    determination_confidence: Optional[float] = Field(
        default=None,
        description=(
            "0–100 score: how certain is the RED/YELLOW/GREEN determination? "
            "Based on distance from 5% threshold + Hansen GFC corroboration + WRI driver. "
            "NOT the same as change_probability (the raw mean deforestation signal)."
        ),
    )

    # ── Buffer zone ────────────────────────────────────────────────────────────
    buffer_change_percentage: Optional[float] = None

    # ── Council analysis ───────────────────────────────────────────────────────
    technical_note: Optional[str] = None
    legal_note: Optional[str] = None
    council_narrative: Optional[str] = None

    # ── Commodity context ──────────────────────────────────────────────────────
    commodity: Optional[str] = None

    # ── JRC EU CropMap V1 (EUCROPMAP) ─────────────────────────────────────────
    # EU-only, 10 m resolution, 2018 + 2022 epochs.
    # Confirms pre-existing agricultural land use vs. post-EUDR forest conversion.
    crop_class_2018: Optional[int] = Field(
        default=None,
        description="EUCROPMAP 2018 modal crop class (pre-EUDR baseline snapshot).",
    )
    crop_label_2018: Optional[str] = Field(
        default=None,
        description="Human-readable EUCROPMAP 2018 crop type label.",
    )
    crop_class_2022: Optional[int] = Field(
        default=None,
        description="EUCROPMAP 2022 modal crop class (post-EUDR snapshot).",
    )
    crop_label_2022: Optional[str] = Field(
        default=None,
        description="Human-readable EUCROPMAP 2022 crop type label.",
    )
    crop_is_eudr_commodity: Optional[bool] = Field(
        default=None,
        description=(
            "True if the 2022 crop is a EUDR Annex I commodity (currently only soya=233 "
            "is present in EUCROPMAP). False for all other crops. None if outside EU."
        ),
    )
    crop_is_pre_existing_agricultural: Optional[bool] = Field(
        default=None,
        description=(
            "True if the 2018 EUCROPMAP classification is an arable class, confirming the "
            "land was already agricultural BEFORE the EUDR reference date (31 Dec 2020). "
            "When True, detected NDVI change is likely crop rotation, not deforestation."
        ),
    )
    crop_is_continuous_agricultural: Optional[bool] = Field(
        default=None,
        description=(
            "True if both 2018 and 2022 EUCROPMAP classifications are arable classes, "
            "confirming continuous crop rotation with no land-cover conversion event."
        ),
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    pdf_url: Optional[str] = None
