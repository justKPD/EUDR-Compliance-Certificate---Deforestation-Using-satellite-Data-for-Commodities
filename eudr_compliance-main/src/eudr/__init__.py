"""
eudr – Osapiens Deforestation Intelligence System

Top-level imports are intentionally lazy so that importing `eudr` in tests
or scripts does not immediately trigger heavy dependencies (earthengine-api,
torch, transformers) that may not be installed in the current environment.
"""
from __future__ import annotations

__all__ = ["EUDRComplianceSystem", "GeoJSONPolygon", "AnalysisResult"]


def __getattr__(name: str):  # noqa: ANN001, ANN201
    """PEP 562 – lazy module-level attribute lookup."""
    if name == "EUDRComplianceSystem":
        from eudr.pipeline import EUDRComplianceSystem  # noqa: PLC0415
        return EUDRComplianceSystem
    if name in ("GeoJSONPolygon", "AnalysisResult"):
        from eudr import schemas  # noqa: PLC0415
        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
