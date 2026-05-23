"""
Integration tests for the FastAPI compliance endpoint.

The EUDRComplianceSystem is mocked so no GPU / network is needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from eudr.schemas import (
    AnalysisResult,
    ChangeMetrics,
    ComplianceColor,
    ComplianceResult,
    ComplianceStatus,
    CouncilAnalysis,
    GeoJSONPolygon,
)


@pytest.fixture
def mock_result() -> AnalysisResult:
    poly = GeoJSONPolygon(
        coordinates=[[[-5.567, 7.234], [-5.567, 7.244], [-5.557, 7.244], [-5.557, 7.234], [-5.567, 7.234]]]
    )
    metrics = ChangeMetrics(
        change_probability=0.02,
        change_percentage=2.0,
        loss_area_m2=4_000.0,
        loss_area_hectares=0.4,
    )
    compliance = ComplianceResult(
        status=ComplianceStatus.COMPLIANT,
        color=ComplianceColor.GREEN,
        reason="No significant deforestation detected.",
    )
    council = CouncilAnalysis(
        technical="Low spectral change; likely seasonal variation.",
        legal="Compliant with EUDR Article 3.",
        narrative="Based on satellite analysis, the polygon is compliant.",
    )
    return AnalysisResult(
        polygon=poly,
        metrics=metrics,
        compliance=compliance,
        council=council,
        metadata={"baseline_date": "2020-12-31", "current_date": "2024-12-31"},
        evidence_hash="abc123def456abc1",
        timestamp=datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def client(mock_result):
    with patch("eudr.pipeline.EUDRComplianceSystem") as MockSystem:
        instance = MockSystem.return_value
        instance.analyze.return_value = mock_result
        instance._ensure_models_loaded = MagicMock()

        # Patch the singleton getter inside the API
        with patch("api.main.get_system", return_value=instance):
            from api.main import app
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestVerifyEndpoint:
    _payload = {
        "polygon": {
            "type": "Polygon",
            "coordinates": [[
                [-5.567, 7.234], [-5.567, 7.244], [-5.557, 7.244],
                [-5.557, 7.234], [-5.567, 7.234],
            ]],
        },
        "generate_pdf": False,
    }

    def test_verify_returns_200(self, client):
        resp = client.post("/api/v1/verify", json=self._payload)
        assert resp.status_code == 200

    def test_verify_response_schema(self, client):
        resp = client.post("/api/v1/verify", json=self._payload)
        body = resp.json()
        assert body["compliance_status"] == "COMPLIANT"
        assert body["color"] == "GREEN"
        assert body["change_percentage"] == pytest.approx(2.0)
        assert body["loss_area_hectares"] == pytest.approx(0.4)
        assert "evidence_hash" in body
        assert "timestamp" in body

    def test_verify_invalid_polygon_returns_422(self, client):
        resp = client.post("/api/v1/verify", json={"polygon": {"type": "Point", "coordinates": [0, 0]}})
        assert resp.status_code == 422


class TestReportEndpoint:
    def test_missing_report_returns_404(self, client):
        resp = client.get("/api/v1/reports/nonexistent_file.pdf")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, client):
        resp = client.get("/api/v1/reports/../secrets.env")
        # The route sanitises the filename, so we get 404 (not found), not 200
        assert resp.status_code == 404
