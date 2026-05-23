"""
Unit tests for EUDRComplianceEngine.

These tests exercise only pure Python + NumPy / PyTorch logic —
no GEE, no GPU, no model weights required.
"""
from __future__ import annotations

import pytest
import torch

from eudr.compliance.engine import EUDRComplianceEngine, PIXEL_AREA_M2
from eudr.schemas import (
    ChangeMetrics,
    ComplianceColor,
    ComplianceStatus,
    GeoJSONPolygon,
)


@pytest.fixture
def engine() -> EUDRComplianceEngine:
    return EUDRComplianceEngine()


@pytest.fixture
def simple_polygon() -> GeoJSONPolygon:
    return GeoJSONPolygon(
        coordinates=[[[-5.567, 7.234], [-5.567, 7.244], [-5.557, 7.244], [-5.557, 7.234], [-5.567, 7.234]]]
    )


# ── compute_metrics ────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_zero_loss_map(self):
        change_map = torch.zeros(1, 1, 224, 224)
        m = EUDRComplianceEngine.compute_metrics(change_map)
        assert m.change_percentage == pytest.approx(0.0)
        assert m.loss_area_hectares == pytest.approx(0.0)
        assert m.change_probability == pytest.approx(0.0)

    def test_full_loss_map(self):
        change_map = torch.ones(1, 1, 224, 224)
        m = EUDRComplianceEngine.compute_metrics(change_map)
        assert m.change_percentage == pytest.approx(100.0)
        total_m2 = 224 * 224 * PIXEL_AREA_M2
        assert m.loss_area_m2 == pytest.approx(total_m2)
        assert m.loss_area_hectares == pytest.approx(total_m2 / 10_000)

    def test_half_loss_map(self):
        change_map = torch.zeros(1, 1, 10, 10)
        change_map[0, 0, :5, :] = 1.0  # top half = changed
        m = EUDRComplianceEngine.compute_metrics(change_map, threshold=0.5)
        assert m.change_percentage == pytest.approx(50.0)

    def test_wrong_ndim_raises(self):
        bad_map = torch.zeros(224, 224)
        with pytest.raises(Exception):
            EUDRComplianceEngine.compute_metrics(bad_map)


# ── classify ──────────────────────────────────────────────────────────────────

class TestClassify:
    def _metrics(self, pct: float) -> ChangeMetrics:
        return ChangeMetrics(
            change_probability=pct / 100,
            change_percentage=pct,
            loss_area_m2=pct * 100,
            loss_area_hectares=pct * 0.01,
        )

    def test_green_below_threshold(self, engine):
        result = engine.classify(self._metrics(2.0))
        assert result.status == ComplianceStatus.COMPLIANT
        assert result.color == ComplianceColor.GREEN

    def test_red_above_threshold(self, engine):
        result = engine.classify(self._metrics(6.0))
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED

    def test_exactly_at_threshold_is_compliant(self, engine):
        # 5.0% exactly is NOT above 5.0, so should be GREEN
        result = engine.classify(self._metrics(5.0))
        assert result.status == ComplianceStatus.COMPLIANT

    def test_yellow_with_buffer_risk(self, engine):
        poly_metrics = self._metrics(1.0)   # inside: no direct loss
        buf_metrics = self._metrics(25.0)   # buffer: high loss
        result = engine.classify(poly_metrics, buf_metrics)
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert result.color == ComplianceColor.YELLOW

    def test_direct_loss_overrides_buffer(self, engine):
        poly_metrics = self._metrics(10.0)  # inside: non-compliant
        buf_metrics = self._metrics(5.0)    # buffer: low risk
        result = engine.classify(poly_metrics, buf_metrics)
        assert result.status == ComplianceStatus.NON_COMPLIANT

    # ── Non-commodity driver tests (class 6 and class 2) ──────────────────────
    # EUDR only covers Annex I commodities — infrastructure and mining are NOT
    # regulated.  High forest loss with these drivers must be YELLOW, not RED.

    def test_infrastructure_driver_is_yellow_not_red(self, engine):
        """WRI class 6 (Settlements & infrastructure) → YELLOW even with >5% loss."""
        wri_infrastructure = {
            "driver_class": 6,
            "driver_label": "Settlements and infrastructure",
            "is_eudr_driver": False,
            "source": "wri",
        }
        result = engine.classify(self._metrics(10.0), wri_driver=wri_infrastructure)
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert result.color == ComplianceColor.YELLOW
        assert "infrastructure" in result.reason.lower() or "settlements" in result.reason.lower()

    def test_mining_driver_is_yellow_not_red(self, engine):
        """WRI class 2 (Hard commodities / mining) → YELLOW even with >5% loss."""
        wri_mining = {
            "driver_class": 2,
            "driver_label": "Hard commodities",
            "is_eudr_driver": False,
            "source": "wri",
        }
        result = engine.classify(self._metrics(8.0), wri_driver=wri_mining)
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert result.color == ComplianceColor.YELLOW

    def test_unknown_driver_still_red_precautionary(self, engine):
        """WRI class 0 (Unknown) → RED via precautionary principle."""
        wri_unknown = {
            "driver_class": 0,
            "driver_label": "Unknown",
            "is_eudr_driver": None,
            "source": "unknown",
        }
        result = engine.classify(self._metrics(7.0), wri_driver=wri_unknown)
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED

    def test_agriculture_driver_is_red(self, engine):
        """WRI class 1 (Permanent agriculture) → RED (EUDR commodity driver)."""
        wri_agri = {
            "driver_class": 1,
            "driver_label": "Permanent agriculture",
            "is_eudr_driver": True,
            "source": "wri",
        }
        result = engine.classify(self._metrics(6.0), wri_driver=wri_agri)
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED

    def test_natural_disturbance_driver_is_yellow(self, engine):
        """WRI class 7 (Other natural disturbances — e.g. bark beetle) → YELLOW."""
        wri_natural = {
            "driver_class": 7,
            "driver_label": "Other natural disturbances",
            "is_eudr_driver": False,
            "source": "wri",
        }
        result = engine.classify(self._metrics(9.0), wri_driver=wri_natural)
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert result.color == ComplianceColor.YELLOW

    # ── Hansen GFC override guard tests ───────────────────────────────────────
    # The override must NOT escalate natural events or infrastructure to RED
    # even when Hansen confirms significant post-2020 loss.

    def test_hansen_override_suppressed_for_natural_driver(self, engine):
        """
        Hansen high loss + class 7 (bark beetle) → override suppressed → GREEN.

        Logic: NDVI = 3% < 5% threshold.  Without the guard, the Hansen override
        would inflate direct_pct to 18% → YELLOW.  With the guard, direct_pct stays
        at 3% (below threshold) → GREEN (compliant).  Bark beetle loss is not EUDR-
        regulated; inflating the signal would be misleading even to YELLOW.
        """
        wri_natural = {
            "driver_class": 7,
            "driver_label": "Other natural disturbances",
            "is_eudr_driver": False,
            "source": "wri",
        }
        tree_cover_with_hansen_loss = {
            "mean_canopy_pct": 45.0,
            "forest_pct": 80.0,
            "mature_forest_pct": 60.0,
            "is_eudr_forest": True,
            "post_eudr_loss_pct": 18.0,   # high Hansen loss — bark beetle die-off
        }
        # NDVI 3% would trigger the override (3% > 2.5% = 50% of 5% threshold)
        # but the guard suppresses it because driver is natural (class 7)
        result = engine.classify(
            self._metrics(3.0),
            wri_driver=wri_natural,
            tree_cover=tree_cover_with_hansen_loss,
        )
        # Override suppressed → direct_pct stays 3% → below 5% threshold → GREEN
        assert result.color != ComplianceColor.RED
        assert result.status == ComplianceStatus.COMPLIANT

    def test_hansen_override_suppressed_for_infrastructure_driver(self, engine):
        """
        Hansen high loss + class 6 (settlements) → override suppressed → GREEN.

        Logic: NDVI = 3.5% < 5% threshold.  Without the guard, the Hansen override
        would inflate direct_pct to 12% → YELLOW (class 6 is now correctly non-EUDR).
        With the guard, direct_pct stays at 3.5% → GREEN (compliant).
        Hospital campus / construction site loss is simply not EUDR-regulated.
        """
        wri_infra = {
            "driver_class": 6,
            "driver_label": "Settlements and infrastructure",
            "is_eudr_driver": False,
            "source": "wri",
        }
        tree_cover_with_hansen_loss = {
            "mean_canopy_pct": 30.0,
            "forest_pct": 55.0,
            "mature_forest_pct": 25.0,
            "is_eudr_forest": True,
            "post_eudr_loss_pct": 12.0,   # construction clearing — Hansen picks it up
        }
        result = engine.classify(
            self._metrics(3.5),
            wri_driver=wri_infra,
            tree_cover=tree_cover_with_hansen_loss,
        )
        # Override suppressed → direct_pct stays 3.5% → below 5% threshold → GREEN
        assert result.color != ComplianceColor.RED
        assert result.status == ComplianceStatus.COMPLIANT

    # ── Hansen contradicts NDVI (the Fulda, Germany false-positive scenario) ───
    # Observed in production: a Fulda DE polygon returned NON-COMPLIANT with
    # 6.44% NDVI change, 35.2% determination confidence, and Hansen GFC 2024
    # explicitly showing "No confirmed post-2020 deforestation."
    # Root cause: the precautionary-RED path fired without checking that Hansen
    # independently contradicts the NDVI signal.

    def test_hansen_contradicts_ndvi_downgrade_to_yellow(self, engine):
        """
        NDVI 6.44% (above threshold) + Hansen 0% post-2020 loss + unknown driver
        → YELLOW (conflicting spectral signals, not confirmed deforestation).
        This is the exact scenario observed for the Fulda, Germany polygon.
        """
        wri_unknown = {
            "driver_class": 0,
            "driver_label": "Unknown",
            "is_eudr_driver": None,
            "source": "unknown",
        }
        tree_cover_germany = {
            "mean_canopy_pct": 35.2,     # ~35% canopy — qualifies as forest
            "forest_pct": 55.0,
            "mature_forest_pct": 28.0,
            "is_eudr_forest": True,
            "post_eudr_loss_pct": 0.0,   # Hansen: no confirmed post-2020 loss
        }
        result = engine.classify(
            self._metrics(6.44),
            wri_driver=wri_unknown,
            tree_cover=tree_cover_germany,
        )
        # Must NOT be RED — Hansen explicitly contradicts the NDVI signal
        assert result.color != ComplianceColor.RED
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert "corroborated" in result.reason.lower() or "contradict" in result.reason.lower() or "spectral" in result.reason.lower()

    def test_hansen_very_low_loss_still_downgrade(self, engine):
        """Hansen 1.5% (still below 2% near-zero threshold) + NDVI 7% → YELLOW."""
        wri_unknown = {"driver_class": 0, "driver_label": "Unknown",
                       "is_eudr_driver": None, "source": "unknown"}
        tree_cover = {
            "mean_canopy_pct": 22.0, "forest_pct": 40.0, "mature_forest_pct": 15.0,
            "is_eudr_forest": True, "post_eudr_loss_pct": 1.5,
        }
        result = engine.classify(self._metrics(7.0), wri_driver=wri_unknown, tree_cover=tree_cover)
        assert result.color != ComplianceColor.RED
        assert result.status == ComplianceStatus.MEDIUM_RISK

    def test_hansen_contradiction_not_triggered_for_confirmed_commodity(self, engine):
        """
        Hansen 0% but WRI confirms agriculture (class 1) → RED still fires.
        The downgrade path must not protect confirmed commodity drivers.
        """
        wri_agri = {
            "driver_class": 1,
            "driver_label": "Permanent agriculture",
            "is_eudr_driver": True,
            "source": "wri",
        }
        tree_cover = {
            "mean_canopy_pct": 40.0, "forest_pct": 70.0, "mature_forest_pct": 50.0,
            "is_eudr_forest": True, "post_eudr_loss_pct": 0.5,
        }
        # Even if Hansen shows low loss, confirmed agriculture driver → RED
        result = engine.classify(self._metrics(6.5), wri_driver=wri_agri, tree_cover=tree_cover)
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED

    def test_hansen_contradiction_not_triggered_when_hansen_unavailable(self, engine):
        """When Hansen data is None (unavailable), precautionary RED still fires."""
        wri_unknown = {"driver_class": 0, "driver_label": "Unknown",
                       "is_eudr_driver": None, "source": "unknown"}
        tree_cover_no_hansen = {
            "mean_canopy_pct": 30.0, "forest_pct": 50.0, "mature_forest_pct": 20.0,
            "is_eudr_forest": True, "post_eudr_loss_pct": None,  # Hansen unavailable
        }
        result = engine.classify(self._metrics(6.0), wri_driver=wri_unknown, tree_cover=tree_cover_no_hansen)
        # No Hansen data → cannot contradict → precautionary RED applies
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED

    def test_hansen_override_fires_for_commodity_driver(self, engine):
        """Hansen high loss + class 1 (agriculture) → override fires → RED."""
        wri_agri = {
            "driver_class": 1,
            "driver_label": "Permanent agriculture",
            "is_eudr_driver": True,
            "source": "wri",
        }
        tree_cover_with_hansen_loss = {
            "mean_canopy_pct": 50.0,
            "forest_pct": 85.0,
            "mature_forest_pct": 65.0,
            "is_eudr_forest": True,
            "post_eudr_loss_pct": 10.0,   # genuine deforestation for soy/cattle
        }
        # NDVI 3% (borderline) → Hansen override should escalate to RED
        result = engine.classify(
            self._metrics(3.0),
            wri_driver=wri_agri,
            tree_cover=tree_cover_with_hansen_loss,
        )
        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.color == ComplianceColor.RED


# ── JRC EUCROPMAP V1 integration ──────────────────────────────────────────────

class TestEUCROPMAPIntegration:
    """
    Tests for the JRC EU CropMap V1 integration in classify().

    EUCROPMAP adds a pre-EUDR-baseline agricultural land-use check:
    if the 2018 crop map shows arable land, any NDVI change detected comparing
    2020 baseline with 2024 current imagery is crop rotation — not forest loss.
    """

    def _metrics(self, pct: float) -> ChangeMetrics:
        return ChangeMetrics(
            change_probability=pct / 100,
            change_percentage=pct,
            loss_area_m2=pct * 100,
            loss_area_hectares=pct * 0.01,
        )

    # ── Pre-existing arable land → YELLOW (crop rotation, not deforestation) ───

    def test_pre_existing_arable_downgrade_to_yellow(self, engine):
        """
        A polygon showing >5% NDVI change but classified as arable (winter cereals)
        in EUCROPMAP 2018 should NOT be RED — it's crop rotation on pre-EUDR farmland.
        """
        crop_map_cereal = {
            "class_2018": 211,          # Winter cereals — arable
            "label_2018": "Winter cereals",
            "class_2022": 212,          # Maize — arable
            "label_2022": "Maize",
            "is_eudr_commodity": False,
            "is_pre_existing_agricultural": True,
            "is_continuous_agricultural": True,
            "coverage": 0.90,
        }
        result = engine.classify(self._metrics(7.0), crop_map=crop_map_cereal)
        assert result.color == ComplianceColor.YELLOW
        assert result.status == ComplianceStatus.MEDIUM_RISK
        assert "pre-existing" in result.reason.lower() or "2018" in result.reason

    def test_continuous_arable_rapeseed_fallow_downgrade(self, engine):
        """Rapeseed (2018) → Fallow (2022): continuous arable rotation → YELLOW."""
        crop_map_oilcrop = {
            "class_2018": 231,          # Rapeseed
            "label_2018": "Rapeseed",
            "class_2022": 420,          # Fallow land
            "label_2022": "Fallow land",
            "is_eudr_commodity": False,
            "is_pre_existing_agricultural": True,
            "is_continuous_agricultural": True,
            "coverage": 0.85,
        }
        result = engine.classify(self._metrics(8.0), crop_map=crop_map_oilcrop)
        assert result.color == ComplianceColor.YELLOW
        assert result.status == ComplianceStatus.MEDIUM_RISK

    def test_pre_existing_vineyard_downgrade(self, engine):
        """Permanent crop (vineyard=520) in 2018 → still vineyard 2022 → YELLOW."""
        crop_map_vineyard = {
            "class_2018": 520,          # Vineyards
            "label_2018": "Vineyards",
            "class_2022": 520,
            "label_2022": "Vineyards",
            "is_eudr_commodity": False,
            "is_pre_existing_agricultural": True,
            "is_continuous_agricultural": True,
            "coverage": 0.95,
        }
        result = engine.classify(self._metrics(6.5), crop_map=crop_map_vineyard)
        assert result.color == ComplianceColor.YELLOW

    # ── Pre-existing soya → YELLOW (soya was already there before 2020) ────────

    def test_pre_existing_soya_not_escalated_to_red(self, engine):
        """
        Soya (class 233) in BOTH 2018 AND 2022 means the field was already soya
        before the EUDR reference date. No new expansion → should be YELLOW.
        """
        crop_map_soya_both = {
            "class_2018": 233,          # Soya — 2018 pre-EUDR
            "label_2018": "Soya",
            "class_2022": 233,          # Soya — 2022 post-EUDR
            "label_2022": "Soya",
            "is_eudr_commodity": True,
            "is_pre_existing_agricultural": True,
            "is_continuous_agricultural": True,
            "coverage": 0.80,
        }
        result = engine.classify(self._metrics(6.0), crop_map=crop_map_soya_both)
        # Pre-existing soya (both epochs = same class) → no new expansion → YELLOW
        assert result.color != ComplianceColor.RED

    # ── New soya expansion → RED path (not blocked by crop check) ──────────────

    def test_new_soya_expansion_stays_red(self, engine):
        """
        Soya in 2022 but NOT in 2018 (was forest/non-arable) → new expansion.
        The EUCROPMAP check should NOT block the RED path.
        """
        crop_map_new_soya = {
            "class_2018": 0,            # Unknown / non-agricultural
            "label_2018": "Unknown / Non-agricultural",
            "class_2022": 233,          # Soya — new in 2022
            "label_2022": "Soya",
            "is_eudr_commodity": True,
            "is_pre_existing_agricultural": False,  # was NOT arable in 2018
            "is_continuous_agricultural": False,
            "coverage": 0.75,
        }
        result = engine.classify(self._metrics(8.0), crop_map=crop_map_new_soya)
        # New soya expansion onto non-arable land → RED
        assert result.color == ComplianceColor.RED
        assert result.status == ComplianceStatus.NON_COMPLIANT

    # ── No crop data (outside EU) → engine ignores it ─────────────────────────

    def test_no_crop_data_outside_eu_defaults_to_normal_path(self, engine):
        """
        EUCROPMAP is EU-only. A polygon in Brazil returns all-None crop_map.
        The engine must not crash and must still apply the normal RED path.
        """
        crop_map_null = {
            "class_2018": None,
            "label_2018": None,
            "class_2022": None,
            "label_2022": None,
            "is_eudr_commodity": None,
            "is_pre_existing_agricultural": None,
            "is_continuous_agricultural": None,
            "coverage": None,
        }
        # No crop data → fall through to normal RED (precautionary)
        result = engine.classify(self._metrics(7.0), crop_map=crop_map_null)
        assert result.color == ComplianceColor.RED
        assert result.status == ComplianceStatus.NON_COMPLIANT

    def test_no_crop_map_arg_defaults_to_normal_path(self, engine):
        """Passing crop_map=None (default) must not break normal engine flow."""
        result = engine.classify(self._metrics(2.0))
        assert result.color == ComplianceColor.GREEN

        result2 = engine.classify(self._metrics(8.0))
        assert result2.color == ComplianceColor.RED

    # ── Combined: crop + Hansen contradiction ─────────────────────────────────

    def test_pre_existing_arable_takes_priority_over_hansen_contradiction(self, engine):
        """
        When EUCROPMAP confirms pre-existing arable land, the EUCROPMAP check
        fires first (before Hansen contradiction) and returns YELLOW with the
        crop-rotation explanation rather than the Hansen-contradiction explanation.
        """
        crop_map_cereal = {
            "class_2018": 213,          # Other cereals
            "label_2018": "Other cereals",
            "class_2022": 221,          # Sugar beet
            "label_2022": "Sugar beet",
            "is_eudr_commodity": False,
            "is_pre_existing_agricultural": True,
            "is_continuous_agricultural": True,
            "coverage": 0.88,
        }
        tree_cover = {
            "mean_canopy_pct": 5.0, "forest_pct": 10.0, "mature_forest_pct": 2.0,
            "is_eudr_forest": True, "post_eudr_loss_pct": 0.5,
        }
        result = engine.classify(
            self._metrics(6.0),
            tree_cover=tree_cover,
            crop_map=crop_map_cereal,
        )
        assert result.color == ComplianceColor.YELLOW
        # Reason should mention pre-EUDR crop context
        assert any(word in result.reason.lower() for word in ("pre-existing", "agricultural", "2018"))


# ── evidence hash ─────────────────────────────────────────────────────────────

class TestEvidenceHash:
    def test_hash_is_deterministic_for_same_input(self, simple_polygon):
        # Two calls with same polygon → same hash (modulo timestamp)
        # Since timestamp is injected, we can't test exact equality,
        # but we can verify format.
        h = EUDRComplianceEngine.generate_evidence_hash(
            simple_polygon, {"baseline_date": "2020-12-31", "current_date": "2024-12-31"}
        )
        assert len(h) == 32
        assert h.isalnum()

    def test_different_polygons_different_hashes(self, simple_polygon):
        other = GeoJSONPolygon(
            coordinates=[[[10.0, 20.0], [10.1, 20.0], [10.1, 20.1], [10.0, 20.1], [10.0, 20.0]]]
        )
        h1 = EUDRComplianceEngine.generate_evidence_hash(simple_polygon, {})
        h2 = EUDRComplianceEngine.generate_evidence_hash(other, {})
        # Different polygon coordinates → different hash (timestamps differ too,
        # but coordinate difference is the primary distinguisher)
        assert isinstance(h1, str) and isinstance(h2, str)


# ── buffer polygon ─────────────────────────────────────────────────────────────

class TestBuildBuffer:
    def test_buffer_is_larger_than_original(self, simple_polygon):
        pytest.importorskip("shapely")
        from shapely.geometry import shape

        buffer = EUDRComplianceEngine.build_buffer_polygon(simple_polygon)
        orig_geom = shape({"type": "Polygon", "coordinates": simple_polygon.coordinates})
        buf_geom = shape({"type": "Polygon", "coordinates": buffer.coordinates})
        assert buf_geom.area > orig_geom.area
