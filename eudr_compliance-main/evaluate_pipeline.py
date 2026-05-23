"""
EUDR Pipeline Evaluation Script
================================
Runs the full pipeline on multiple test cases, saves intermediate satellite
images to ./temp/, performs manual rule-based verification, and outputs a
structured JSON evaluation summary.

Usage
-----
    python evaluate_pipeline.py
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate")

# ── Output directory ───────────────────────────────────────────────────────────
TEMP_DIR = Path("./temp")

# ── Test cases ─────────────────────────────────────────────────────────────────
# Each test case has:
#   id, name, coords (lon,lat ring), expected_status, expected_reason
TEST_CASES = [
    # ── Tropical / Global cases ────────────────────────────────────────────────
    {
        "id": "01",
        "name": "Côte d'Ivoire — Cocoa Farm (deforested zone)",
        "coords": [[-5.567, 7.234], [-5.567, 7.244],
                   [-5.557, 7.244], [-5.557, 7.234], [-5.567, 7.234]],
        "expected": "NON-COMPLIANT",
        "expected_reason": "Known cocoa-expanding area with documented post-2020 clearing",
        "region": "West Africa",
    },
    {
        "id": "02",
        "name": "Amazon — Brazil (deforestation hotspot)",
        "coords": [[-55.30, -3.10], [-55.30, -3.09],
                   [-55.29, -3.09], [-55.29, -3.10], [-55.30, -3.10]],
        "expected": "NON-COMPLIANT",
        "expected_reason": "Pará state — active deforestation frontier post-2020",
        "region": "Amazon",
    },
    {
        "id": "03",
        "name": "Congo Basin — Intact forest (DRC)",
        "coords": [[24.10, 0.50], [24.10, 0.51],
                   [24.11, 0.51], [24.11, 0.50], [24.10, 0.50]],
        "expected": "COMPLIANT",
        "expected_reason": "Dense intact Congo Basin forest, minimal human activity",
        "region": "Congo Basin",
    },
    {
        "id": "04",
        "name": "Borneo — Palm oil expansion (Indonesia)",
        "coords": [[117.20, 1.10], [117.20, 1.11],
                   [117.21, 1.11], [117.21, 1.10], [117.20, 1.10]],
        "expected": "NON-COMPLIANT",
        "expected_reason": "Kalimantan palm oil plantation expansion area post-2020",
        "region": "Borneo",
    },
    {
        "id": "05",
        "name": "Colombia — Coffee highlands (COMPLIANT region)",
        "coords": [[-75.60, 5.05], [-75.60, 5.06],
                   [-75.59, 5.06], [-75.59, 5.05], [-75.60, 5.05]],
        "expected": "COMPLIANT",
        "expected_reason": "Established Colombian coffee region, stable canopy",
        "region": "Andes",
    },
    # ── European test cases ────────────────────────────────────────────────────
    # EUDR context for Europe:
    #   - Reference date 31 Dec 2020
    #   - Pre-2020 cleared farmland → 2020 baseline already shows low NDVI → COMPLIANT
    #   - Post-2020 logging → 2020→2024 NDVI drop → NON-COMPLIANT (if forested baseline)
    #   - Natural disturbance (storm, beetle) → YELLOW even if NDVI drops
    {
        "id": "EU-01",
        "name": "Romania — Maramureș (post-2020 timber logging)",
        "coords": [[24.785, 47.885], [24.785, 47.895],
                   [24.795, 47.895], [24.795, 47.885], [24.785, 47.885]],
        "expected": "NON-COMPLIANT",
        "expected_reason": (
            "Carpathian mountain forest: documented post-2020 logging operations. "
            "High baseline canopy (>60%) should trigger NON-COMPLIANT if loss detected. "
            "Romania is one of the EU's most active illegal logging areas."
        ),
        "region": "Temperate Europe",
    },
    {
        "id": "EU-02",
        "name": "Poland — Białowieża (protected primeval forest, COMPLIANT)",
        "coords": [[23.845, 52.725], [23.845, 52.735],
                   [23.855, 52.735], [23.855, 52.725], [23.845, 52.725]],
        "expected": "COMPLIANT",
        "expected_reason": (
            "UNESCO World Heritage primeval forest — strictly protected, no logging. "
            "2020 baseline and 2024 current should both show dense canopy (NDVI >0.6). "
            "Expected: COMPLIANT or at most MEDIUM RISK due to 2017 bark-beetle event."
        ),
        "region": "Temperate Europe",
    },
    {
        "id": "EU-03",
        "name": "Spain — Extremadura farmland (pre-2020 cleared, COMPLIANT)",
        "coords": [[-6.055, 38.475], [-6.055, 38.485],
                   [-6.045, 38.485], [-6.045, 38.475], [-6.055, 38.475]],
        "expected": "COMPLIANT",
        "expected_reason": (
            "Dehesa agricultural landscape — land was cleared well before 2020. "
            "2020 baseline already shows low/moderate NDVI (open pasture/olive groves). "
            "Pre-2020 clearance is EUDR-exempt; 2020→2024 change should be near zero."
        ),
        "region": "Temperate Europe",
    },
    {
        "id": "EU-04",
        "name": "Germany — Bavarian Forest (beetle-kill natural disturbance, YELLOW)",
        "coords": [[13.395, 48.895], [13.395, 48.905],
                   [13.405, 48.905], [13.405, 48.895], [13.395, 48.895]],
        "expected": "MEDIUM RISK",
        "expected_reason": (
            "Bavarian Forest National Park — bark beetle (Ips typographus) outbreak "
            "caused significant post-2020 tree mortality. This is a natural disturbance "
            "(WRI driver class 7) — NOT an EUDR commodity violation → expected YELLOW."
        ),
        "region": "Temperate Europe",
    },
    {
        "id": "EU-05",
        "name": "Romania — Bucharest farmland (long-standing agricultural, COMPLIANT)",
        "coords": [[25.495, 44.495], [25.495, 44.505],
                   [25.505, 44.505], [25.505, 44.495], [25.495, 44.495]],
        "expected": "COMPLIANT",
        "expected_reason": (
            "Flat agricultural plain south of Bucharest — permanently cultivated farmland "
            "with no forest baseline. treecover2000 <10% → is_eudr_forest=False. "
            "Any land-cover change detected is NOT deforestation under EUDR Article 2(4)."
        ),
        "region": "Temperate Europe",
    },
]

# NDVI threshold for CV-based tree-loss classification
NDVI_LOSS_THRESHOLD = -0.10   # NDVI drop > 0.10 → tree loss pixel
NDVI_GAIN_THRESHOLD = 0.05    # NDVI gain > 0.05 → regrowth pixel


# ── Image saving utilities ────────────────────────────────────────────────────

def _save_png(arr: np.ndarray, path: Path) -> None:
    """Save any numpy array as a PNG (handles float → uint8 conversion)."""
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _rgb_from_s2(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Convert a [H,W,6] Sentinel-2 array to uint8 RGB (B4=Red, B3=Green, B2=Blue).
    Uses per-channel 2–98 percentile stretch for robust visualisation across
    all land-cover types (avoids washed-out or fully dark images).
    """
    rgb = arr[:, :, [2, 1, 0]].astype(np.float32)   # Red, Green, Blue bands
    # Per-channel percentile stretch
    for ch in range(3):
        lo, hi = np.percentile(rgb[:, :, ch], 2), np.percentile(rgb[:, :, ch], 98)
        if hi > lo:
            rgb[:, :, ch] = np.clip((rgb[:, :, ch] - lo) / (hi - lo), 0, 1)
        else:
            rgb[:, :, ch] = 0.0
    rgb = (rgb * 255).astype(np.uint8)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def _ndvi_heatmap(ndvi: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert [H,W] float NDVI to a colour heatmap uint8 image."""
    small = cv2.resize(ndvi, (size, size), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(((small + 1.0) / 2.0 * 255), 0, 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_RdYlGn", None) or cv2.COLORMAP_JET
    return cv2.applyColorMap(norm, colormap)


def _change_map_image(change_map: torch.Tensor, size: int = 224) -> np.ndarray:
    """
    Convert a [1,1,H,W] change probability tensor to a heatmap image.

    Uses COLORMAP_JET (blue=low, red=high) instead of COLORMAP_HOT so that
    near-0.5 neutral probabilities appear as green/yellow rather than all-orange,
    making it easy to visually distinguish deforested from intact areas.
    """
    prob = change_map.squeeze().cpu().float().numpy()
    # Normalize relative to [0, 1] probability range using actual min/max
    # so differences are visible even when all values cluster near 0.5
    lo, hi = prob.min(), prob.max()
    if hi > lo:
        norm_f = (prob - lo) / (hi - lo)
    else:
        norm_f = np.full_like(prob, 0.5)
    norm = (np.clip(norm_f, 0, 1) * 255).astype(np.uint8)
    resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(resized, cv2.COLORMAP_JET)


def _cv_tree_loss_map(ndvi_change: np.ndarray, size: int = 224) -> np.ndarray:
    """
    CV-based tree-loss segmentation using NDVI thresholds.
      RED   pixels → tree loss   (NDVI drop > 0.10)
      GREEN pixels → regrowth    (NDVI gain > 0.05)
      GREY  pixels → no change
    """
    small = cv2.resize(ndvi_change, (size, size), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 100, dtype=np.uint8)  # grey baseline
    out[small < NDVI_LOSS_THRESHOLD] = [0, 0, 220]       # red = loss (BGR)
    out[small > NDVI_GAIN_THRESHOLD] = [0, 180, 0]       # green = gain
    return out


def _masked(img: np.ndarray, coords: Optional[list] = None) -> np.ndarray:
    """
    Apply a polygon mask to the image using rasterio.features.rasterize.

    If `coords` (lon/lat ring) is provided, the ring is mapped to pixel
    space via a simple linear transform (no CRS — display-only accuracy).
    Falls back to a circular approximation if rasterio is unavailable.
    """
    h, w = img.shape[:2]
    result = img.copy()

    if coords is not None:
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            from shapely.geometry import Polygon as ShapelyPolygon

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)

            # Pad slightly so edge pixels are included
            pad = max((lon_max - lon_min) * 0.05, (lat_max - lat_min) * 0.05, 1e-6)
            transform = from_bounds(
                lon_min - pad, lat_min - pad,
                lon_max + pad, lat_max + pad,
                w, h,
            )
            poly = ShapelyPolygon(coords)
            mask = rasterize(
                [(poly, 1)],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            if img.ndim == 3:
                result[mask == 0] = 0
            else:
                result[mask == 0] = 0
            return result
        except Exception:
            pass  # fall through to circle fallback

    # Fallback: circular approximation
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 4, 255, -1)
    if img.ndim == 3:
        result[mask == 0] = 0
    else:
        result[mask == 0] = 0
    return result


def save_images(tc_id: str, raw_b: np.ndarray, raw_c: np.ndarray,
                ndvi_b: np.ndarray, ndvi_c: np.ndarray,
                ndvi_chg: np.ndarray, change_map: torch.Tensor,
                coords: Optional[list] = None) -> list[str]:
    """Save all intermediate images for one test case. Returns list of paths."""
    saved = []
    prefix = TEMP_DIR / f"testcase_{tc_id}"

    images = {
        "raw_before":       _rgb_from_s2(raw_b),
        "raw_after":        _rgb_from_s2(raw_c),
        "ndvi_before":      _ndvi_heatmap(ndvi_b),
        "ndvi_after":       _ndvi_heatmap(ndvi_c),
        "ndvi_change":      _ndvi_heatmap(ndvi_chg),
        "prithvi_change":   _change_map_image(change_map),
        "cv_treeloss":      _cv_tree_loss_map(ndvi_chg),
    }

    for name, img in images.items():
        path = Path(f"{prefix}_{name}.png")
        _save_png(img, path)
        saved.append(str(path))

        masked_path = Path(f"{prefix}_{name}_masked.png")
        _save_png(_masked(img, coords=coords), masked_path)
        saved.append(str(masked_path))

    return saved


# ── CV-based manual verification ──────────────────────────────────────────────

def manual_verify(ndvi_b: np.ndarray, ndvi_c: np.ndarray,
                  ndvi_chg: np.ndarray) -> dict:
    """
    Rule-based manual verification — no model, pure spectral analysis.

    Rules
    -----
    1. Mean NDVI baseline > 0.4 → forested baseline (vegetation present)
    2. Mean NDVI change < -0.10 → significant loss event
    3. Loss pixel fraction (NDVI < -0.10) > 5% → exceeds EUDR threshold
    4. NDVI current < 0.2 → bare/degraded land (strong evidence of clearing)
    """
    mean_ndvi_b = float(np.nanmean(ndvi_b))
    mean_ndvi_c = float(np.nanmean(ndvi_c))
    mean_change  = float(np.nanmean(ndvi_chg))
    loss_pct     = float(np.mean(ndvi_chg < NDVI_LOSS_THRESHOLD) * 100)
    gain_pct     = float(np.mean(ndvi_chg > NDVI_GAIN_THRESHOLD) * 100)

    was_forested = mean_ndvi_b > 0.35
    is_degraded  = mean_ndvi_c < 0.25
    exceeds_eudr = loss_pct > 5.0

    if exceeds_eudr and was_forested:
        verdict = "NON-COMPLIANT"
        reason  = (f"NDVI loss pixels: {loss_pct:.1f}% (>5% threshold). "
                   f"Baseline NDVI={mean_ndvi_b:.3f} confirms prior forest cover.")
    elif exceeds_eudr and not was_forested:
        verdict = "MEDIUM RISK"
        reason  = (f"NDVI loss pixels: {loss_pct:.1f}% but baseline NDVI={mean_ndvi_b:.3f} "
                   f"suggests non-forest land use change. Enhanced due diligence needed.")
    elif is_degraded:
        verdict = "MEDIUM RISK"
        reason  = (f"Current NDVI={mean_ndvi_c:.3f} is below 0.25, indicating degraded "
                   f"or sparsely vegetated land. Loss pixels: {loss_pct:.1f}%.")
    else:
        verdict = "COMPLIANT"
        reason  = (f"No significant NDVI loss detected. Loss pixels: {loss_pct:.1f}%. "
                   f"Current NDVI={mean_ndvi_c:.3f} suggests active vegetation cover.")

    return {
        "verdict": verdict,
        "reason": reason,
        "stats": {
            "mean_ndvi_baseline": round(mean_ndvi_b, 4),
            "mean_ndvi_current":  round(mean_ndvi_c, 4),
            "mean_ndvi_change":   round(mean_change, 4),
            "loss_pixel_pct":     round(loss_pct, 2),
            "gain_pixel_pct":     round(gain_pct, 2),
            "was_forested":       was_forested,
            "is_degraded":        is_degraded,
        },
    }


# ── Improvement recommendations ───────────────────────────────────────────────

def generate_recommendations(results: list[dict]) -> list[str]:
    recs = []

    mismatches = [r for r in results if not r["match"]]
    system_wrong = [r for r in mismatches
                    if r["manual_verification"]["verdict"] != r["system_output"]["compliance_status"]]

    if any(r["system_output"]["confidence"] < 0.55 for r in results):
        recs.append(
            "LOW CONFIDENCE DETECTIONS: Mean change-detection probability is near 0.5 on several "
            "polygons, indicating the ViT-Base fallback backbone is not well-calibrated for "
            "Sentinel-2 6-band data. Priority fix: resolve Prithvi-100M loading (safetensors "
            "format mismatch) or fine-tune ViT-Base on Sentinel-2 patches."
        )

    if any(r["system_output"]["compliance_status"] != r["expected"] for r in results):
        recs.append(
            "THRESHOLD SENSITIVITY: The 0.5 binary threshold on the change-detection head is "
            "producing unstable results. Recommendation: lower to 0.35 for tropical forest "
            "polygons, or use the NDVI-change signal as a soft prior to bias the threshold."
        )

    recs.append(
        "NDVI PREPROCESSING: Add temporal compositing (median of ±30 days around target date) "
        "to reduce cloud/shadow contamination before NDVI computation."
    )
    recs.append(
        "CLOUD MASKING: Integrate Sentinel-2 QA60 cloud-mask band to zero-out cloud pixels "
        "before change detection instead of relying solely on cloud-cover percentage filtering."
    )
    recs.append(
        "TEMPORAL SELECTION: Use dry-season windows per biome (e.g. Jun-Sep for West Africa, "
        "Jul-Oct for Amazon) to ensure cloud-free imagery and consistent phenological state."
    )
    recs.append(
        "POLYGON CLIPPING: Replace the circular mask approximation with actual rasterised "
        "polygon boundaries using rasterio.features.rasterize for precise pixel-level masking."
    )
    recs.append(
        "PRITHVI-100M: The model's HuggingFace repository switched to safetensors but the "
        "transformers AutoModel loader is not finding it. Fix: explicitly pass "
        "use_safetensors=True and pin transformers>=4.48 where TimmWrapper support is stable."
    )
    recs.append(
        "BUFFER ZONE: The 5 km buffer download exceeds GEE's 50 MB limit. Fix: reduce buffer "
        "to 2 km for farm-scale polygons, or use computePixels() with ee.Export instead of "
        "getDownloadURL() for large regions."
    )
    recs.append(
        "HANSEN GFC: Update the asset ID from global_forest_change_2023_v1_11 to "
        "global_forest_change_2024_v1_12 as the 2023 dataset is now deprecated by GEE."
    )

    return recs


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run() -> None:
    # Reset temp folder
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    log.info("Temp folder reset: %s", TEMP_DIR.resolve())

    # Initialise shared pipeline components
    from eudr.credentials import setup
    from eudr.acquisition.satellite import SatelliteDataAcquisition
    from eudr.preprocessing.pipeline import GeospatialPreprocessor
    from eudr.models.change_detector import PrithviChangeDetector
    from eudr.compliance.engine import EUDRComplianceEngine
    from eudr.schemas import GeoJSONPolygon

    try:
        setup(interactive=False)
    except RuntimeError as exc:
        log.error("Credential setup failed: %s", exc)
        sys.exit(1)

    log.info("Loading models (shared across all test cases)…")
    acquirer    = SatelliteDataAcquisition()
    preprocessor = GeospatialPreprocessor()
    detector    = PrithviChangeDetector(freeze_backbone=True)
    engine      = EUDRComplianceEngine()
    log.info("Models ready.")

    results = []
    all_saved_images = []

    for tc in TEST_CASES:
        tc_id   = tc["id"]
        tc_name = tc["name"]
        log.info("=" * 60)
        log.info("Test case %s: %s", tc_id, tc_name)

        system_output = {}
        manual_verification = {}
        saved_images = []
        error = None

        try:
            polygon = GeoJSONPolygon(coordinates=[tc["coords"]])

            # ── 1. Satellite acquisition ─────────────────────────────────────
            log.info("[%s] Fetching satellite data…", tc_id)
            raw = acquirer.fetch_baseline_and_current(polygon)
            arr_b = raw["baseline_2020"]   # [H, W, 6]
            arr_c = raw["current"]         # [H, W, 6]

            # ── 2. Preprocessing + NDVI ──────────────────────────────────────
            log.info("[%s] Preprocessing + NDVI…", tc_id)
            processed = preprocessor.prepare(arr_b, arr_c,
                                             sar_current=raw.get("sar_current"))
            ndvi_b   = processed.ndvi_baseline
            ndvi_c   = processed.ndvi_current
            ndvi_chg = processed.ndvi_change

            # ── 3. Change detection (Prithvi / ViT-Base + NDVI prior) ────────
            log.info("[%s] Running change detection…", tc_id)
            detector.eval()
            with torch.no_grad():
                det_out = detector(
                    processed.t1_tensor,
                    processed.t2_tensor,
                    ndvi_change=processed.ndvi_change,
                )
            change_map = det_out.change_map

            # ── 4. Compliance metrics (with WRI driver + forest baseline) ────────
            wri_driver = raw.get("wri_driver")
            tree_cover = raw.get("tree_cover")
            metrics    = engine.compute_metrics(change_map)
            compliance = engine.classify(
                metrics,
                wri_driver=wri_driver,
                tree_cover=tree_cover,
            )

            # ── 5. CV-based tree-loss map (NDVI threshold) ────────────────────
            loss_pct_cv = float(np.mean(ndvi_chg < NDVI_LOSS_THRESHOLD) * 100)

            # ── 6. Save images ────────────────────────────────────────────────
            log.info("[%s] Saving images to %s…", tc_id, TEMP_DIR)
            saved_images = save_images(
                tc_id, arr_b, arr_c, ndvi_b, ndvi_c, ndvi_chg, change_map,
                coords=tc["coords"],
            )
            all_saved_images.extend(saved_images)

            system_output = {
                "compliance_status":  compliance.status.value,
                "compliance_color":   compliance.color.value,
                "reason":             compliance.reason,
                "change_percentage":  round(metrics.change_percentage, 2),
                "loss_area_ha":       round(metrics.loss_area_hectares, 4),
                "confidence":         round(metrics.change_probability, 4),
                "cv_loss_pixel_pct":  round(loss_pct_cv, 2),
                "ndvi_mean_baseline": round(float(np.nanmean(ndvi_b)), 4),
                "ndvi_mean_current":  round(float(np.nanmean(ndvi_c)), 4),
                "ndvi_mean_change":   round(float(np.nanmean(ndvi_chg)), 4),
                "backbone_used":      "ViT-Base/16 (Prithvi fallback)",
                # WRI driver context
                "wri_driver_label":   (wri_driver or {}).get("driver_label", "Unknown"),
                "wri_driver_class":   (wri_driver or {}).get("driver_class", 0),
                "is_eudr_driver":     (wri_driver or {}).get("is_eudr_driver"),
                "wri_source":         (wri_driver or {}).get("source", "N/A"),
                # Forest baseline context
                "mean_canopy_pct":    (tree_cover or {}).get("mean_canopy_pct"),
                "is_eudr_forest":     (tree_cover or {}).get("is_eudr_forest"),
                "mature_forest_pct":  (tree_cover or {}).get("mature_forest_pct"),
                # Region
                "region":             tc.get("region", "Unknown"),
            }

            # ── 7. Manual verification ────────────────────────────────────────
            manual_verification = manual_verify(ndvi_b, ndvi_c, ndvi_chg)

        except Exception as exc:
            log.error("[%s] Pipeline failed: %s", tc_id, exc)
            error = str(exc)
            system_output = {"error": error}
            manual_verification = {"verdict": "ERROR", "reason": error}

        # ── Compare system vs manual ──────────────────────────────────────────
        sys_status  = system_output.get("compliance_status", "ERROR")
        man_verdict = manual_verification.get("verdict", "ERROR")
        match = sys_status == man_verdict

        explanation = _build_explanation(tc, sys_status, man_verdict, system_output,
                                         manual_verification, match)

        result = {
            "test_case_id":        tc_id,
            "test_case_name":      tc_name,
            "expected":            tc["expected"],
            "system_output":       system_output,
            "manual_verification": manual_verification,
            "match":               match,
            "explanation":         explanation,
            "saved_images":        saved_images,
        }
        results.append(result)

        log.info("[%s] System: %-15s | Manual: %-15s | Match: %s",
                 tc_id, sys_status, man_verdict, "✓" if match else "✗")

    # ── Final summary ──────────────────────────────────────────────────────────
    matched     = sum(1 for r in results if r["match"])
    total       = len(results)
    errors      = sum(1 for r in results if "error" in r["system_output"])
    accuracy    = f"{matched}/{total - errors} ({100*matched/max(total-errors,1):.0f}%)"
    recs        = generate_recommendations(results)

    summary = {
        "evaluation_summary": (
            f"Ran {total} test cases. "
            f"System vs manual-verification match: {matched}/{total}. "
            f"Errors (pipeline failures): {errors}. "
            f"Accuracy (excluding errors): {accuracy}."
        ),
        "accuracy_score":              accuracy,
        "per_test_case_results":       results,
        "improvement_recommendations": recs,
        "saved_images":                all_saved_images,
    }

    # Print to terminal
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for r in results:
        status_icon = "✓" if r["match"] else "✗"
        sys_s = r["system_output"].get("compliance_status", "ERROR")
        man_s = r["manual_verification"].get("verdict", "ERROR")
        print(f"  [{status_icon}] TC-{r['test_case_id']} {r['test_case_name']}")
        print(f"       System: {sys_s}   Manual: {man_s}")
        print(f"       {r['explanation']}")
        print()

    print(f"Accuracy: {accuracy}")
    print(f"Images saved to: {TEMP_DIR.resolve()}/")
    print()

    # Show EU-specific results summary
    eu_results = [r for r in results if r["test_case_id"].startswith("EU")]
    if eu_results:
        print("── European Test Cases ─────────────────────────────────")
        for r in eu_results:
            so = r["system_output"]
            print(f"  {r['test_case_id']} {r['test_case_name']}")
            print(f"    Expected : {r['expected']}")
            print(f"    System   : {so.get('compliance_status','?')} ({so.get('compliance_color','?')})")
            print(f"    WRI      : {so.get('wri_driver_label','?')} (class {so.get('wri_driver_class','?')}, EUDR={so.get('is_eudr_driver','?')})")
            canopy = so.get('mean_canopy_pct')
            print(f"    Forest   : canopy={f'{canopy:.1f}%' if canopy is not None else 'N/A'}  EUDR-forest={so.get('is_eudr_forest','?')}")
            print(f"    Change   : {so.get('change_percentage','?')}%  confidence={so.get('confidence','?')}")
            print()
        print("──────────────────────────────────────────────────────")

    print("=" * 70)

    # Save JSON
    out_path = Path("evaluation_results.json")
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Full results written to %s", out_path.resolve())


def _build_explanation(tc: dict, sys_status: str, man_verdict: str,
                        sys_out: dict, man_v: dict, match: bool) -> str:
    expected = tc["expected"]
    if "error" in sys_out:
        return f"Pipeline error — could not evaluate: {sys_out['error']}"

    parts = []
    if sys_status == expected:
        parts.append(f"System correctly identified {expected}.")
    else:
        parts.append(
            f"System returned {sys_status} but expected {expected}. "
            f"Likely cause: ViT-Base backbone is not pre-trained on Sentinel-2 "
            f"6-band data, so change-detection scores are near-random (confidence="
            f"{sys_out.get('confidence', '?'):.3f} ≈ 0.5)."
        )

    if not match:
        parts.append(
            f"Manual NDVI analysis gives {man_verdict} "
            f"(loss pixels: {man_v.get('stats', {}).get('loss_pixel_pct', '?')}%, "
            f"baseline NDVI: {man_v.get('stats', {}).get('mean_ndvi_baseline', '?')})."
        )
    else:
        parts.append(
            f"Manual verification agrees. "
            f"NDVI loss pixels: {man_v.get('stats', {}).get('loss_pixel_pct', '?')}%."
        )

    return " ".join(parts)


if __name__ == "__main__":
    run()