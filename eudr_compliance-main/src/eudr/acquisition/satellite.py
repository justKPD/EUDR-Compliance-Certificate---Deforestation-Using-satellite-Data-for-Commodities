"""
Satellite data acquisition via Google Earth Engine.

GEE is initialised lazily — importing this module never triggers authentication.
The first call to ``SatelliteDataAcquisition.fetch_baseline_and_current()``
will call ``ee.Initialize()``, which succeeds silently if Application Default
Credentials are already on disk (from ``python setup_auth.py`` or
``earthengine authenticate``).

If no credentials are found a ``DataAcquisitionError`` is raised with clear
instructions — nothing ever hangs waiting for a browser.

Data sources
------------
  Sentinel-2 MSI    (optical, 10 m)  — primary change detection
  Sentinel-1 GRD    (SAR, 10 m)      — all-weather cloud-penetrating backup
  Hansen GFC 2023   (30 m)           — forest loss labels for fine-tuning
  WRI/GDM Drivers   (1 km)           — deforestation cause classification
"""
from __future__ import annotations

import io
import logging
import urllib.request
from typing import Optional

import numpy as np

from eudr.config import settings
from eudr.exceptions import DataAcquisitionError
from eudr.regulations import (
    EUCROPMAP_ARABLE_CLASSES,
    EUCROPMAP_CLASSES,
    EUCROPMAP_EUDR_CLASSES,
)
from eudr.schemas import GeoJSONPolygon

log = logging.getLogger(__name__)

# Sentinel-2 bands required by Prithvi-100M — order is critical
S2_BANDS: list[str] = ["B2", "B3", "B4", "B8A", "B11", "B12"]
S2_BAND_NAMES: list[str] = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
S1_BANDS: list[str] = ["VV", "VH"]

RESOLUTION_M: int = 10
MAX_CLOUD_PCT: float = 20.0
FALLBACK_CLOUD_PCT: float = 40.0

HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
EUCROPMAP_ASSET = "JRC/D5/EUCROPMAP/V1"

# WRI Global Drivers of Deforestation 1km v1.2 (2001–2024)
# GEE catalog: projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024
# Two asset path variants exist in GEE — we try both.
WRI_DRIVERS_ASSET_PRIMARY = (
    "projects/landandcarbon/assets/"
    "wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024"
)
WRI_DRIVERS_ASSET_ALT = (
    "projects/landandcarbon/assets/"
    "wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024"
)

# Official 7-class taxonomy (v1.2, 2001-2024)
# Source: https://developers.google.com/earth-engine/datasets/catalog/
#         projects_landandcarbon_assets_wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024
WRI_DRIVER_LABELS: dict[int, str] = {
    1: "Permanent agriculture",       # EUDR-relevant: crops / pasture
    2: "Hard commodities",            # mining / extractives
    3: "Shifting cultivation",        # EUDR-relevant: subsistence farming
    4: "Logging",                     # EUDR-relevant: timber / rubber
    5: "Wildfire",                    # natural disturbance — NOT EUDR violation
    6: "Settlements & infrastructure",# urbanisation
    7: "Other natural disturbances",  # natural disturbance — NOT EUDR violation
}

# EUDR-relevant driver classes: forest loss attributable to commodity supply chains
# (EUDR Regulation EU 2023/1115, Article 2 — applicable commodities)
# These constants mirror eudr.regulations — keep in sync.
EUDR_COMMODITY_DRIVERS:     set[int] = {1, 3, 4}   # agriculture / logging → RED
EUDR_NATURAL_DRIVERS:       set[int] = {5, 7}       # wildfire / natural → YELLOW (not EUDR)
EUDR_NON_COMMODITY_DRIVERS: set[int] = {2, 6}       # mining / settlements → YELLOW (not EUDR)

# Forest baseline: EUDR defines forest as trees able to reach >5 m, ≥10% canopy cover
# (EUDR Article 2(4), referencing FAO FRA definitions).
# We use Hansen treecover2000 (% canopy closure for trees >5 m) as the proxy.
FOREST_CANOPY_MIN_PCT:  int = 10   # EUDR-compliant minimum canopy cover
FOREST_CANOPY_HIGH_PCT: int = 30   # proxy for taller trees (~15 m mature canopy)

# Biome dry-season windows: (biome_name, lon_min, lon_max, lat_min, lat_max, start_month, end_month)
# Used to select cloud-free imagery with consistent phenological state
_BIOME_WINDOWS = [
    # (name, lon_min, lon_max, lat_min, lat_max, dry_start_mm, dry_end_mm)
    ("West Africa",      -20,  20,   0, 15, 11,  4),   # Nov–Apr  dry harmattan
    ("Amazon",           -80, -45, -20,  5,  6, 10),   # Jun–Oct  dry season
    ("Congo Basin",       10,  35, -10,  5,  6,  9),   # Jun–Sep  boreal-summer dry
    ("Borneo",            95, 120,  -5,  8,  4,  7),   # Apr–Jul  reduced rain
    ("Andes",            -80, -65,  -5, 12, 12,  3),   # Dec–Mar  dry highland
    ("Temperate Europe", -30,  45,  35, 72,  6,  8),   # Jun–Aug  peak growing season
]


class SatelliteDataAcquisition:
    """
    Fetches all satellite and reference data needed for one polygon analysis.

    GEE is initialised on construction (not at module level).
    Raises ``DataAcquisitionError`` with clear instructions if auth fails.
    """

    def __init__(self) -> None:
        self._ee = self._import_ee()
        self._init_gee()

    # ── Initialisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _import_ee():
        try:
            import ee  # type: ignore[import]
            return ee
        except ImportError as exc:
            raise DataAcquisitionError(
                "earthengine-api is not installed.\n"
                "Run: pip install earthengine-api"
            ) from exc

    def _init_gee(self) -> None:
        """
        Initialise GEE using the best available credential source.
        Never calls ee.Authenticate() — that is a one-time setup step
        performed by ``setup_auth.py``.
        """
        ee = self._ee
        project_id = settings.require_gee()

        # 1. Service-account key file
        key = settings.gee_service_account_key.strip()
        if key:
            try:
                creds = ee.ServiceAccountCredentials(email="", key_file=key)
                ee.Initialize(credentials=creds, project=project_id)
                log.info("GEE initialised with service-account key (project: %s)", project_id)
                return
            except Exception as exc:
                log.warning("Service-account auth failed: %s — falling back to ADC", exc)

        # 2. Application Default Credentials (cached by prior earthengine authenticate)
        try:
            ee.Initialize(project=project_id)
            log.info("GEE initialised with ADC (project: %s)", project_id)
            return
        except Exception as exc:
            raise DataAcquisitionError(
                f"Google Earth Engine authentication failed: {exc}\n\n"
                "Run one of the following to authenticate:\n"
                "  python setup_auth.py\n"
                "  earthengine authenticate\n"
                "  gcloud auth application-default login"
            ) from exc

    # ── Biome date windows ─────────────────────────────────────────────────────

    @staticmethod
    def _get_biome_dates(geometry, year: int, ee) -> tuple:
        """
        Return (start_date, end_date) strings for the dry season that best
        covers the given geometry centroid.  Falls back to full-year range.
        """
        try:
            centroid = geometry.centroid().coordinates().getInfo()
            lon, lat = float(centroid[0]), float(centroid[1])
        except Exception:
            return f"{year}-01-01", f"{year}-12-31"

        for _name, lon_min, lon_max, lat_min, lat_max, m_start, m_end in _BIOME_WINDOWS:
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                # Dry season may span a calendar year boundary
                if m_start <= m_end:
                    start = f"{year}-{m_start:02d}-01"
                    # last day of end month (approx)
                    end   = f"{year}-{m_end:02d}-28"
                else:
                    # e.g. Nov (11) → Apr (4): start in previous year for baseline
                    start = f"{year - 1}-{m_start:02d}-01"
                    end   = f"{year}-{m_end:02d}-28"
                log.debug("Biome dry window for (%.2f, %.2f): %s → %s", lon, lat, start, end)
                return start, end

        return f"{year}-01-01", f"{year}-12-31"

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_baseline_and_current(
        self,
        polygon: GeoJSONPolygon,
        baseline_year: int = 2020,
        current_year: int = 2024,
    ) -> dict:
        """
        Fetch Sentinel-2, SAR, Hansen, and WRI data for a polygon.

        Returns a dict with:
          baseline_2020     ndarray [H, W, 6]
          current           ndarray [H, W, 6]
          sar_current       ndarray [H, W, 2] | None
          hansen_mask       ndarray [H, W]    | None
          tree_cover        dict  (mean_canopy_pct, forest_pct, mature_forest_pct,
                                   is_eudr_forest)
          wri_driver        dict  (driver_class, driver_label, confidence,
                                   source, is_eudr_driver)
          metadata          dict
        """
        ee = self._ee
        geometry = ee.Geometry.Polygon(polygon.coordinates)

        # Use biome-aware dry-season windows for cloud-free, phenologically
        # consistent imagery.  Falls back to full-year if biome not matched.
        b_start, b_end = self._get_biome_dates(geometry, baseline_year, ee)
        c_start, c_end = self._get_biome_dates(geometry, current_year, ee)

        log.info("Fetching Sentinel-2 %d baseline (%s → %s) …", baseline_year, b_start, b_end)
        s2_b = self._get_optical(geometry, b_start, b_end)
        arr_b = self._to_numpy(s2_b, geometry, S2_BANDS)

        log.info("Fetching Sentinel-2 %d current (%s → %s) …", current_year, c_start, c_end)
        s2_c = self._get_optical(geometry, c_start, c_end)
        arr_c = self._to_numpy(s2_c, geometry, S2_BANDS)

        sar: Optional[np.ndarray] = None
        try:
            s1 = self._get_sar(geometry, f"{current_year}-01-01", f"{current_year}-12-31")
            sar = self._to_numpy(s1, geometry, S1_BANDS)
            log.info("SAR acquired: %s", sar.shape)
        except Exception as exc:
            log.warning("SAR unavailable (optical-only): %s", exc)

        hansen: Optional[np.ndarray] = None
        try:
            hansen = self._get_hansen_mask(geometry)
        except Exception as exc:
            log.warning("Hansen mask unavailable: %s", exc)

        # Tree cover baseline — confirms area was forested per EUDR Article 2(4)
        log.info("Fetching Hansen treecover2000 baseline …")
        tree_cover = self._get_tree_cover(geometry)
        if tree_cover["mean_canopy_pct"] is not None:
            log.info(
                "Tree cover: mean=%.1f%%  EUDR-forest=%.1f%%  mature(≥30%%)=%.1f%%",
                tree_cover["mean_canopy_pct"],
                tree_cover["forest_pct"],
                tree_cover["mature_forest_pct"],
            )

        # WRI GDM Drivers — classifies the cause of any detected forest loss
        log.info("Fetching WRI GDM deforestation drivers …")
        wri_driver = self._get_wri_driver(geometry)
        log.info(
            "WRI driver: %s (class=%d, EUDR-relevant=%s, source=%s)",
            wri_driver["driver_label"],
            wri_driver["driver_class"],
            wri_driver["is_eudr_driver"],
            wri_driver["source"],
        )

        # JRC EU CropMap V1 — confirms pre-existing agricultural land-use
        # (2018 = pre-EUDR baseline; 2022 = post-EUDR snapshot)
        log.info("Fetching JRC EU CropMap V1 (EUCROPMAP) …")
        crop_map = self._get_eucropmap(geometry)
        if crop_map["coverage"] is not None:
            log.info(
                "EUCROPMAP: 2018=%s (%s)  2022=%s (%s)  eudr_crop=%s  pre_existing=%s",
                crop_map["class_2018"], crop_map["label_2018"],
                crop_map["class_2022"], crop_map["label_2022"],
                crop_map["is_eudr_commodity"],
                crop_map["is_pre_existing_agricultural"],
            )

        return {
            "baseline_2020": arr_b,
            "current": arr_c,
            "sar_current": sar,
            "hansen_mask": hansen,
            "tree_cover": tree_cover,
            "wri_driver": wri_driver,
            "crop_map": crop_map,
            # Legacy keys kept for backward compatibility with pipeline.py
            "wri_driver_label": wri_driver["driver_label"],
            "metadata": {
                "baseline_date":     f"{baseline_year}-12-31",
                "current_date":      f"{current_year}-12-31",
                "resolution_m":      RESOLUTION_M,
                "bands":             S2_BAND_NAMES,
                "wri_driver":        wri_driver["driver_label"],
                "wri_driver_class":  wri_driver["driver_class"],
                "wri_driver_source": wri_driver["source"],
                "is_eudr_driver":    wri_driver["is_eudr_driver"],
                "mean_canopy_pct":      tree_cover["mean_canopy_pct"],
                "is_eudr_forest":       tree_cover["is_eudr_forest"],
                "mature_forest_pct":    tree_cover["mature_forest_pct"],
                "post_eudr_loss_pct":   tree_cover["post_eudr_loss_pct"],
                # EUCROPMAP fields
                "crop_class_2018":               crop_map["class_2018"],
                "crop_label_2018":               crop_map["label_2018"],
                "crop_class_2022":               crop_map["class_2022"],
                "crop_label_2022":               crop_map["label_2022"],
                "crop_is_eudr_commodity":        crop_map["is_eudr_commodity"],
                "crop_is_pre_existing_agricultural": crop_map["is_pre_existing_agricultural"],
                "crop_is_continuous_agricultural":   crop_map["is_continuous_agricultural"],
                "crop_coverage":                 crop_map["coverage"],
            },
        }

    # ── Optical (Sentinel-2) ───────────────────────────────────────────────────

    def _mask_clouds_s2(self, image):
        """
        Apply Sentinel-2 QA60 bitmask to mask clouds and cirrus.
        Bit 10 = opaque clouds; Bit 11 = cirrus clouds.
        Returns the image with cloudy pixels masked (set to 0 / no-data).
        """
        ee = self._ee
        qa = image.select("QA60")
        cloud_bit_mask  = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask)

    def _get_optical(self, geometry, start: str, end: str):
        ee = self._ee

        def _build_col(cloud_pct: float):
            return (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
                .map(self._mask_clouds_s2)       # per-pixel QA60 cloud mask
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

        col = _build_col(MAX_CLOUD_PCT)
        if col.size().getInfo() == 0:
            log.warning("No S2 at <%.0f%% cloud — retrying at %.0f%%",
                        MAX_CLOUD_PCT, FALLBACK_CLOUD_PCT)
            col = _build_col(FALLBACK_CLOUD_PCT)
            if col.size().getInfo() == 0:
                raise DataAcquisitionError(
                    f"No Sentinel-2 scenes available between {start} and {end}. "
                    "Widen the date range or check the polygon coordinates."
                )
        return col.median().select(S2_BANDS)

    # ── SAR (Sentinel-1) ──────────────────────────────────────────────────────

    def _get_sar(self, geometry, start: str, end: str):
        ee = self._ee
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        if col.size().getInfo() == 0:
            raise DataAcquisitionError(f"No Sentinel-1 scenes between {start} and {end}.")
        return col.median().select(S1_BANDS)

    # ── Hansen mask ────────────────────────────────────────────────────────────

    def _get_hansen_mask(self, geometry) -> np.ndarray:
        """
        Extract post-EUDR (2021-2024) forest loss from Hansen GFC 2024 v1.12.
        Returns a [H, W] binary float32 mask: 1 = loss, 0 = no loss.
        """
        ee = self._ee
        hansen = ee.Image(HANSEN_ASSET)
        loss_yr = hansen.select("lossyear")
        # Loss years 21-24 = 2021-2024 (post EUDR reference date Dec 31 2020)
        post_eudr = loss_yr.gte(21).And(loss_yr.lte(24)).rename("post_eudr_loss")
        raw = self._to_numpy(post_eudr, geometry, ["post_eudr_loss"])
        return raw[:, :, 0]

    # ── Hansen tree-cover baseline ─────────────────────────────────────────────

    def _get_tree_cover(self, geometry) -> dict:
        """
        Retrieve baseline forest cover statistics from Hansen GFC 2024.

        Uses the ``treecover2000`` band — canopy closure for all vegetation
        taller than 5 m as of year 2000 (the reference baseline for loss).

        EUDR Article 2(4) defines forest as trees capable of reaching >5 m with
        ≥10% canopy cover.  We also flag if cover exceeds 30% as a proxy for
        mature canopy typically associated with trees ≥15 m.

        Returns
        -------
        dict with keys:
          mean_canopy_pct     – mean % canopy cover across the polygon
          forest_pct          – fraction of pixels with canopy ≥10% (EUDR threshold)
          mature_forest_pct   – fraction of pixels with canopy ≥30% (~15 m proxy)
          is_eudr_forest      – True if mean_canopy_pct ≥ FOREST_CANOPY_MIN_PCT
          post_eudr_loss_pct  – % of polygon pixels with confirmed Hansen loss 2021-2024
                                 (independent ground-truth corroboration of NDVI signal)
        """
        ee = self._ee
        try:
            hansen = ee.Image(HANSEN_ASSET)
            tc       = hansen.select("treecover2000")
            loss_yr  = hansen.select("lossyear")

            # treecover2000 stats
            stats = tc.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e8,
            )
            mean_pct = stats.get("treecover2000").getInfo()
            if mean_pct is None:
                raise ValueError("treecover2000 returned None")
            mean_pct = float(mean_pct)

            # Fraction of pixels meeting EUDR forest threshold (≥10% canopy)
            forest_mask = tc.gte(FOREST_CANOPY_MIN_PCT)
            mature_mask = tc.gte(FOREST_CANOPY_HIGH_PCT)
            forest_frac = forest_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e8,
            ).get("treecover2000").getInfo() or 0.0
            mature_frac = mature_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e8,
            ).get("treecover2000").getInfo() or 0.0

            # Hansen GFC confirmed post-EUDR loss (lossyear 21–24 = 2021–2024).
            # This is the gold-standard independent corroboration: a pixel that
            # appears in Hansen lossyear 21-24 is a verified satellite observation
            # of forest clearing after the EUDR reference date.
            post_eudr_mask = loss_yr.gte(21).And(loss_yr.lte(24))
            post_eudr_frac = post_eudr_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e8,
            ).get("lossyear").getInfo() or 0.0
            post_eudr_loss_pct = round(float(post_eudr_frac) * 100, 2)

            log.info(
                "Hansen GFC: canopy=%.1f%%  EUDR-forest=%s  post-EUDR loss=%.2f%%",
                mean_pct,
                mean_pct >= FOREST_CANOPY_MIN_PCT,
                post_eudr_loss_pct,
            )

            return {
                "mean_canopy_pct":    round(float(mean_pct), 1),
                "forest_pct":         round(float(forest_frac) * 100, 1),
                "mature_forest_pct":  round(float(mature_frac) * 100, 1),
                "is_eudr_forest":     mean_pct >= FOREST_CANOPY_MIN_PCT,
                "post_eudr_loss_pct": post_eudr_loss_pct,
            }
        except Exception as exc:
            log.warning("Tree-cover baseline unavailable: %s", exc)
            return {
                "mean_canopy_pct":    None,
                "forest_pct":         None,
                "mature_forest_pct":  None,
                "is_eudr_forest":     None,
                "post_eudr_loss_pct": None,
            }

    # ── WRI drivers ───────────────────────────────────────────────────────────

    def _get_wri_driver(self, geometry) -> dict:
        """
        Retrieve the WRI GDM Drivers of Forest Loss classification for the polygon.

        Returns the modal (most common) driver class across all pixels, plus
        the pixel-level confidence (0–1) for that driver.

        Tries both known asset path variants, then falls back to a
        Hansen-based heuristic if the dataset is inaccessible.

        Returns
        -------
        dict with keys:
          driver_class   – int 1-7 (0 = unknown)
          driver_label   – human-readable string
          confidence     – float 0.0-1.0 (from probability band) or None
          source         – "wri" | "hansen_fallback" | "unknown"
          is_eudr_driver – True if driver is commodity/logging-driven
        """
        ee = self._ee

        def _try_asset(asset_id: str) -> Optional[int]:
            """Attempt to get modal driver value from a GEE asset path."""
            try:
                img = ee.Image(asset_id)
                # Try selecting the classification band by common names
                for band_name in ("driver", "classification", "b1"):
                    try:
                        band = img.select(band_name)
                        modal = band.reduceRegion(
                            reducer=ee.Reducer.mode(),
                            geometry=geometry,
                            scale=1000,
                            maxPixels=1e8,
                        )
                        val = modal.get(band_name).getInfo()
                        if val is not None:
                            log.info("WRI driver: class=%d from asset=%s band=%s",
                                     int(val), asset_id, band_name)
                            return int(val)
                    except Exception:
                        continue
            except Exception as exc:
                log.debug("WRI asset %s unavailable: %s", asset_id, exc)
            return None

        # Try primary asset path, then alternate path
        driver_class = _try_asset(WRI_DRIVERS_ASSET_PRIMARY)
        if driver_class is None:
            log.debug("WRI primary path failed — trying alternate path")
            driver_class = _try_asset(WRI_DRIVERS_ASSET_ALT)

        if driver_class is not None:
            label = WRI_DRIVER_LABELS.get(driver_class, f"Class {driver_class}")
            return {
                "driver_class":   driver_class,
                "driver_label":   label,
                "confidence":     None,   # probability bands require per-band fetch
                "source":         "wri",
                "is_eudr_driver": driver_class in EUDR_COMMODITY_DRIVERS,
            }

        log.warning(
            "WRI GDM Drivers dataset inaccessible — driver classification unavailable. "
            "The compliance engine will apply the EUDR precautionary principle (unknown → RED). "
            "To enable WRI access, ensure the GEE project has read permission on "
            "projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km_v1_2_2001_2024"
        )

        # NOTE: We deliberately do NOT infer a driver class from the Hansen loss
        # fraction alone.  Loss magnitude cannot distinguish commodity-driven
        # deforestation from bark beetle die-off, storm damage, urban development,
        # or any other cause — especially in temperate Europe.  Returning class 0
        # (Unknown) is the only honest answer; the engine's precautionary principle
        # handles it correctly.  Previously this code incorrectly assigned class 1
        # (Permanent agriculture) for loss_frac > 5%, causing false-positive RED
        # flags on hospital campuses, infrastructure corridors, and natural
        # disturbance sites in Central Europe.

        return {
            "driver_class":   0,
            "driver_label":   "Unknown",
            "confidence":     None,
            "source":         "unknown",
            "is_eudr_driver": None,
        }

    # ── JRC EU CropMap V1 (EUCROPMAP) ─────────────────────────────────────────

    def _get_eucropmap(self, geometry) -> dict:
        """
        Query the JRC EU CropMap V1 (EUCROPMAP) for the modal crop class inside
        the polygon at two time points:

          2018 — pre-EUDR baseline (EUDR reference date: 31 December 2020)
          2022 — post-EUDR snapshot (most recent EUCROPMAP epoch)

        Returns
        -------
        dict with keys:
          class_2018                   – int modal crop class for 2018 (None if outside EU / unavailable)
          label_2018                   – human-readable class label
          class_2022                   – int modal crop class for 2022
          label_2022                   – human-readable class label
          is_eudr_commodity            – True if the 2022 crop is a EUDR Annex I commodity (soya=233)
          is_pre_existing_agricultural – True if 2018 classification is an arable class:
                                         the land was already agricultural BEFORE the EUDR cutoff,
                                         so any NDVI change is likely crop rotation, not deforestation.
          is_continuous_agricultural   – True if BOTH 2018 and 2022 are arable classes:
                                         confirms ongoing crop rotation with no land-cover conversion.
          coverage                     – float 0–1: fraction of polygon pixels that are cropland in 2022
                                         (None when dataset is unavailable / polygon outside EU).

        Notes
        -----
        - EUCROPMAP covers EU member states only.  Polygons outside the EU will return
          all None values; the engine treats this as "no crop data" and ignores the signal.
        - The dataset has 10 m resolution but the GEE reducer uses mode() over the polygon,
          returning the most frequent crop class.
        - Soya (class 233) is the only EUDR Annex I commodity present in EUCROPMAP.
          All other cropland classes are evidence of pre-existing agricultural land.
        """
        ee = self._ee
        _null = {
            "class_2018":                   None,
            "label_2018":                   None,
            "class_2022":                   None,
            "label_2022":                   None,
            "is_eudr_commodity":            None,
            "is_pre_existing_agricultural": None,
            "is_continuous_agricultural":   None,
            "coverage":                     None,
        }

        try:
            col = ee.ImageCollection(EUCROPMAP_ASSET)

            def _modal_class(year: int) -> Optional[int]:
                """Return the modal classification value for the given year."""
                img = col.filter(ee.Filter.calendarRange(year, year, "year")).first()
                if img is None:
                    return None
                stats = img.select("classification").reduceRegion(
                    reducer=ee.Reducer.mode(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e8,
                )
                val = stats.get("classification").getInfo()
                return int(val) if val is not None else None

            def _cropland_coverage(year: int) -> Optional[float]:
                """Return the fraction of pixels classified as any arable class in the given year."""
                img = col.filter(ee.Filter.calendarRange(year, year, "year")).first()
                if img is None:
                    return None
                # Build a list of arable values and create a binary arable mask
                arable_list = list(EUCROPMAP_ARABLE_CLASSES)
                # Use remap: arable → 1, everything else → 0
                arable_mask = img.select("classification").remap(
                    arable_list, [1] * len(arable_list), defaultValue=0
                )
                frac = arable_mask.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e8,
                ).get("remapped").getInfo()
                return float(frac) if frac is not None else None

            class_2018 = _modal_class(2018)
            class_2022 = _modal_class(2022)
            coverage   = _cropland_coverage(2022)

            label_2018 = EUCROPMAP_CLASSES.get(class_2018, "Unknown") if class_2018 is not None else None
            label_2022 = EUCROPMAP_CLASSES.get(class_2022, "Unknown") if class_2022 is not None else None

            is_pre_existing = (class_2018 in EUCROPMAP_ARABLE_CLASSES) if class_2018 is not None else None
            is_continuous   = (
                (class_2018 in EUCROPMAP_ARABLE_CLASSES) and (class_2022 in EUCROPMAP_ARABLE_CLASSES)
            ) if (class_2018 is not None and class_2022 is not None) else None
            is_eudr_crop = (class_2022 in EUCROPMAP_EUDR_CLASSES) if class_2022 is not None else None

            log.info(
                "EUCROPMAP result: 2018=%s(%s) 2022=%s(%s) pre_existing=%s continuous=%s eudr_crop=%s coverage=%.0f%%",
                class_2018, label_2018,
                class_2022, label_2022,
                is_pre_existing, is_continuous, is_eudr_crop,
                (coverage * 100 if coverage is not None else 0),
            )

            return {
                "class_2018":                   class_2018,
                "label_2018":                   label_2018,
                "class_2022":                   class_2022,
                "label_2022":                   label_2022,
                "is_eudr_commodity":            is_eudr_crop,
                "is_pre_existing_agricultural": is_pre_existing,
                "is_continuous_agricultural":   is_continuous,
                "coverage":                     coverage,
            }

        except Exception as exc:
            log.warning(
                "EUCROPMAP unavailable (polygon may be outside EU, or GEE dataset not accessible): %s",
                exc,
            )
            return _null

    # ── Download ───────────────────────────────────────────────────────────────

    def _to_numpy(self, image, geometry, bands: list[str]) -> np.ndarray:
        """
        Download a GEE image as a numpy array.

        Adaptive scale selection keeps the request under GEE's ~50 MB limit:
          - Farm-scale polygons  (<~5 km):  10 m native S2 resolution
          - Buffer zones (~5 km radius):    ~50 m avoids size errors
          - Very large regions:             Scale up further as needed

        Target: ≤ 512 px per side (sufficient, preprocessor resizes to 224×224).
        """
        bounds = geometry.bounds().getInfo()["coordinates"][0]
        lons = [pt[0] for pt in bounds]
        lats = [pt[1] for pt in bounds]
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)

        # Conservative pixel budget: 512 px per side keeps well under 50 MB
        MAX_PX = 512
        deg_per_pixel = RESOLUTION_M / 111_000
        est_w = lon_span / deg_per_pixel
        est_h = lat_span / deg_per_pixel
        scale = RESOLUTION_M
        if max(est_w, est_h) > MAX_PX:
            scale = int(RESOLUTION_M * max(est_w, est_h) / MAX_PX)
            # Round up to nearest 10 m for clean GEE handling
            scale = max(scale, RESOLUTION_M)
            scale = (scale + 9) // 10 * 10
            log.debug("Large region — using scale=%d m (est pixels: %.0f × %.0f)",
                      scale, est_w, est_h)

        url: str = image.getDownloadURL({
            "scale": scale,
            "region": geometry,
            "format": "NPY",
        })
        try:
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                raw = resp.read()
        except Exception as exc:
            raise DataAcquisitionError(f"GEE image download failed: {exc}") from exc

        data: np.ndarray = np.load(io.BytesIO(raw))
        h, w = data.shape
        arr = np.zeros((h, w, len(bands)), dtype=np.float32)
        for i, band in enumerate(bands):
            if data.dtype.names and band in data.dtype.names:
                arr[:, :, i] = data[band]
            elif not data.dtype.names:
                arr[:, :, i] = data
        return arr
