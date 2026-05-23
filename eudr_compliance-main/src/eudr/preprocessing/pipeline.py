"""
OpenCV-based preprocessing pipeline.

Converts raw GEE numpy arrays → PyTorch tensors for Prithvi-100M,
and computes NDVI, SAR difference, and change-visualisation artefacts
that feed both the AI engine and the PDF report.

Band order expected (Sentinel-2):
  [0] B2  Blue   490 nm
  [1] B3  Green  560 nm
  [2] B4  Red    665 nm
  [3] B8A NIR    865 nm  ← narrow NIR (not B08)
  [4] B11 SWIR1  1610 nm
  [5] B12 SWIR2  2190 nm
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from eudr.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

PRITHVI_INPUT_SIZE: int = 224
S2_SCALE: float = 10_000.0

# Band indices in the Sentinel-2 array
IDX_RED = 2
IDX_NIR = 3


@dataclass
class PreprocessedData:
    """All artefacts produced by the preprocessing pipeline."""

    t1_tensor: torch.Tensor           # [1, C, 224, 224] baseline  (Prithvi input)
    t2_tensor: torch.Tensor           # [1, C, 224, 224] current   (Prithvi input)
    diff_visualization: np.ndarray    # [224, 224, 3]  uint8 BGR enhanced diff map
    ndvi_baseline: np.ndarray         # [H, W] float32 NDVI of baseline
    ndvi_current: np.ndarray          # [H, W] float32 NDVI of current
    ndvi_change: np.ndarray           # [H, W] float32 NDVI delta (current - baseline)
    sar_diff: Optional[np.ndarray]    # [H, W] float32 SAR VV difference (or None)
    original_shape: tuple[int, int]   # (H, W) before resize


class GeospatialPreprocessor:
    """
    Prepares Sentinel-2 (and optionally Sentinel-1 SAR) data for Prithvi-100M.

    Steps
    -----
    1. NaN / Inf cleaning
    2. Reflectance normalisation to [0, 1]
    3. NDVI computation on native resolution
    4. Resize to 224 × 224 (INTER_LINEAR)
    5. Channel-first tensor [1, C, H, W]
    6. SAR VV difference map (if SAR data provided)
    7. Enhanced RGB diff visualisation for PDF
    """

    def __init__(
        self,
        target_size: int = PRITHVI_INPUT_SIZE,
        device: Optional[torch.device] = None,
    ) -> None:
        self.target_size = target_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Public API ─────────────────────────────────────────────────────────────

    def prepare(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        sar_baseline: Optional[np.ndarray] = None,
        sar_current: Optional[np.ndarray] = None,
    ) -> PreprocessedData:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        baseline, current:
            Raw Sentinel-2 arrays [H, W, 6] (DN values, typically 0–10 000).
        sar_baseline, sar_current:
            Optional Sentinel-1 arrays [H, W, 2] (VV, VH in dB).
        """
        self._validate(baseline, current)

        t1 = self._to_tensor(baseline)
        t2 = self._to_tensor(current)

        ndvi_b = self._ndvi(baseline)
        ndvi_c = self._ndvi(current)
        ndvi_change = ndvi_c - ndvi_b

        sar_diff: Optional[np.ndarray] = None
        if sar_baseline is not None and sar_current is not None:
            sar_diff = self._sar_difference(sar_baseline, sar_current)

        diff_vis = self._diff_visualization(baseline, current, ndvi_change)

        return PreprocessedData(
            t1_tensor=t1,
            t2_tensor=t2,
            diff_visualization=diff_vis,
            ndvi_baseline=ndvi_b,
            ndvi_current=ndvi_c,
            ndvi_change=ndvi_change,
            sar_diff=sar_diff,
            original_shape=(baseline.shape[0], baseline.shape[1]),
        )

    # ── NDVI ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _ndvi(arr: np.ndarray) -> np.ndarray:
        """
        Compute NDVI = (NIR - Red) / (NIR + Red) on native resolution.
        Returns [H, W] float32 in [-1, 1].  Mask out-of-range DN as NaN.
        """
        red = arr[:, :, IDX_RED].astype(np.float32)
        nir = arr[:, :, IDX_NIR].astype(np.float32)

        # Scale if DN values (0–10000) not yet normalised
        if red.max() > 2.0 or nir.max() > 2.0:
            red = red / S2_SCALE
            nir = nir / S2_SCALE

        denominator = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
        return ndvi.astype(np.float32)

    # ── SAR difference ────────────────────────────────────────────────────────

    @staticmethod
    def _sar_difference(sar_b: np.ndarray, sar_c: np.ndarray) -> np.ndarray:
        """
        Compute VV channel difference (current - baseline) in dB space.
        A significant drop in backscatter indicates forest loss.
        Returns [H, W] float32.
        """
        vv_b = sar_b[:, :, 0].astype(np.float32)
        vv_c = sar_c[:, :, 0].astype(np.float32)
        diff = vv_c - vv_b
        # Resize to common size if shapes differ
        if diff.shape != vv_b.shape:
            diff = cv2.resize(diff, (vv_b.shape[1], vv_b.shape[0]), interpolation=cv2.INTER_LINEAR)
        return diff

    # ── Tensor conversion ─────────────────────────────────────────────────────

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Normalise, resize, and convert [H, W, C] → [1, C, H, W]."""
        normed = self._normalize(arr)
        resized = cv2.resize(normed, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self.device)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if clean.max() > 10.0:
            clean = clean / S2_SCALE
        return np.clip(clean, 0.0, 1.0).astype(np.float32)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _diff_visualization(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        ndvi_change: np.ndarray,
    ) -> np.ndarray:
        """
        Build a three-panel composite BGR image for the PDF report:
          Left   – 2020 baseline False Color Infrared (FIR)
          Centre – NDVI change heatmap (red = loss, green = gain)
          Right  – 2024 current False Color Infrared (FIR)

        False Color Infrared (NIR-Red-Green):
        - Healthy vegetation appears RED
        - Dead/cleared land appears BLUE/WHITE
        - Water appears DARK BLUE
        This is much better for detecting deforestation than RGB.

        Returns uint8 [target_size, target_size * 3, 3].
        """
        b_norm = self._normalize(baseline)
        c_norm = self._normalize(current)

        # Sentinel-2 bands: B2=Blue, B3=Green, B4=Red, B8A=NIR, B11=SWIR1, B12=SWIR2
        # False Color Infrared: NIR → Red channel, Red → Green, Green → Blue
        # For 6-band data: band order is [B2, B3, B4, B8A, B11, B12] = [0,1,2,3,4,5]
        
        def make_fir(img_norm):
            """Create False Color Infrared composite or SWIR fallback"""
            num_bands = img_norm.shape[2] if img_norm.ndim == 3 else 0
            
            # Try to create FIR: NIR (band 4) → Red, Red (band 3) → Green, Green (band 2) → Blue
            if num_bands >= 4:
                nir = img_norm[:, :, 3]  # Band 8A
                red = img_norm[:, :, 2]  # Band 4
                green = img_norm[:, :, 1]  # Band 3
                fir = np.stack([nir, red, green], axis=2)
            elif num_bands >= 3:
                # Use SWIR-based composite if no NIR: SWIR1→Red, NIR→Green, Red→Blue
                if num_bands >= 6:
                    swir1 = img_norm[:, :, 4]  # Band 11
                    nir = img_norm[:, :, 3]   # Band 8A
                    red = img_norm[:, :, 2]   # Band 4
                    fir = np.stack([swir1, nir, red], axis=2)  # SWIR composite
                else:
                    # Fallback to enhanced RGB using SWIR for moisture
                    if num_bands >= 5:
                        swir = img_norm[:, :, 4]
                        nir = img_norm[:, :, 3] if num_bands > 3 else img_norm[:, :, 0]
                        rgb = img_norm[:, :, :3].mean(axis=2)
                        fir = np.stack([swir, nir, rgb], axis=2)
                    else:
                        # Just stretch the available bands
                        fir = np.zeros((img_norm.shape[0], img_norm.shape[1], 3), dtype=np.float32)
                        for i in range(min(num_bands, 3)):
                            fir[:, :, i] = img_norm[:, :, i]
            else:
                # No bands - create blank
                fir = np.zeros((img_norm.shape[0], img_norm.shape[1], 3), dtype=np.float32)
            
            fir = np.clip(fir * 255, 0, 255).astype(np.uint8)
            return cv2.resize(fir, (self.target_size, self.target_size))

        b_fir = make_fir(b_norm)
        c_fir = make_fir(c_norm)

        # Enhanced NDVI change heatmap with explicit deforestation highlighting
        ndvi_small = cv2.resize(ndvi_change, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_LINEAR)
        
        # Create custom diverging colormap: Red (loss) → Black (no change) → Green (gain)
        # Normalize: -1 → 0, 0 → 127, +1 → 255
        ndvi_uint8 = np.clip(((ndvi_small + 1.0) / 2.0 * 255), 0, 255).astype(np.uint8)
        
        # Use COLORMAP_RdYlGn which shows: Red=loss, Yellow=neutral, Green=gain
        colormap = getattr(cv2, "COLORMAP_RdYlGn", cv2.COLORMAP_JET)
        ndvi_heat = cv2.applyColorMap(ndvi_uint8, colormap)

        # Also create a deforestation mask overlay (bright red for significant loss)
        loss_mask = (ndvi_small < -0.1).astype(np.uint8) * 255
        loss_mask_resized = cv2.resize(loss_mask, (self.target_size, self.target_size))
        loss_mask_colored = cv2.cvtColor(loss_mask_resized, cv2.COLOR_GRAY2BGR)
        # Blend the loss mask with the heatmap for emphasis
        ndvi_heat = cv2.addWeighted(ndvi_heat, 0.85, loss_mask_colored, 0.15, 0)

        # Stitch three panels side by side: FIR 2020 | NDVI Change | FIR 2024
        return np.concatenate([b_fir, ndvi_heat, c_fir], axis=1)

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(baseline: np.ndarray, current: np.ndarray) -> None:
        if baseline.ndim != 3 or baseline.shape[2] < 3:
            raise PreprocessingError(f"Unexpected baseline shape: {baseline.shape}")
        if baseline.shape != current.shape:
            raise PreprocessingError(
                f"Shape mismatch: baseline {baseline.shape} vs current {current.shape}"
            )
