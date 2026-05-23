"""
Unit tests for GeospatialPreprocessor.

No GPU or network required.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eudr.preprocessing.pipeline import GeospatialPreprocessor, PRITHVI_INPUT_SIZE


@pytest.fixture
def preprocessor() -> GeospatialPreprocessor:
    return GeospatialPreprocessor(device=torch.device("cpu"))


def _random_image(h: int = 128, w: int = 128, c: int = 6, scale: float = 1.0) -> np.ndarray:
    return (np.random.rand(h, w, c) * scale).astype(np.float32)


class TestGeospatialPreprocessor:
    def test_output_shapes(self, preprocessor):
        baseline = _random_image()
        current = _random_image()
        result = preprocessor.prepare(baseline, current)

        assert result.t1_tensor.shape == (1, 6, PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE)
        assert result.t2_tensor.shape == (1, 6, PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE)
        # diff_visualization is a 3-panel composite (baseline | NDVI heatmap | current)
        assert result.diff_visualization.shape == (PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE * 3, 3)
        assert result.original_shape == (128, 128)

    def test_normalisation_range(self, preprocessor):
        # DN values > 10 → should be divided by 10 000 and clipped to [0,1]
        baseline = _random_image(scale=10_000.0)
        current = _random_image(scale=10_000.0)
        result = preprocessor.prepare(baseline, current)
        assert result.t1_tensor.min().item() >= 0.0
        assert result.t1_tensor.max().item() <= 1.0

    def test_nan_handling(self, preprocessor):
        baseline = _random_image()
        baseline[0, 0, 0] = float("nan")
        baseline[5, 5, :] = float("inf")
        current = _random_image()
        # Should not raise
        result = preprocessor.prepare(baseline, current)
        assert not torch.isnan(result.t1_tensor).any()
        assert not torch.isinf(result.t1_tensor).any()

    def test_shape_mismatch_raises(self, preprocessor):
        baseline = _random_image(128, 128)
        current = _random_image(64, 64)  # different spatial size
        with pytest.raises(Exception, match="Shape mismatch"):
            preprocessor.prepare(baseline, current)

    def test_wrong_ndim_raises(self, preprocessor):
        flat = np.random.rand(128, 128).astype(np.float32)
        current = _random_image()
        with pytest.raises(Exception):
            preprocessor.prepare(flat, current)  # type: ignore[arg-type]

    def test_diff_visualization_is_uint8(self, preprocessor):
        baseline = _random_image()
        current = _random_image()
        result = preprocessor.prepare(baseline, current)
        assert result.diff_visualization.dtype == np.uint8

    def test_device_placement(self):
        proc = GeospatialPreprocessor(device=torch.device("cpu"))
        baseline = _random_image()
        current = _random_image()
        result = proc.prepare(baseline, current)
        assert result.t1_tensor.device.type == "cpu"
        assert result.t2_tensor.device.type == "cpu"
