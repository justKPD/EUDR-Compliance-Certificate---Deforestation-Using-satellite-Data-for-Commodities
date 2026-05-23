"""
LoRA fine-tuning of PrithviChangeDetector on Hansen Global Forest Change labels.

The Hansen dataset (UMD/hansen/global_forest_change_2023_v1_11) provides:
  - ``lossyear``: year of forest loss 2001–2023 (0 = no loss)
  - ``loss``:     binary loss mask

We use Hansen as weak supervision: any pixel with loss year 2021-2023
(post-EUDR cutoff) is labelled "changed"; pixels with no loss are "unchanged".
This gives a free, globally consistent label set.

Usage
-----
    from eudr.models.fine_tuning import FineTuner
    from eudr.models.change_detector import PrithviChangeDetector

    model = PrithviChangeDetector(freeze_backbone=False)
    model.apply_lora()

    tuner = FineTuner(model)
    tuner.run(
        polygons=[polygon1, polygon2, ...],   # GeoJSONPolygon list
        epochs=5,
        output_path="checkpoints/prithvi_eudr.pt",
    )
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eudr.acquisition.satellite import SatelliteDataAcquisition
from eudr.exceptions import ModelInferenceError
from eudr.models.change_detector import PrithviChangeDetector
from eudr.preprocessing.pipeline import GeospatialPreprocessor
from eudr.schemas import GeoJSONPolygon

logger = logging.getLogger(__name__)

# Hansen collection on GEE
HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"
POST_EUDR_YEARS = list(range(21, 24))  # 2021, 2022, 2023 (Hansen stores 2-digit years)


class HansenPatchDataset(Dataset):
    """
    In-memory dataset of (t1, t2, label) patch triples.

    Each label is a binary [1, 1, H, W] tensor:
      1 = forest loss (post-EUDR cutoff, i.e. lossyear 21-23)
      0 = stable / no loss

    Parameters
    ----------
    patches:
        List of (baseline_array, current_array, loss_mask) triples as numpy arrays.
    preprocessor:
        GeospatialPreprocessor instance for tensor conversion.
    """

    def __init__(
        self,
        patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        preprocessor: GeospatialPreprocessor,
    ) -> None:
        self.patches = patches
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline, current, mask = self.patches[idx]
        proc = self.preprocessor.prepare(baseline, current)
        # mask: [H, W] → [1, 1, H, W]  (binary float)
        label = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return proc.t1_tensor.squeeze(0), proc.t2_tensor.squeeze(0), label.squeeze(0)


class FineTuner:
    """
    Wraps the training loop for LoRA fine-tuning of PrithviChangeDetector.

    Parameters
    ----------
    model:
        PrithviChangeDetector with LoRA already applied (call model.apply_lora() first).
    learning_rate:
        AdamW learning rate for the LoRA + change-head parameters.
    """

    def __init__(
        self,
        model: PrithviChangeDetector,
        learning_rate: float = 1e-4,
    ) -> None:
        self.model = model
        self.device = model._device
        # Only optimise trainable parameters (LoRA adapters + change head)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "No trainable parameters found. Call model.apply_lora() before FineTuner."
            )
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
        # Weighted BCE to handle class imbalance (forest loss pixels are rare)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        polygons: list[GeoJSONPolygon],
        epochs: int = 5,
        batch_size: int = 4,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Full fine-tuning pipeline:
        1. Fetch Sentinel-2 + Hansen labels for each polygon via GEE
        2. Build HansenPatchDataset
        3. Train for `epochs` epochs
        4. Save checkpoint

        Parameters
        ----------
        polygons:
            Production polygons to train on (fetches surrounding area for diversity).
        epochs:
            Number of training epochs (5 is sufficient for MVP adaptation).
        batch_size:
            Number of patches per gradient step.
        output_path:
            Where to save the model checkpoint.  Defaults to ``checkpoints/`` folder.

        Returns
        -------
        Path to the saved checkpoint.
        """
        logger.info("Collecting Hansen training patches for %d polygons …", len(polygons))
        patches = self._collect_patches(polygons)

        if not patches:
            raise ValueError("No training patches collected. Check GEE connectivity.")

        logger.info("Building dataset from %d patches …", len(patches))
        preprocessor = GeospatialPreprocessor(device=self.device)
        dataset = HansenPatchDataset(patches, preprocessor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        logger.info("Starting fine-tuning for %d epochs …", epochs)
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = self._train_epoch(loader, epoch)
            logger.info("  Epoch %d/%d — loss: %.4f", epoch, epochs, epoch_loss)

        checkpoint_path = self._save(output_path)
        logger.info("Fine-tuning complete. Checkpoint: %s", checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        model: PrithviChangeDetector,
        checkpoint_path: Path,
    ) -> None:
        """Load a saved fine-tuned checkpoint into an existing model."""
        state = torch.load(checkpoint_path, map_location=model._device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        logger.info("Loaded checkpoint from %s", checkpoint_path)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        total_loss = 0.0
        for t1, t2, labels in loader:
            t1 = t1.to(self.device)
            t2 = t2.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                output = self.model(t1.unsqueeze(0) if t1.ndim == 3 else t1,
                                    t2.unsqueeze(0) if t2.ndim == 3 else t2)
                # BCEWithLogitsLoss works on raw logits before sigmoid
                # We apply the head without final sigmoid during training
                loss = self.criterion(output.change_map, labels)

            loss.backward()
            # Gradient clipping for stable LoRA training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _collect_patches(
        self,
        polygons: list[GeoJSONPolygon],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        For each polygon, fetch Sentinel-2 imagery and the Hansen loss mask from GEE.
        Returns a list of (baseline, current, mask) numpy triples.
        """
        try:
            import ee  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("earthengine-api required for fine-tuning data collection.") from exc

        acquirer = SatelliteDataAcquisition()
        patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for i, polygon in enumerate(polygons):
            logger.info("  Fetching patch %d/%d …", i + 1, len(polygons))
            try:
                raw = acquirer.fetch_baseline_and_current(polygon)
                mask = self._fetch_hansen_mask(polygon)
                if mask is not None:
                    patches.append((raw["baseline_2020"], raw["current"], mask))
            except Exception as exc:
                logger.warning("  Skipping polygon %d: %s", i, exc)

        return patches

    @staticmethod
    def _fetch_hansen_mask(polygon: GeoJSONPolygon) -> Optional[np.ndarray]:
        """
        Download the Hansen forest-loss mask for a polygon from GEE.
        Returns binary [H, W] float32 array: 1 = post-EUDR loss, 0 = stable.
        """
        try:
            import ee
            import io
            import urllib.request

            geometry = ee.Geometry.Polygon(polygon.coordinates)
            hansen = ee.Image(HANSEN_ASSET)

            # lossyear band: integer year offset from 2000 (21 = 2021)
            loss_year = hansen.select("lossyear")
            post_eudr_mask = loss_year.gte(21).And(loss_year.lte(23))

            url = post_eudr_mask.getDownloadURL({
                "scale": 10,
                "region": geometry,
                "format": "NPY",
            })
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                data = np.load(io.BytesIO(resp.read()))

            # Structured → plain array
            if data.dtype.names:
                return data[data.dtype.names[0]].astype(np.float32)
            return data.astype(np.float32)

        except Exception as exc:
            logger.warning("Hansen mask fetch failed: %s", exc)
            return None

    def _save(self, output_path: Optional[Path]) -> Path:
        if output_path is None:
            Path("checkpoints").mkdir(exist_ok=True)
            output_path = Path("checkpoints/prithvi_eudr_lora.pt")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, output_path)
        return output_path
