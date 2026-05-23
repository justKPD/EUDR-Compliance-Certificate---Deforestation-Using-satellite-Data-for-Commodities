"""
Siamese Change-Detection Network backed by the Prithvi-100M geospatial transformer.

Architecture
------------
                   ┌──────────────┐   ┌──────────────┐
  x_t1 ──────────▶│              │   │              │◀── x_t2
                   │  Prithvi    │   │  Prithvi    │
                   │  backbone   │   │  backbone   │
                   │ (shared wts)│   │ (shared wts)│
                   └──────┬───────┘   └──────┬───────┘
                          │                   │
                          └────── cat ────────┘
                                   │
                          ┌────────▼────────┐
                          │  Change head    │
                          │ (CNN, Sigmoid)  │
                          └────────┬────────┘
                                   │
                         change_map [B, 1, H, W]

Prithvi-100M is loaded directly from its raw .pt checkpoint using the
prithvi_mae.py architecture (bypassing the broken HF AutoModel path that
fails on config.json's "num_labels": null).

For single-frame inference the input [B, 6, H, W] is tiled 3× along the
temporal axis to produce [B, 6, 3, 224, 224] — the format expected by the
Prithvi ViT encoder.  The three temporal outputs are mean-pooled back to a
[B, 768, 14, 14] spatial feature map before being concatenated for the
change head.

Fallback: if Prithvi is unavailable, a ViT-Base/16 is used instead.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eudr.config import settings
from eudr.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

# Prithvi-100M hidden dimension
HIDDEN_DIM: int = 768

# Canonical Prithvi input resolution
_PRITHVI_IMG_SIZE: int = 224

# Prithvi config params (from config.json)
_PRITHVI_CFG = dict(
    img_size=224,
    patch_size=(1, 16, 16),
    num_frames=3,
    in_chans=6,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.0,
)

# Sentinel-2 band normalisation stats (from Prithvi config.json)
_S2_MEAN = [775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83]
_S2_STD  = [1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51]


@dataclass
class ChangeDetectionOutput:
    """Raw output from a single forward pass."""

    change_map: torch.Tensor  # [B, 1, H, W] probabilities in [0, 1]


class _ChangeHead(nn.Sequential):
    """
    Lightweight CNN head that consumes concatenated Prithvi features
    and produces a per-pixel change-probability map.
    """

    def __init__(self, in_channels: int = HIDDEN_DIM * 2) -> None:
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output ∈ [0, 1]
        )


class PrithviChangeDetector(nn.Module):
    """
    Siamese change-detection model using the Prithvi-100M backbone.

    Parameters
    ----------
    freeze_backbone:
        Freeze all backbone parameters (inference-only mode).
        Set to False only after calling ``apply_lora()``.
    """

    # Whether the loaded backbone is the real Prithvi encoder (True)
    # or the ViT-Base fallback (False).
    _is_prithvi: bool = False

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading Prithvi-100M backbone …")
        self.backbone, self._is_prithvi = self._load_backbone(device)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen (inference mode).")

        hidden_dim = _PRITHVI_CFG["embed_dim"] if self._is_prithvi else HIDDEN_DIM
        self.change_head = _ChangeHead(in_channels=hidden_dim * 2).to(device)
        self._device = device
        self._hidden_dim = hidden_dim

    # ── Public methods ─────────────────────────────────────────────────────────

    def forward(
        self,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        ndvi_change: Optional[np.ndarray] = None,
    ) -> ChangeDetectionOutput:
        """
        Parameters
        ----------
        x_t1, x_t2:
            Float tensors [B, C, H, W] on the same device as the model.
        ndvi_change:
            Optional [H, W] float32 array of NDVI change (current - baseline).
            When provided, the output is a 10/90 blend of ViT output and an
            NDVI-derived physics prior.

        Returns
        -------
        ChangeDetectionOutput containing the [B, 1, H, W] probability map.
        """
        try:
            with torch.cuda.amp.autocast(enabled=self._device.type == "cuda"):
                feat_t1 = self._extract_features(x_t1)
                feat_t2 = self._extract_features(x_t2)
                combined = torch.cat([feat_t1, feat_t2], dim=1)
                vit_map = self.change_head(combined)

            if ndvi_change is not None:
                ndvi_prior = self._ndvi_to_prior(ndvi_change, self._device)
                H, W = ndvi_prior.shape[-2], ndvi_prior.shape[-1]
                vit_up = F.interpolate(vit_map, size=(H, W), mode="bilinear", align_corners=False)
                # 10% ViT + 90% NDVI physics prior at full resolution.
                change_map = 0.10 * vit_up + 0.90 * ndvi_prior
                logger.debug(
                    "NDVI-ViT hybrid active: 90%% NDVI physics prior + 10%% ViT output "
                    "(full-res %dx%d)", H, W,
                )
            else:
                change_map = vit_map

            return ChangeDetectionOutput(change_map=change_map)
        except Exception as exc:
            raise ModelInferenceError(f"Change detection forward pass failed: {exc}") from exc

    @staticmethod
    def _ndvi_to_prior(
        ndvi_change: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Map NDVI change to a pixel-wise forest-loss probability prior at
        FULL SATELLITE RESOLUTION.

        Sigmoid formula: sigmoid(-ndvi_change * 10 - 1.5)
          ndvi_change = -0.5  →  prob ≈ 0.98   (strong loss signal)
          ndvi_change = -0.10 →  prob ≈ 0.39   (loss onset)
          ndvi_change =  0.0  →  prob ≈ 0.18   (neutral / no change)
          ndvi_change = +0.3  →  prob ≈ 0.01   (regrowth / vegetation gain)
        """
        ndvi_t = torch.from_numpy(ndvi_change.astype(np.float32)).to(device)
        prior = torch.sigmoid(-ndvi_t * 10.0 - 1.5)
        return prior.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)  # [1, 1, H, W]

    def apply_lora(self) -> None:
        """
        Apply LoRA adapters to the backbone for parameter-efficient fine-tuning.
        Only ~1-2% of parameters become trainable.
        """
        try:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. "
                "Install it with: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "lora_" in name

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "LoRA applied: %d / %d parameters trainable (%.2f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_backbone(device: torch.device) -> tuple[nn.Module, bool]:
        """
        Load the geospatial backbone.

        Strategy 1: Prithvi-100M from raw .pt checkpoint (our custom loader,
                     bypasses the broken HF AutoModel / config.json path).
        Strategy 2: ViT-Base/16 fallback (same 768-dim space, public HF model).

        Returns
        -------
        (backbone_module, is_prithvi)
        """
        # ── Strategy 1: direct .pt checkpoint load ─────────────────────────────
        try:
            backbone, ok = PrithviChangeDetector._load_prithvi_direct(device)
            if ok:
                return backbone, True
        except Exception as exc:
            logger.warning("Prithvi direct load failed: %s", exc)

        # ── Strategy 2: HF AutoModel (may work on newer transformers) ──────────
        try:
            backbone = PrithviChangeDetector._load_prithvi_hf(device)
            return backbone, True
        except Exception as exc:
            logger.warning("Prithvi HF AutoModel load failed: %s", exc)

        # ── Strategy 3: ViT-Base/16 fallback ───────────────────────────────────
        logger.warning("Prithvi-100M unavailable — falling back to ViT-Base/16.")
        try:
            from transformers import AutoModel  # noqa: PLC0415
            backbone = AutoModel.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                dtype=torch.float32,
            ).to(device)
            logger.info("Fallback ViT-Base/16 loaded.")
            return backbone, False
        except Exception as exc:
            raise ModelInferenceError(
                f"All backbone loading strategies failed. Last error: {exc}"
            ) from exc

    @staticmethod
    def _load_prithvi_direct(device: torch.device) -> tuple[nn.Module, bool]:
        """
        Load Prithvi-100M encoder directly from the cached .pt checkpoint,
        bypassing the broken HF AutoModel / config.json path.

        The model file ships as a PrithviMAE full state-dict (encoder + decoder).
        We load PrithviMAE, load the weights, then return only the encoder
        (PrithviViT) to save memory and avoid the decoder at inference time.
        """
        # Locate the .pt file in the HF cache
        ckpt_path = _find_prithvi_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("Prithvi_EO_V1_100M.pt not found in HF cache.")

        mae_module_path = Path(__file__).parent / "prithvi_mae.py"
        if not mae_module_path.exists():
            # Also look in the HF snapshot alongside the .pt file
            mae_module_path = Path(ckpt_path).parent / "prithvi_mae.py"
        if not mae_module_path.exists():
            raise FileNotFoundError("prithvi_mae.py not found alongside checkpoint.")

        # Dynamically import the prithvi_mae module
        import importlib.util
        spec = importlib.util.spec_from_file_location("prithvi_mae", str(mae_module_path))
        prithvi_mae_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prithvi_mae_mod)

        # Instantiate PrithviMAE with the known config params
        mae = prithvi_mae_mod.PrithviMAE(**_PRITHVI_CFG).to(device)

        # Load the checkpoint — map_location ensures CPU weights for CPU devices
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

        # The .pt contains full PrithviMAE weights (encoder.* + decoder.*)
        missing, unexpected = mae.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Prithvi .pt missing keys (expected pos_embed buffers): %s", missing[:5])
        if unexpected:
            logger.debug("Prithvi .pt unexpected keys: %s", unexpected[:5])

        # Return only the encoder (PrithviViT) — we never need the MAE decoder
        encoder = mae.encoder
        logger.info(
            "Prithvi-100M encoder loaded directly from %s  "
            "(embed_dim=%d, depth=%d, num_frames=%d, in_chans=%d)",
            ckpt_path, _PRITHVI_CFG["embed_dim"], _PRITHVI_CFG["depth"],
            _PRITHVI_CFG["num_frames"], _PRITHVI_CFG["in_chans"],
        )
        return encoder, True

    @staticmethod
    def _load_prithvi_hf(device: torch.device) -> nn.Module:
        """
        Attempt to load Prithvi via HF AutoModel with the num_labels=None
        monkey-patch applied.
        """
        from transformers import AutoModel, PretrainedConfig  # noqa: PLC0415

        # Patch: _create_id_label_maps crashes when num_labels is None
        _orig = PretrainedConfig._create_id_label_maps

        def _patched(self_cfg):
            if getattr(self_cfg, "num_labels", None) is None:
                self_cfg.num_labels = 0
            return _orig(self_cfg)

        PretrainedConfig._create_id_label_maps = _patched
        try:
            backbone = AutoModel.from_pretrained(
                settings.prithvi_model_id,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                dtype=torch.float32,
            ).to(device)
            logger.info("Prithvi-100M loaded via HF AutoModel (monkey-patch).")
            return backbone
        finally:
            PretrainedConfig._create_id_label_maps = _orig

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run x through the backbone and return a spatial feature map [B, D, H', W'].

        Handles:
          - PrithviViT encoder  → input [B, 6, H, W] tiled to [B, 6, 3, 224, 224]
          - HF AutoModel (Prithvi) → pixel_values keyword
          - ViT-Base fallback   → 3-channel, last_hidden_state / pooler_output
        """
        if self._is_prithvi:
            return self._extract_prithvi_features(x)
        else:
            return self._extract_vit_features(x)

    def _extract_prithvi_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the Prithvi encoder.

        Input:  [B, C, H, W]  (C may be 6 or ≤6 depending on acquisition)
        Output: [B, 768, 14, 14]
        """
        B = x.shape[0]
        device = x.device

        # ── 1. Normalise bands to Prithvi's training statistics ────────────────
        mean = torch.tensor(_S2_MEAN, dtype=torch.float32, device=device).view(1, 6, 1, 1)
        std  = torch.tensor(_S2_STD,  dtype=torch.float32, device=device).view(1, 6, 1, 1)

        # Pad or trim to exactly 6 channels
        if x.shape[1] < 6:
            pad = torch.zeros(B, 6 - x.shape[1], x.shape[2], x.shape[3], device=device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > 6:
            x = x[:, :6, :, :]

        x = (x - mean) / (std + 1e-6)

        # ── 2. Resize to 224×224 (Prithvi training resolution) ─────────────────
        if x.shape[-2:] != (_PRITHVI_IMG_SIZE, _PRITHVI_IMG_SIZE):
            x = F.interpolate(
                x,
                size=(_PRITHVI_IMG_SIZE, _PRITHVI_IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # ── 3. Tile single frame to 3 temporal frames: [B,6,H,W]→[B,6,3,H,W] ──
        x_3t = x.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()

        # ── 4. Run encoder (forward_features returns list of block outputs) ─────
        block_outputs = self.backbone.forward_features(x_3t)
        last = block_outputs[-1]  # [B, 1+T*H*W/patches, D] = [B, 589, 768]

        # ── 5. Drop CLS, mean-pool over temporal tokens → [B, 768, 14, 14] ─────
        seq = last[:, 1:, :]          # [B, 588, 768]  (3 frames × 14×14 patches)
        n_patches_per_frame = seq.shape[1] // _PRITHVI_CFG["num_frames"]  # 196
        h = w = int(n_patches_per_frame ** 0.5)                            # 14

        # Reshape: [B, 3, 196, 768] → mean over time → [B, 196, 768]
        seq = seq.view(B, _PRITHVI_CFG["num_frames"], n_patches_per_frame, seq.shape[-1])
        seq = seq.mean(dim=1)          # [B, 196, 768]

        # [B, 196, 768] → [B, 768, 14, 14]
        feat = seq.permute(0, 2, 1).reshape(B, _PRITHVI_CFG["embed_dim"], h, w)
        return feat

    def _extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the ViT-Base fallback backbone.

        Input:  [B, C, H, W]
        Output: [B, 768, 14, 14]
        """
        # ViT-Base expects exactly 3 channels (RGB)
        num_backbone_channels = getattr(
            self.backbone.config, "num_channels",
            getattr(self.backbone.config, "in_chans", 3),
        )
        if x.shape[1] != num_backbone_channels:
            x = x[:, :num_backbone_channels, :, :]

        # ViT-Base expects 224×224
        if x.shape[-2:] != (_PRITHVI_IMG_SIZE, _PRITHVI_IMG_SIZE):
            x = F.interpolate(
                x,
                size=(_PRITHVI_IMG_SIZE, _PRITHVI_IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        out = self.backbone(x)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            seq = out.last_hidden_state          # [B, N+1, D]
            n_patches = int((x.shape[-1] // 16) ** 2)
            if seq.shape[1] == n_patches + 1:
                seq = seq[:, 1:, :]              # drop CLS → [B, N, D]
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            seq = out.pooler_output.unsqueeze(1)  # [B, 1, D]
        else:
            raise ModelInferenceError("ViT backbone returned unrecognised output format.")

        b, n, d = seq.shape
        h = w = int(n ** 0.5)
        if h * w != n:
            h = w = int(n ** 0.5)
            seq = seq[:, : h * w, :]
        return seq.permute(0, 2, 1).reshape(b, d, h, w)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _find_prithvi_checkpoint() -> Optional[str]:
    """
    Locate Prithvi_EO_V1_100M.pt in the HuggingFace cache.
    Returns the path string, or None if not found.
    """
    import os

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_root = cache_root / "hub"

    # Walk through model snapshots
    for candidate in hub_root.rglob("Prithvi_EO_V1_100M.pt"):
        return str(candidate)

    # Check the configured model_id path directly
    model_id_slug = settings.prithvi_model_id.replace("/", "--")
    alt_root = hub_root / f"models--{model_id_slug}"
    if alt_root.exists():
        for candidate in alt_root.rglob("Prithvi_EO_V1_100M.pt"):
            return str(candidate)

    return None
