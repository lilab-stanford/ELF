"""ELF slide encoder used at inference: interpolate → LayerNorm → 8-head ABMIL.

Matches the paper and the trained ``momentum_enc`` forward.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedABMIL(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.attention_c = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


class ELFSlideEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.ModuleList(
            [BatchedABMIL(embed_dim // num_heads) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, lens: Optional[torch.Tensor] = None):
        """
        Args:
            x: ``[B, N, C]`` patch features (``C`` in {768, 1024, 1280, 1536}).
            lens: ``[B]`` native ``C`` per item; defaults to ``x.shape[-1]``.

        Returns:
            features_dim: ``[B, C]`` — ``softmax(ᾱ)ᵀ X``
            features: ``[B, 768]`` — ``softmax(ᾱ)ᵀ X_768``
            attention: ``[B, 1, N]``
        """
        if x.ndim != 3:
            raise ValueError(f"expected [B, N, C], got {tuple(x.shape)}")
        batch, n_tiles, feat_dim = x.shape
        if lens is None:
            lens = torch.full((batch,), feat_dim, dtype=torch.long, device=x.device)

        x768 = []
        for i in range(batch):
            c = int(lens[i].item())
            x768.append(
                F.interpolate(x[i, :, :c].unsqueeze(0), size=self.embed_dim, mode="linear", align_corners=True).squeeze(0)
            )
        x768 = self.norm(torch.stack(x768, dim=0))

        head_dim = self.embed_dim // self.num_heads
        heads = x768.view(batch, n_tiles, head_dim, self.num_heads)
        logits = torch.stack([self.attn[h](heads[:, :, :, h]) for h in range(self.num_heads)], dim=-1)
        attn = F.softmax(logits.mean(dim=-1).transpose(1, 2), dim=-1)

        feat_768 = torch.bmm(attn, x768)[:, 0]
        feat_native = torch.stack(
            [torch.bmm(attn[i : i + 1], x[i : i + 1, :, : int(lens[i].item())])[0, 0] for i in range(batch)]
        )
        return feat_native, feat_768, attn


def preprocess_patch_features(features: torch.Tensor, foundation_model: Optional[str] = None) -> torch.Tensor:
    x = features.float()
    if (foundation_model or "").lower() == "virchow2" and x.shape[-1] >= 2560:
        x = 0.5 * (x[..., :1280] + x[..., 1280:2560])
    return x


def _unwrap_state_dict(raw) -> OrderedDict:
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in raw.items())


def extract_inference_weights(state_dict: dict) -> OrderedDict:
    """Keep only LayerNorm + 8-head ABMIL (the tensors in the trained forward)."""
    keys = list(state_dict.keys())
    prefix = ""
    if any(k.startswith("momentum_enc.") for k in keys):
        prefix = "momentum_enc."
    keep = ("norm.", "attn.")
    out = OrderedDict(
        (k[len(prefix) :], v)
        for k, v in state_dict.items()
        if k.startswith(prefix) and k[len(prefix) :].startswith(keep)
    )
    if not out:
        raise KeyError(f"no norm/attn weights found; prefixes={sorted({k.split('.')[0] for k in keys})[:12]}")
    return out


def load_encoder(
    checkpoint: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    embed_dim: int = 768,
    num_heads: int = 8,
) -> ELFSlideEncoder:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    weights = extract_inference_weights(_unwrap_state_dict(ckpt))
    model = ELFSlideEncoder(embed_dim=embed_dim, num_heads=num_heads)
    missing, unexpected = model.load_state_dict(weights, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load mismatch missing={missing} unexpected={unexpected}")
    return model.to(device).eval()
