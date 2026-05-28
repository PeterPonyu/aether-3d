"""Utility helpers for the Lumina / Aether flow-matching library.

This module contains small, pure helpers that are used across the flow
implementation. All functions are numerically stable and well-documented.

References (for the overall flow-matching approach):
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transport
  with Rectified Flow" (ICLR 2023)
"""

from __future__ import annotations

from typing import Any

import torch


def expand_time_like_data(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape scalar-per-sample time `t` to be broadcastable with data `x`.

    Args:
        t: 1-D tensor of shape (batch_size,).
        x: N-D tensor of shape (batch_size, ...).

    Returns:
        t reshaped to (batch_size, 1, 1, ..., 1) with the right number of
        singleton dimensions so that t * x works elementwise.
    """
    if t.dim() != 1:
        raise ValueError(f"Time tensor must be 1-D, got shape {t.shape}")
    if t.shape[0] != x.shape[0]:
        raise ValueError(
            f"Batch size mismatch: t has {t.shape[0]} samples, x has {x.shape[0]}"
        )

    dims = [1] * (x.dim() - 1)
    return t.view(t.size(0), *dims)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions (common in diffusion losses)."""
    return tensor.mean(dim=list(range(1, tensor.dim())))


@torch.no_grad()
def log_state(state_dict: dict[str, Any], prefix: str = "") -> None:
    """Tiny helper for debugging tensor statistics (used sparingly in training)."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: shape={tuple(v.shape)}, "
                  f"mean={v.mean().item():.4f}, std={v.std().item():.4f}")
        else:
            print(f"{prefix}{k}: {v}")
