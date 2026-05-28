"""
Regression test for issue #82: ``reconstruct_continuous_volume`` must pipe
``z`` and ``delta_z`` conditioning through the velocity-field model so that
inputs at inference match what ``SerialSliceTrajectoryDataset`` and
``AetherFlowModule.training_step`` produce during training.

On unfixed ``main``, the drift function builds ``state = {"x", "g", "c"}``
only; the model silently defaults ``z`` and ``delta_z`` to zeros. That breaks
the train/inference invariant for any pair of slices whose ``z_coord`` differs
(which is every real pair).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import torch

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor


class _CaptureModel(torch.nn.Module):
    """A stand-in velocity field that records every ``state`` it sees."""

    def __init__(self, spatial_dim: int, gene_dim: int, num_classes: int) -> None:
        super().__init__()
        self._spatial_dim = spatial_dim
        self._gene_dim = gene_dim
        self._num_classes = num_classes
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        state: dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Snapshot just the keys + their first-row contents to keep memory bounded.
        snap = {k: v.detach().clone() for k, v in state.items()}
        snap["__t"] = t.detach().clone()
        self.calls.append(snap)
        b = state["x"].shape[0]
        return {
            "vx": torch.zeros(b, self._spatial_dim),
            "vg": torch.zeros(b, self._gene_dim),
            "vc": torch.zeros(b, self._num_classes),
        }


def _two_slice_adatas(z0: float = 0.0, z1: float = 2.0) -> list[ad.AnnData]:
    rng = np.random.default_rng(0)
    out = []
    for z in (z0, z1):
        a = ad.AnnData(
            X=rng.normal(size=(6, 8)).astype(np.float32),
            obs={
                "cell_class": ["T", "B", "T", "B", "T", "B"],
                "z_coord": [float(z)] * 6,
            },
        )
        a.obsm["spatial"] = rng.normal(size=(6, 2)).astype(np.float32)
        out.append(a)
    return out


def test_z_conditioning_matches_training(tmp_path) -> None:
    """``reconstruct_continuous_volume`` must pass z/delta_z in the state dict."""
    adatas = _two_slice_adatas(z0=0.0, z1=2.0)
    cfg = Aether3DConfig(
        n_samples_base=6,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_volume=4,
        output_dir=tmp_path,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adatas)

    capture = _CaptureModel(
        spatial_dim=recon.spatial_dim,
        gene_dim=recon.gene_dim,
        num_classes=recon.num_classes,
    )
    recon.model = capture
    recon.ema_model = capture

    _ = recon.reconstruct_continuous_volume(adatas, thickness=2.0, num_depths=3)

    assert capture.calls, "model was never called during reconstruction"

    z_seen = [call.get("z") for call in capture.calls]
    dz_seen = [call.get("delta_z") for call in capture.calls]

    # The fix must populate both keys in the inference state dict, matching
    # AetherFlowModule.training_step's state construction.
    assert all(z is not None for z in z_seen), (
        "reconstruct_continuous_volume must include 'z' in state dict "
        "(matches SerialSliceTrajectoryDataset training inputs)"
    )
    assert all(dz is not None for dz in dz_seen), (
        "reconstruct_continuous_volume must include 'delta_z' in state dict "
        "(matches SerialSliceTrajectoryDataset training inputs)"
    )

    # delta_z for an adjacent pair (z=0 → z=2) must be 2 — the same value the
    # training dataset would emit for that pair.
    for dz in dz_seen:
        assert torch.allclose(
            dz.view(-1), torch.full_like(dz.view(-1), 2.0)
        ), f"expected delta_z == 2.0 (z1 - z0), got {dz.view(-1)}"

    # z must reflect the interpolated training path: z = z_start + t * delta_z.
    # On the unfixed code z was an all-zeros fallback regardless of slice
    # z_coord; require at least one call where z is non-trivial.
    assert any(z.abs().sum().item() > 0 for z in z_seen), (
        "expected non-zero z conditioning (matches training-time interpolation); "
        "got all-zeros fallback"
    )
