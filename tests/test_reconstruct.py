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
import pytest
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


# ---------------------------------------------------------------------------
# Issue #81 — the unconditional 2nd–98th z-percentile "density pruning" at the
# end of reconstruct_continuous_volume silently dropped sparse endpoint z-planes
# (virtual planes are deterministic, so there are never real z outliers). The
# fix makes pruning opt-in (cfg.prune_z_outliers, default False) and warns with
# the dropped count + z-planes when enabled.
# ---------------------------------------------------------------------------


def _uneven_adatas(n0: int, n1: int, n_genes: int = 8, seed: int = 3) -> list[ad.AnnData]:
    """Two slices with very different cell counts so the lower-density
    endpoint plane falls below the 2nd z-percentile under the old clip."""
    rng = np.random.default_rng(seed)
    out = []
    for z, n in ((0.0, n0), (1.0, n1)):
        a = ad.AnnData(
            X=rng.normal(size=(n, n_genes)).astype(np.float32),
            obs={
                "cell_class": (["T", "B"] * ((n + 1) // 2))[:n],
                "z_coord": [float(z)] * n,
            },
        )
        a.obsm["spatial"] = rng.normal(size=(n, 2)).astype(np.float32)
        out.append(a)
    return out


def _prune_cfg(**overrides: Any) -> Aether3DConfig:
    base: dict[str, Any] = dict(
        seed=42,
        hidden_size=8,
        depth=1,
        num_heads=2,
        patch_size=4,
        n_samples_base=200,
        n_samples_volume=200,
        thickness=10.0,
    )
    base.update(overrides)
    return Aether3DConfig(**base)


def test_boundary_slices_not_silently_dropped() -> None:
    """By default (prune_z_outliers=False), the sparse endpoint z-planes must
    survive — on unfixed main the unconditional 2/98 clip deletes the z=0
    plane (its cell share is < 2%)."""
    adatas = _uneven_adatas(n0=2, n1=120)
    recon = AetherReconstructor(_prune_cfg())
    recon.setup_data(adatas)

    volume = recon.reconstruct_continuous_volume(
        adatas, thickness=10.0, n_samples=200, num_depths=3
    )

    zs = set(np.round(volume.obs["z_3d"].astype(float), 3))
    assert 0.0 in zs, f"endpoint z=0 plane was silently dropped; z-planes={sorted(zs)}"
    assert max(zs) == pytest.approx(10.0), (
        f"top endpoint z=10 plane was silently dropped; z-planes={sorted(zs)}"
    )


def test_prune_z_outliers_opt_in_warns_about_dropped_cells() -> None:
    """When explicitly enabled, pruning must still work but emit a RuntimeWarning
    naming how many cells (and which z-planes) were removed — never silent."""
    adatas = _uneven_adatas(n0=2, n1=120)
    recon = AetherReconstructor(_prune_cfg(prune_z_outliers=True))
    recon.setup_data(adatas)

    with pytest.warns(RuntimeWarning, match="prune_z_outliers dropped"):
        volume = recon.reconstruct_continuous_volume(
            adatas, thickness=10.0, n_samples=200, num_depths=3
        )
    assert volume.n_obs > 0
