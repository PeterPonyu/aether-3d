"""AetherReconstructor scored through the volume-adapter contract (issue #87).

Verifies that the package's own method runs end-to-end under the same audited
holdout protocol as the baselines, and that the reconstructor's synthetic
``z_3d`` output is remapped onto the physical ``inp.z_key`` so the held-out
slice's per-depth metrics are actually scored.

A single run_holdout call covers both assertions because each call trains +
reconstructs (an adaptive-ODE pass), which is the dominant cost.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from aether_3d.benchmarks import run_holdout
from aether_3d.benchmarks.adapters import AetherAdapter


def _make_benchmark_slices(
    n: int = 3, n_cells: int = 6, n_genes: int = 4, seed: int = 0
) -> list[ad.AnnData]:
    rng = np.random.default_rng(seed)
    slices: list[ad.AnnData] = []
    for z in range(n):
        a = ad.AnnData(X=rng.normal(size=(n_cells, n_genes)).astype(np.float32))
        a.obs["cell_type"] = ["A", "B"] * (n_cells // 2)
        a.obs["z"] = float(z)  # contract default z_key
        a.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
        slices.append(a)
    return slices


def test_aether_adapter_scores_through_contract() -> None:
    # Visible z = {0, 2}, held-out z = 1. num_depths=3 yields an interior
    # virtual plane (depth fraction 0.5 -> physical z = 1) after the
    # z_3d -> inp.z_key remap, so the held-out metric window finds cells.
    slices = _make_benchmark_slices(n=3)
    results = run_holdout(
        [AetherAdapter(num_depths=3)],  # default max_epochs=0: reconstruct only
        slices,
        held_out_indices=[1],
        z_key="z",
    )
    res = results[0]
    assert res.status == "ok", res.status
    assert res.metrics_json["n_virtual_cells"] > 0

    # The z-key remap must place virtual cells at the held-out physical depth so
    # the per-slice metric window finds them (otherwise per_holdout_slice would
    # report no_virtual_cells_at_z).
    per_slice = res.metrics_json.get("per_holdout_slice", [])
    assert per_slice, "expected a per-holdout-slice metric entry"
    assert per_slice[0]["n_virtual"] > 0, (
        "z-key remap failed: no virtual cells scored at the held-out physical depth"
    )


def _make_recon(visible: list[ad.AnnData]):
    """Reconstructor with dims inferred from ``visible`` and the AetherAdapter
    default architecture, so a checkpoint built here loads cleanly into it."""
    from aether_3d.config.aether_config import Aether3DConfig
    from aether_3d.core.aether_reconstructor import AetherReconstructor

    cfg = Aether3DConfig(
        spatial_key="spatial",
        z_key="z",
        label_key="cell_type",
        hidden_size=32,
        depth=2,
        num_heads=2,
        patch_size=4,
        num_workers=0,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(visible)
    assert recon.model is not None
    return recon


def _field_state_dict(visible: list[ad.AnnData]) -> dict:
    """A velocity-field ``state_dict`` matching the adapter's default arch."""
    return _make_recon(visible).model.state_dict()


def test_aether_adapter_loads_raw_checkpoint(tmp_path) -> None:
    """checkpoint_path with a raw field state_dict scores through the contract
    without inline training (#285/#288)."""
    import torch

    slices = _make_benchmark_slices(n=3)
    visible = [slices[0], slices[2]]  # held_out_indices=[1]
    ckpt = tmp_path / "field.pt"
    torch.save(_field_state_dict(visible), ckpt)

    results = run_holdout(
        [AetherAdapter(num_depths=3, checkpoint_path=ckpt)],  # max_epochs=0
        slices,
        held_out_indices=[1],
        z_key="z",
    )
    res = results[0]
    assert res.status == "ok", res.status
    assert res.metrics_json["n_virtual_cells"] > 0


def test_aether_adapter_loads_lightning_checkpoint(tmp_path) -> None:
    """A Lightning-style checkpoint (nested state_dict with model./ema_model.
    prefixes) loads; EMA weights are preferred."""
    import torch

    slices = _make_benchmark_slices(n=3)
    visible = [slices[0], slices[2]]
    field_sd = _field_state_dict(visible)
    lightning = {
        "state_dict": {
            **{f"model.{k}": v for k, v in field_sd.items()},
            **{f"ema_model.{k}": v for k, v in field_sd.items()},
        }
    }
    ckpt = tmp_path / "lightning.ckpt"
    torch.save(lightning, ckpt)

    results = run_holdout(
        [AetherAdapter(num_depths=3, checkpoint_path=ckpt)],
        slices,
        held_out_indices=[1],
        z_key="z",
    )
    assert results[0].status == "ok", results[0].status


def test_aether_adapter_rejects_zero_match_checkpoint(tmp_path) -> None:
    """A checkpoint matching zero parameters must fail loudly, not silently
    score a randomly initialised field (silent-failure guard)."""
    import torch

    slices = _make_benchmark_slices(n=3)
    ckpt = tmp_path / "bogus.pt"
    torch.save({"totally.bogus.key": torch.zeros(2)}, ckpt)

    results = run_holdout(
        [AetherAdapter(num_depths=3, checkpoint_path=ckpt)],
        slices,
        held_out_indices=[1],
        z_key="z",
    )
    res = results[0]
    assert res.status.startswith("error"), res.status
    assert "incomplete match" in res.status, res.status


def test_load_weights_prefers_ema_over_model(tmp_path) -> None:
    """When a Lightning checkpoint carries DIFFERING model./ema_model. weights,
    the EMA (inference) weights win — the reconstructor serves EMA post-train."""
    import torch

    slices = _make_benchmark_slices(n=3)
    visible = [slices[0], slices[2]]
    base = _field_state_dict(visible)
    model_sd = {k: torch.zeros_like(v) for k, v in base.items()}
    ema_sd = {k: torch.ones_like(v) for k, v in base.items()}
    ckpt = tmp_path / "ema_vs_model.ckpt"
    torch.save(
        {
            "state_dict": {
                **{f"model.{k}": v for k, v in model_sd.items()},
                **{f"ema_model.{k}": v for k, v in ema_sd.items()},
            }
        },
        ckpt,
    )

    recon = _make_recon(visible)
    AetherAdapter(checkpoint_path=ckpt)._load_weights(recon)

    # Every float parameter must equal the EMA (ones) values, not model (zeros).
    loaded = recon.model.state_dict()
    checked = 0
    for k, v in loaded.items():
        if v.dtype.is_floating_point and v.numel() > 0:
            assert torch.allclose(v, torch.ones_like(v)), f"{k} did not take EMA weights"
            checked += 1
    assert checked > 0, "no float params were verified"


def test_load_weights_rejects_partial_checkpoint(tmp_path) -> None:
    """A checkpoint covering only a SUBSET of the field's parameters must raise —
    a partial load would leave the rest randomly initialised (claim-grade guard)."""
    import pytest
    import torch

    slices = _make_benchmark_slices(n=3)
    visible = [slices[0], slices[2]]
    base = _field_state_dict(visible)
    keys = list(base.keys())
    partial = {k: base[k] for k in keys[: max(1, len(keys) // 3)]}  # drop ~2/3
    ckpt = tmp_path / "partial.pt"
    torch.save(partial, ckpt)

    recon = _make_recon(visible)
    with pytest.raises(RuntimeError, match="incomplete match"):
        AetherAdapter(checkpoint_path=ckpt)._load_weights(recon)
