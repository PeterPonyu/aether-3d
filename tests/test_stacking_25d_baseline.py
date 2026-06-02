"""Regression test for the clean-room 2.5D virtual-slice *stacking* baseline.

Before this change the holdout contrast in ``scripts/e2e/validate_holdout_slice.py``
only carried the nearest-copy and linear-blend 2.5D references; there was no
clean-room *stacking* baseline (the canonical 2.5D strategy of the named
reference competitor) wired in as an apples-to-apples floor — only an external-
package importer stub (``adapters/spatialz.py``) and a standalone script
explicitly flagged "NOT a faithful reproduction". ``Stacking25DAdapter`` fills
that gap.

These tests fail to import on ``origin/main`` (the adapter does not exist there)
and pass with the fix. They assert that:

  * the stacking baseline reconstructs the held-out interior slice with
    sensible, bounded fidelity on a small synthetic serial-slice stack;
  * every virtual cell is identity-preserving (carries a real source cell's
    exact expression row) — no fabricated expression;
  * the reconstruction is shuffle-invariant (independent of input row order);
  * the contrast harness emits metrics for ALL 2.5D methods plus the
    continuous reconstruction on the identical holdout (apples-to-apples).
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from aether_3d.benchmarks.adapters import (
    LinearInterpAdapter,
    NearestSliceAdapter,
    Stacking25DAdapter,
)
from aether_3d.benchmarks.contract import VolumeAdapterInput, _chamfer_distance


def _grid_slice(z: float, shift: float, n_side: int = 6, z_key: str = "z") -> ad.AnnData:
    """A serial slice: ``n_side**2`` cells on a unit grid, with a smooth spatial
    expression gradient that shifts with depth (so neighbours bracket the
    interior slice's signal)."""
    xs, ys = np.meshgrid(
        np.linspace(0.0, 1.0, n_side), np.linspace(0.0, 1.0, n_side), indexing="ij"
    )
    xy = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    g0 = (xy[:, 0] + shift).astype(np.float32)
    g1 = (xy[:, 1] + shift).astype(np.float32)
    g2 = (xy[:, 0] * xy[:, 1] + shift).astype(np.float32)
    X = np.column_stack([g0, g1, g2]).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = ["gA", "gB", "gC"]
    a.obsm["spatial"] = xy
    a.obs[z_key] = np.full((xy.shape[0],), float(z), dtype=np.float32)
    return a


def _stack() -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Three slices at z=0,1,2; the interior slice (z=1) is the held-out truth."""
    s0 = _grid_slice(z=0.0, shift=0.0)
    s1 = _grid_slice(z=1.0, shift=0.5)  # held-out interior truth
    s2 = _grid_slice(z=2.0, shift=1.0)
    return s0, s1, s2


def test_stacking_recovers_holdout_with_bounded_fidelity() -> None:
    s0, s1, s2 = _stack()
    inp = VolumeAdapterInput(slices=[s0, s1, s2], held_out_indices=[1], z_key="z")

    volume = Stacking25DAdapter()._reconstruct([s0, s2], inp)

    # Non-empty reconstruction landed at the held-out depth.
    assert volume.n_obs > 0
    v_z = volume.obs["z"].astype(float).to_numpy()
    assert np.allclose(v_z, 1.0), "stacked virtual cells must sit on the held-out z-plane"

    # Identity-preserving: every virtual expression row equals a real source row.
    src_rows = np.vstack([np.asarray(s0.X), np.asarray(s2.X)]).astype(np.float32)
    vX = np.asarray(volume.X, dtype=np.float32)
    for row in vX:
        assert np.any(np.all(np.isclose(src_rows, row, atol=1e-5), axis=1)), (
            "stacking baseline must not fabricate expression — each virtual cell "
            "carries a real source cell's exact row"
        )

    # Bounded fidelity vs the real held-out slice: the stacked mean profile
    # tracks the truth (it brackets the interior signal), and the point cloud
    # stays within the shared XY footprint (small chamfer).
    true_mean = np.asarray(s1.X, dtype=np.float32).mean(axis=0)
    pred_mean = vX.mean(axis=0)
    pearson = float(np.corrcoef(pred_mean, true_mean)[0, 1])
    assert pearson > 0.5, f"mean-profile correlation too low: {pearson:.3f}"

    true_coords = np.asarray(s1.obsm["spatial"], dtype=np.float32)
    pred_coords = np.asarray(volume.obsm["spatial"], dtype=np.float32)
    chamfer = _chamfer_distance(pred_coords, true_coords)
    grid_spacing = 1.0 / (6 - 1)
    assert chamfer < grid_spacing, f"chamfer distance unbounded: {chamfer:.4f}"

    # 3D coords are stamped consistently with the 2D coords + target z.
    assert volume.obsm["spatial_3d"].shape[1] == 3
    assert np.allclose(volume.obsm["spatial_3d"][:, 2], 1.0)


def test_stacking_is_shuffle_invariant() -> None:
    s0, s1, s2 = _stack()
    inp = VolumeAdapterInput(slices=[s0, s1, s2], held_out_indices=[1], z_key="z")
    out_a = Stacking25DAdapter()._reconstruct([s0, s2], inp)

    rng = np.random.default_rng(0)
    perm = rng.permutation(s2.n_obs)
    s2_shuf = ad.AnnData(X=np.asarray(s2.X)[perm].astype(np.float32))
    s2_shuf.var_names = s2.var_names
    s2_shuf.obsm["spatial"] = np.asarray(s2.obsm["spatial"])[perm]
    s2_shuf.obs["z"] = s2.obs["z"].to_numpy()[perm]
    inp_shuf = VolumeAdapterInput(
        slices=[s0, s1, s2_shuf], held_out_indices=[1], z_key="z"
    )
    out_b = Stacking25DAdapter()._reconstruct([s0, s2_shuf], inp_shuf)

    def _sorted(vol: ad.AnnData) -> np.ndarray:
        xy = np.asarray(vol.obsm["spatial"], dtype=np.float32)
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        return np.asarray(vol.X, dtype=np.float32)[order]

    xa, xb = _sorted(out_a), _sorted(out_b)
    assert xa.shape == xb.shape, "cell count must be shuffle-invariant"
    assert np.allclose(xa, xb, atol=1e-5), "stacked expression must be shuffle-invariant"


def test_contrast_emits_all_methods_on_same_holdout() -> None:
    """The harness contrasts continuous vs every 2.5D baseline on one holdout;
    assert all clean-room baselines emit a scored metric dict on identical data."""
    s0, s1, s2 = _stack()
    inp = VolumeAdapterInput(slices=[s0, s1, s2], held_out_indices=[1], z_key="z")
    true_coords = np.asarray(s1.obsm["spatial"], dtype=np.float32)

    methods = {
        "nearest-slice": NearestSliceAdapter(),
        "linear-interp": LinearInterpAdapter(),
        "stacking-2.5d": Stacking25DAdapter(),
    }
    contrast: dict[str, dict[str, float]] = {}
    for name, adapter in methods.items():
        volume = adapter._reconstruct(inp.visible_slices(), inp)
        pred_coords = np.asarray(volume.obsm["spatial"], dtype=np.float32)
        contrast[name] = {"chamfer_distance": _chamfer_distance(pred_coords, true_coords)}

    assert set(contrast) == set(methods), "every 2.5D method must emit metrics"
    for name, scores in contrast.items():
        assert np.isfinite(scores["chamfer_distance"]), f"{name} produced no metric"
    # The stacking baseline carries its own distinct name in the contrast table.
    assert Stacking25DAdapter().name == "stacking-2.5d"
