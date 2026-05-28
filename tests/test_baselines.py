"""
Regression test for issue #147: ``LinearInterpAdapter`` paired cells across
visible slices by raw row index (``a.X[:n_pair]`` / ``b.X[:n_pair]``), so the
baseline number depended on the arbitrary order rows happened to be in the
input file. Every reported Aether-vs-baseline comparison was a function of
input ordering — silent publication risk.

The fix pairs cells by a deterministic property of the data (nearest neighbor
in spatial coords), making the baseline output shuffle-invariant.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from aether_3d.benchmarks.adapters.linear import LinearInterpAdapter
from aether_3d.benchmarks.contract import VolumeAdapterInput


def _build_slice(
    xy: np.ndarray, X: np.ndarray, z: float, z_key: str = "z"
) -> ad.AnnData:
    a = ad.AnnData(X=X.astype(np.float32))
    a.obsm["spatial"] = xy.astype(np.float32)
    a.obs[z_key] = np.full((xy.shape[0],), float(z), dtype=np.float32)
    return a


def _sorted_view(vol: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, spatial) sorted by (x, y) so row ordering doesn't matter."""
    xy = np.asarray(vol.obsm["spatial"], dtype=np.float32)
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    return np.asarray(vol.X, dtype=np.float32)[order], xy[order]


def test_linear_interp_shuffle_invariant() -> None:
    """Permuting row order of one visible slice must not change the output."""
    rng = np.random.default_rng(0)
    n = 8
    xy_a = rng.normal(size=(n, 2))
    xy_b = rng.normal(size=(n, 2))
    X_a = rng.normal(size=(n, 4))
    X_b = rng.normal(size=(n, 4))

    ad_a = _build_slice(xy_a, X_a, z=0.0)
    ad_b = _build_slice(xy_b, X_b, z=2.0)

    # Held-out truth slice at z=1.0; its content is irrelevant to the adapter
    # — the adapter only sees ``visible`` and the truth z-values.
    held = ad.AnnData(X=np.zeros((1, 4), dtype=np.float32))
    held.obsm["spatial"] = np.zeros((1, 2), dtype=np.float32)
    held.obs["z"] = np.array([1.0], dtype=np.float32)

    adapter = LinearInterpAdapter()

    inp_orig = VolumeAdapterInput(slices=[ad_a, ad_b, held], held_out_indices=[2])
    out_orig = adapter._reconstruct([ad_a, ad_b], inp_orig)

    # Shuffle slice B's row order and rerun.
    perm = rng.permutation(n)
    ad_b_shuf = _build_slice(xy_b[perm], X_b[perm], z=2.0)
    inp_shuf = VolumeAdapterInput(
        slices=[ad_a, ad_b_shuf, held], held_out_indices=[2]
    )
    out_shuf = adapter._reconstruct([ad_a, ad_b_shuf], inp_shuf)

    X1, xy1 = _sorted_view(out_orig)
    X2, xy2 = _sorted_view(out_shuf)

    assert X1.shape == X2.shape, (
        f"output cell count must be shuffle-invariant; got {X1.shape} vs {X2.shape}"
    )
    assert np.allclose(X1, X2, atol=1e-5), (
        "expression matrix must be shuffle-invariant; baseline currently pairs "
        "by raw row index so X_v depends on input ordering"
    )
    assert np.allclose(xy1, xy2, atol=1e-5), (
        "interpolated spatial coords must be shuffle-invariant"
    )
