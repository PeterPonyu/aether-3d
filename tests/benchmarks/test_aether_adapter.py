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
