#!/usr/bin/env python
"""End-to-end smoke benchmark for Aether3D 3D reconstruction adapters.

Generates a synthetic 5-slice stack with random spatial coords + Poisson
expression, drops the middle slice as held-out truth, asks every adapter
to reconstruct, and writes the result JSON.

Usage:
    python scripts/ci/run_synthetic_holdout.py
    python scripts/ci/run_synthetic_holdout.py --out results/benchmark/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from aether_3d.benchmarks import (
    aggregate_volume_results,
    run_holdout,
    write_volume_results_json,
)
from aether_3d.benchmarks.adapters import (
    AetherAdapter,
    ASIGNAdapter,
    InterpolAIAdapter,
    LinearInterpAdapter,
    NearestSliceAdapter,
    SpatialZAdapter,
    ThreeDOTAdapter,
)


def make_synthetic_stack(
    z_values: list[float],
    n_cells: int = 80,
    n_genes: int = 30,
    seed: int = 0,
) -> list[ad.AnnData]:
    stack = []
    for i, z in enumerate(z_values):
        rng = np.random.default_rng(seed + i)
        X = rng.poisson(2.5, size=(n_cells, n_genes)).astype(np.float32)
        coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE_{j:03d}" for j in range(n_genes)]
        adata.obsm["spatial"] = coords
        adata.obs["z"] = float(z)
        adata.obs["cell_type"] = ["A"] * n_cells
        stack.append(adata)
    return stack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmark/synthetic_holdout.json",
        help="Output JSON path (relative to aether-3d/ root)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells", type=int, default=80)
    args = parser.parse_args()

    z_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    stack = make_synthetic_stack(z_values, n_cells=args.n_cells, seed=args.seed)
    held_out = [2]  # drop middle slice

    adapters = [
        AetherAdapter(),
        NearestSliceAdapter(),
        LinearInterpAdapter(),
        SpatialZAdapter(),
        ThreeDOTAdapter(),
        ASIGNAdapter(wsi_stack=None),
        InterpolAIAdapter(),
    ]

    print(f"[smoke] Running {len(adapters)} adapters on n_slices={len(stack)}, "
          f"held_out={held_out}, n_cells_per_slice={args.n_cells}")
    results = run_holdout(
        adapters, stack, held_out_indices=held_out, seed=args.seed,
        dataset_name="synthetic-stack-smoke",
    )

    for r in results:
        chamfer = r.metrics_json.get("mean_chamfer", float("nan"))
        rmse = r.metrics_json.get("mean_coord_rmse", float("nan"))
        n_v = r.metrics_json.get("n_virtual_cells", 0)
        status_short = r.status if len(r.status) < 50 else r.status[:50] + "..."
        print(f"  {r.method:16s} status={status_short:55s} "
              f"chamfer={chamfer:.4f} rmse={rmse:.4f} n_virtual={n_v} "
              f"runtime={r.runtime_s:.3f}s")

    aggregated = aggregate_volume_results(
        {("synthetic-stack-smoke", f"holdout-{held_out[0]}"): results}
    )

    out_path = Path(__file__).resolve().parents[2] / args.out
    write_volume_results_json(aggregated, out_path)
    print(f"[smoke] Wrote {out_path}")

    ok = [r for r in results if r.status == "ok"]
    if not ok:
        print("[smoke] FAIL: no adapter succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
