#!/usr/bin/env python
"""End-to-end UOT-cost ablation: sweep (α_spatial, λ_class) and score the
resulting coupling against a known synthetic pairing.

The heatmap-shaped JSON powers the Aether3D 'Mechanistic interpretability'
figure and table.

Usage:
    python scripts/ci/run_uot_ablation.py
    python scripts/ci/run_uot_ablation.py --grid coarse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aether_3d.benchmarks import (
    UOTAblationPoint,
    aggregate_ablation,
    run_uot_ablation,
)


GRIDS = {
    "coarse": ([0.0, 0.5, 1.0], [0.0, 10.0, 100.0]),
    "fine": ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 1.0, 10.0, 100.0]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", choices=list(GRIDS), default="coarse",
        help="Grid resolution (default: coarse for CI)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells", type=int, default=60)
    parser.add_argument("--n-genes", type=int, default=20)
    parser.add_argument("--spatial-noise", type=float, default=2.0)
    parser.add_argument("--gene-noise", type=float, default=0.3)
    parser.add_argument("--uot-samples", type=int, default=5000)
    parser.add_argument(
        "--out",
        default="results/benchmark/uot_ablation.json",
    )
    args = parser.parse_args()

    alphas, lambdas = GRIDS[args.grid]
    points = [
        UOTAblationPoint(alpha_spatial=a, lambda_class=l, seed=args.seed)
        for a in alphas for l in lambdas
    ]

    print(f"[uot-ablation] {args.grid} grid: {len(alphas)} alphas × "
          f"{len(lambdas)} lambdas = {len(points)} points; "
          f"n_cells={args.n_cells}, spatial_noise={args.spatial_noise}, "
          f"gene_noise={args.gene_noise}")
    results = run_uot_ablation(
        points,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        spatial_noise=args.spatial_noise,
        gene_noise=args.gene_noise,
        uot_samples=args.uot_samples,
    )

    print(f"{'α_spatial':>10s} {'λ_class':>10s} {'top1_acc':>10s} "
          f"{'true_mass':>11s} {'runtime':>9s}")
    for r in results:
        print(f"{r.point.alpha_spatial:>10.3f} {r.point.lambda_class:>10.3f} "
              f"{r.top1_accuracy:>10.3f} {r.mean_true_pair_mass:>11.4f} "
              f"{r.runtime_s:>9.3f}")

    aggregated = aggregate_ablation(results)
    out_path = Path(__file__).resolve().parents[2] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[uot-ablation] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
