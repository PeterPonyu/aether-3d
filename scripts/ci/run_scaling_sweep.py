#!/usr/bin/env python
"""Reproducible scaling-curve harness.

`--quick` runs a tiny synthetic sweep on CPU in ~10s — useful for CI to
verify the harness still works and the JSON schema is stable.

`--full` runs the documented research sweep (cells × slices grid). Defaults
are conservative; pass --max-cells to push the curve further on a real GPU.

The output JSON is hardware-honest: device, CUDA/torch versions, peak memory,
runtime, hostname, and git SHA are all recorded per measurement so reviewers
can reproduce the curve exactly.

Usage:
    python scripts/ci/run_scaling_sweep.py --quick
    python scripts/ci/run_scaling_sweep.py --full --max-cells 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aether_3d.benchmarks import (
    ScalingPoint,
    aggregate_scaling,
    sweep,
)
from aether_3d.benchmarks.adapters import (
    LinearInterpAdapter,
    NearestSliceAdapter,
)

QUICK_POINTS = [
    ScalingPoint(n_cells_per_slice=50, n_slices=3, n_genes=20),
    ScalingPoint(n_cells_per_slice=200, n_slices=3, n_genes=20),
]

FULL_POINTS = [
    ScalingPoint(n_cells_per_slice=int(1e3), n_slices=4, n_genes=50),
    ScalingPoint(n_cells_per_slice=int(1e4), n_slices=4, n_genes=50),
    ScalingPoint(n_cells_per_slice=int(5e4), n_slices=4, n_genes=50),
    ScalingPoint(n_cells_per_slice=int(1e5), n_slices=4, n_genes=50),
    # Larger points must be enabled explicitly via --max-cells; we do not
    # ship a default that requires a 24 GB GPU.
]


def filter_points(points, max_cells: int | None):
    if max_cells is None:
        return points
    return [p for p in points if p.total_cells <= max_cells]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny CPU sweep for CI")
    parser.add_argument("--full", action="store_true", help="documented research sweep")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Cap on total cells per sweep point (safety guard for limited hardware)",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark/scaling_curve.json",
        help="Output JSON path (relative to aether-3d/ root)",
    )
    args = parser.parse_args()

    if not (args.quick or args.full):
        parser.error("Pass --quick or --full")

    points = QUICK_POINTS if args.quick else FULL_POINTS
    points = filter_points(points, args.max_cells)

    adapters = [NearestSliceAdapter(), LinearInterpAdapter()]

    print(f"[scaling] {'quick' if args.quick else 'full'} sweep: "
          f"{len(adapters)} adapters × {len(points)} points")
    results = sweep(adapters, points)

    print(f"[scaling] device={results[0].device}, "
          f"torch={results[0].torch_version}, cuda={results[0].cuda_version}")
    print(f"{'adapter':16s} {'cells':>8s} {'slices':>7s} {'runtime_s':>10s} {'peak_MB':>9s} {'n_virtual':>10s} status")
    for r in results:
        print(f"{r.adapter:16s} {r.point.n_cells_per_slice:>8d} {r.point.n_slices:>7d} "
              f"{r.runtime_s:>10.3f} {r.peak_memory_mb:>9.1f} {r.n_virtual_cells:>10d} {r.status}")

    aggregated = aggregate_scaling(results)
    out_path = Path(__file__).resolve().parents[2] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[scaling] Wrote {out_path}")

    failed = [r for r in results if not r.status == "ok"]
    if failed:
        print(f"[scaling] {len(failed)} measurement(s) did not succeed:", file=sys.stderr)
        for r in failed:
            print(f"  {r.adapter}@{r.point.total_cells}cells: {r.status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
