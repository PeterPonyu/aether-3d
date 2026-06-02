#!/usr/bin/env python
"""End-to-end UOT-cost ablation: sweep (α_spatial, λ_class) and score the
resulting coupling against a known pairing.

Two data paths:
  * synthetic (default) — a controlled paired-slice generator with a *known*
    ground-truth permutation, so coupling quality is measured exactly.
  * real (``--real-data --data-dir <dir>``) — two cached MERFISH serial slices
    are loaded, their gene panels intersected, and an adjacent-slice 2D
    nearest-neighbour pairing is used as the (approximate) ground-truth partner
    map. Synthetic stays the default to keep CI hermetic.

The heatmap-shaped JSON powers the Aether3D 'Mechanistic interpretability'
figure and table.

Usage:
    python scripts/ci/run_uot_ablation.py
    python scripts/ci/run_uot_ablation.py --grid coarse
    python scripts/ci/run_uot_ablation.py --real-data \\
        --data-dir data/baselines/deepspatial/merfish_mouse_hypothalamus
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import reduce
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from aether_3d.benchmarks import (
    UOTAblationPoint,
    UOTAblationResult,
    aggregate_ablation,
    run_uot_ablation,
    score_coupling,
)


GRIDS = {
    "coarse": ([0.0, 0.5, 1.0], [0.0, 10.0, 100.0]),
    "fine": ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 1.0, 10.0, 100.0]),
}


def _slice_dict(adata: Any, n_classes_ref: Optional[Sequence[str]] = None) -> dict:
    """Project an AnnData slice into the {'x','g','c'} modality dict the cost
    function expects: 2D coords, gene matrix, one-hot cell-class labels."""
    import anndata as ad  # local import keeps the synthetic path import-light

    assert isinstance(adata, ad.AnnData)
    x = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    X = adata.X
    g = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    if "cell_class" in adata.obs:
        labels = adata.obs["cell_class"].astype(str).values
    elif "cell_type" in adata.obs:
        labels = adata.obs["cell_type"].astype(str).values
    else:
        labels = np.array(["_"] * adata.n_obs)
    classes = list(dict.fromkeys(labels)) if n_classes_ref is None else list(n_classes_ref)
    idx = {c: i for i, c in enumerate(classes)}
    c = np.zeros((adata.n_obs, max(len(classes), 1)), dtype=np.float32)
    for r, lab in enumerate(labels):
        c[r, idx.get(lab, 0)] = 1.0
    return {"x": x, "g": g, "c": c, "_labels": labels, "_classes": classes}


def load_real_paired_slices(
    data_dir: str,
    slice_a: int = 0,
    slice_b: int = 1,
    max_cells: Optional[int] = 400,
    seed: int = 0,
):
    """Load two cached MERFISH slices and build a (s0, s1, perm) triple.

    Reads ``<data_dir>/merfish_{slice_a}.h5ad`` and ``merfish_{slice_b}.h5ad``,
    intersects their gene panels to a shared sorted set, optionally seeded-caps
    each slice to ``max_cells`` cells (kept small so the dense cost matrix is
    tractable), and derives an *approximate* ground-truth partner map by 2D
    nearest-neighbour from slice A to slice B (real serial slices have no exact
    cell correspondence, so this is the honest best-effort pairing). Returns the
    same ``(s0, s1, perm)`` contract as ``make_paired_slices`` so the existing
    ablation scoring is reused unchanged.
    """
    import anndata as ad

    base = Path(data_dir)
    pa = base / f"merfish_{slice_a}.h5ad"
    pb = base / f"merfish_{slice_b}.h5ad"
    a = ad.read_h5ad(pa)
    b = ad.read_h5ad(pb)

    shared = reduce(np.intersect1d, [a.var_names.to_numpy(), b.var_names.to_numpy()])
    shared = np.sort(shared)
    a = a[:, list(shared)].copy()
    b = b[:, list(shared)].copy()

    rng = np.random.default_rng(seed)
    if max_cells is not None:
        if a.n_obs > max_cells:
            a = a[np.sort(rng.choice(a.n_obs, max_cells, replace=False))].copy()
        if b.n_obs > max_cells:
            b = b[np.sort(rng.choice(b.n_obs, max_cells, replace=False))].copy()

    # Shared cell-class vocabulary so the one-hot columns line up across slices.
    sa = _slice_dict(a)
    sb = _slice_dict(b, n_classes_ref=sa["_classes"])
    sa = {"x": sa["x"], "g": sa["g"], "c": sa["c"]}

    # Approximate partner map: nearest 2D neighbour in B for each cell in A.
    xa, xb = sa["x"], sb["x"]
    dists = ((xa[:, None, :] - xb[None, :, :]) ** 2).sum(axis=2)
    perm = np.argmin(dists, axis=1).astype(np.int64)

    sb = {"x": sb["x"], "g": sb["g"], "c": sb["c"]}
    return sa, sb, perm


def _run_real_ablation(
    points: Sequence[UOTAblationPoint],
    s0: dict,
    s1: dict,
    perm: np.ndarray,
    uot_reg: float = 0.8,
    uot_tau: float = 0.05,
    uot_samples: int = 5000,
) -> list[UOTAblationResult]:
    """Run the (α, λ) ablation on a single fixed real paired-slice triple."""
    import time

    from aether_3d.coupling.uot import compute_hybrid_cost, compute_uot_coupling

    out: list[UOTAblationResult] = []
    for point in points:
        t0 = time.perf_counter()
        cost = compute_hybrid_cost(
            s0["x"], s0["g"], s0["c"],
            s1["x"], s1["g"], s1["c"],
            alpha_spatial=point.alpha_spatial,
            lambda_class=point.lambda_class,
        )
        ablation_rng = np.random.default_rng(point.seed)
        src, tgt, weights = compute_uot_coupling(
            cost, reg=uot_reg, tau=uot_tau, n_samples=uot_samples, rng=ablation_rng
        )
        n0, n1 = s0["x"].shape[0], s1["x"].shape[0]
        P = np.zeros((n0, n1), dtype=np.float64)
        for s_, t_, w_ in zip(np.asarray(src), np.asarray(tgt), np.asarray(weights)):
            P[int(s_), int(t_)] += float(w_)
        runtime = time.perf_counter() - t0
        scores = score_coupling(P, perm)
        out.append(UOTAblationResult(
            point=point,
            top1_accuracy=scores["top1_accuracy"],
            mean_true_pair_mass=scores["mean_true_pair_mass"],
            runtime_s=runtime,
        ))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    # Real-data path (#227): synthetic stays the default.
    parser.add_argument(
        "--real-data", action="store_true",
        help="Run the ablation on real cached MERFISH slices instead of synthetic",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory holding merfish_{a,b}.h5ad (required with --real-data)",
    )
    parser.add_argument("--slice-a", type=int, default=0, help="First real slice index")
    parser.add_argument("--slice-b", type=int, default=1, help="Second real slice index")
    parser.add_argument(
        "--max-cells", type=int, default=400,
        help="Seeded per-slice cell cap for the real-data dense cost matrix",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark/uot_ablation.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.real_data and not args.data_dir:
        parser.error("--real-data requires --data-dir")

    alphas, lambdas = GRIDS[args.grid]
    points = [
        UOTAblationPoint(alpha_spatial=a, lambda_class=lam, seed=args.seed)
        for a in alphas for lam in lambdas
    ]

    if args.real_data:
        print(f"[uot-ablation] REAL data from {args.data_dir} "
              f"(slices {args.slice_a},{args.slice_b}); {args.grid} grid: "
              f"{len(alphas)}×{len(lambdas)}={len(points)} points; "
              f"max_cells={args.max_cells}")
        s0, s1, perm = load_real_paired_slices(
            args.data_dir,
            slice_a=args.slice_a,
            slice_b=args.slice_b,
            max_cells=args.max_cells if args.max_cells and args.max_cells > 0 else None,
            seed=args.seed,
        )
        results = _run_real_ablation(
            points, s0, s1, perm, uot_samples=args.uot_samples
        )
        data_source = "real"
    else:
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
        data_source = "synthetic"

    print(f"{'α_spatial':>10s} {'λ_class':>10s} {'top1_acc':>10s} "
          f"{'true_mass':>11s} {'runtime':>9s}")
    for r in results:
        print(f"{r.point.alpha_spatial:>10.3f} {r.point.lambda_class:>10.3f} "
              f"{r.top1_accuracy:>10.3f} {r.mean_true_pair_mass:>11.4f} "
              f"{r.runtime_s:>9.3f}")

    aggregated = aggregate_ablation(results)
    aggregated["data_source"] = data_source
    if args.real_data:
        aggregated["data_dir"] = str(args.data_dir)
        aggregated["slices"] = [args.slice_a, args.slice_b]
    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else (
        Path(__file__).resolve().parents[2] / out_arg
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[uot-ablation] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
