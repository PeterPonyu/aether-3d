"""UOT-cost component ablation for Aether3D.

The UOT cost combines three components:
    C = α_spatial · D^spat + (1 - α_spatial) · D^gene + λ_class · P^cell
This module runs a controlled experiment that scores the resulting coupling
against a *known* ground-truth pairing — built by permuting a synthetic
slice — so we can measure how each component contributes to coupling quality.

Outputs power the (α_spatial × λ_class) ablation heatmap that justifies the
hybrid-cost weighting choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ..coupling.uot import compute_hybrid_cost, compute_uot_coupling


@dataclass(frozen=True)
class UOTAblationPoint:
    """One cell of the (α_spatial, λ_class) ablation grid."""

    alpha_spatial: float
    lambda_class: float
    seed: int = 0


@dataclass
class UOTAblationResult:
    point: UOTAblationPoint
    top1_accuracy: float  # fraction of cells whose argmax-coupling lands on the true partner
    mean_true_pair_mass: float  # average P[i, true_j[i]] under row-normalized soft coupling
    runtime_s: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["point"] = asdict(self.point)
        return d


def make_paired_slices(
    n_cells: int = 60,
    n_genes: int = 20,
    n_classes: int = 4,
    spatial_noise: float = 1.0,
    gene_noise: float = 0.1,
    seed: int = 0,
) -> tuple[
    dict[str, npt.NDArray[np.float32]],
    dict[str, npt.NDArray[np.float32]],
    npt.NDArray[np.int64],
]:
    """Build two synthetic slices where slice 1 is a noisy copy of slice 0.

    Returns (slice0, slice1, true_permutation) where true_permutation[i] is the
    index of the cell in slice 1 that corresponds to cell i in slice 0.

    slice0/1 are dicts with keys: 'x' (2D coords), 'g' (gene matrix), 'c'
    (one-hot class labels).
    """
    rng = np.random.default_rng(seed)
    coords0 = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    g0 = rng.normal(0, 1, size=(n_cells, n_genes)).astype(np.float32)
    labels0 = rng.integers(0, n_classes, size=n_cells)
    c0 = np.zeros((n_cells, n_classes), dtype=np.float32)
    c0[np.arange(n_cells), labels0] = 1.0

    # Shuffle slice 1 to a known permutation, add per-modality noise.
    # Construction: slice1[k] is taken from slice0[shuffle[k]].
    shuffle = rng.permutation(n_cells)
    coords1 = coords0[shuffle] + rng.normal(0, spatial_noise, coords0.shape).astype(np.float32)
    g1 = g0[shuffle] + rng.normal(0, gene_noise, g0.shape).astype(np.float32)
    c1 = c0[shuffle].copy()

    # The "source → target" mapping callers want: source i in slice 0 corresponds to
    # slice 1 index inverse[i] (i.e. inverse[shuffle[k]] = k).
    inverse = np.empty_like(shuffle)
    inverse[shuffle] = np.arange(n_cells)

    s0 = {"x": coords0, "g": g0, "c": c0}
    s1 = {"x": coords1, "g": g1, "c": c1}
    return s0, s1, inverse


def score_coupling(
    P: npt.NDArray[Any],
    true_permutation: npt.NDArray[np.int64],
) -> dict[str, float]:
    """Score a soft coupling matrix P against the known ground-truth pairing.

    P[i, j] is the (unnormalized) mass between source i and target j. We
    row-normalize and then report:

    - top1_accuracy: fraction of i for which argmax_j P[i, j] == true_permutation[i]
    - mean_true_pair_mass: average row-normalized P[i, true_permutation[i]]
    """
    P = np.asarray(P, dtype=np.float64)
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    Pn = P / row_sum

    n0 = Pn.shape[0]
    pred = np.argmax(Pn, axis=1)
    top1 = float(np.mean(pred == true_permutation))
    true_mass = float(np.mean(Pn[np.arange(n0), true_permutation]))

    return {
        "top1_accuracy": top1,
        "mean_true_pair_mass": true_mass,
    }


def run_uot_ablation(
    points: Sequence[UOTAblationPoint],
    n_cells: int = 60,
    n_genes: int = 20,
    n_classes: int = 4,
    spatial_noise: float = 1.0,
    gene_noise: float = 0.1,
    uot_reg: float = 0.8,
    uot_tau: float = 0.05,
    uot_samples: int = 5000,
) -> list[UOTAblationResult]:
    """Run every (α_spatial, λ_class) cell with a fresh synthetic pairing."""
    import time

    out: list[UOTAblationResult] = []
    for point in points:
        s0, s1, perm = make_paired_slices(
            n_cells=n_cells,
            n_genes=n_genes,
            n_classes=n_classes,
            spatial_noise=spatial_noise,
            gene_noise=gene_noise,
            seed=point.seed,
        )

        t0 = time.perf_counter()
        # Compute hybrid cost at this point's (α, λ)
        cost = compute_hybrid_cost(
            s0["x"], s0["g"], s0["c"],
            s1["x"], s1["g"], s1["c"],
            alpha_spatial=point.alpha_spatial,
            lambda_class=point.lambda_class,
        )

        # Solve UOT on the resulting cost matrix
        np.random.seed(point.seed)
        src, tgt, weights = compute_uot_coupling(
            cost, reg=uot_reg, tau=uot_tau, n_samples=uot_samples,
        )

        # Reconstruct soft P from sampled pairs
        n0 = s0["x"].shape[0]
        n1 = s1["x"].shape[0]
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


def aggregate_ablation(results: Sequence[UOTAblationResult]) -> dict[str, Any]:
    """JSON-serializable aggregation for the heatmap figure."""
    return {
        "schema_version": "1",
        "n_points": len(results),
        "results": [r.to_dict() for r in results],
    }
