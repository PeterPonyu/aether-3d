"""Reproducibility regression tests (issue #118).

``run_uot_ablation`` previously seeded only the legacy global state via
``np.random.seed(point.seed)``, but ``compute_uot_coupling`` internally
constructs a fresh ``np.random.default_rng()`` which does *not* consult that
legacy state.  As a result the sampled coupling — and therefore both
``top1_accuracy`` and ``mean_true_pair_mass`` — were nondeterministic across
runs.  This file pins the explicit-RNG fix that threads a seeded
``np.random.Generator`` straight into the solver.
"""

from __future__ import annotations

from aether_3d.benchmarks.uot_ablation import (
    UOTAblationPoint,
    run_uot_ablation,
)


def test_uot_ablation_deterministic():
    """Two ablation runs with the same point seed must produce identical metrics."""
    points = [
        UOTAblationPoint(alpha_spatial=0.5, lambda_class=10.0, seed=7),
        UOTAblationPoint(alpha_spatial=0.2, lambda_class=5.0, seed=11),
    ]

    runs = [
        run_uot_ablation(
            points,
            n_cells=24,
            n_genes=8,
            n_classes=3,
            spatial_noise=1.0,
            gene_noise=0.1,
            uot_reg=0.8,
            uot_tau=0.05,
            uot_samples=2000,
        )
        for _ in range(2)
    ]

    assert len(runs[0]) == len(runs[1]) == len(points)
    for r0, r1 in zip(runs[0], runs[1]):
        assert r0.point == r1.point
        assert r0.top1_accuracy == r1.top1_accuracy, (
            f"top1_accuracy nondeterministic for {r0.point}: "
            f"{r0.top1_accuracy} != {r1.top1_accuracy}"
        )
        assert r0.mean_true_pair_mass == r1.mean_true_pair_mass, (
            f"mean_true_pair_mass nondeterministic for {r0.point}: "
            f"{r0.mean_true_pair_mass} != {r1.mean_true_pair_mass}"
        )
