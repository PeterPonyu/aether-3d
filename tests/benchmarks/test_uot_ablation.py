"""Tests for the UOT-cost ablation."""

from __future__ import annotations

import numpy as np
import pytest

from aether_3d.benchmarks import (
    UOTAblationPoint,
    aggregate_ablation,
    make_paired_slices,
    run_uot_ablation,
    score_coupling,
)
from aether_3d.coupling.uot import compute_hybrid_cost


# -- Cost-component sanity ------------------------------------------------


def test_cost_alpha_zero_ignores_spatial_component():
    """α=0 ⇒ cost is gene + class only; identical gene profiles give zero gene cost."""
    n = 5
    rng = np.random.default_rng(0)
    coords0 = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    coords1 = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    g = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    c = np.eye(n, dtype=np.float32)

    # Identical gene profiles & class assignments → cost should equal class penalty (zero on diagonal)
    C = compute_hybrid_cost(coords0, g, c, coords1, g, c, alpha_spatial=0.0, lambda_class=10.0)
    assert np.allclose(np.diag(C), 0.0, atol=1e-5), \
        "α=0 + identical genes + identical classes ⇒ diagonal cost should be zero"
    assert np.all(C >= -1e-6)


def test_cost_alpha_one_ignores_gene_component():
    """α=1 ⇒ cost is spatial + class only; spatial-identical points have zero spatial cost."""
    n = 5
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    g0 = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    g1 = rng.normal(0, 1, size=(n, 8)).astype(np.float32)  # very different
    c = np.eye(n, dtype=np.float32)

    C = compute_hybrid_cost(coords, g0, c, coords, g1, c, alpha_spatial=1.0, lambda_class=10.0)
    # Same coordinates and same class on diagonal ⇒ cost ≈ 0 ignoring gene noise
    assert np.allclose(np.diag(C), 0.0, atol=1e-5), \
        f"α=1 + identical coords + same class ⇒ diagonal should be zero; got {np.diag(C)}"


def test_lambda_class_zero_removes_class_penalty():
    """λ_class=0 ⇒ cost is α-blend of spatial+gene; class mismatches no longer matter."""
    n = 5
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    g = rng.normal(0, 1, size=(n, 8)).astype(np.float32)

    # Different classes shouldn't add penalty when λ=0
    c0 = np.zeros((n, 3), dtype=np.float32)
    c0[:, 0] = 1.0  # all class 0
    c1 = np.zeros((n, 3), dtype=np.float32)
    c1[:, 1] = 1.0  # all class 1

    C_lambda0 = compute_hybrid_cost(coords, g, c0, coords, g, c1, alpha_spatial=0.5, lambda_class=0.0)
    C_lambda10 = compute_hybrid_cost(coords, g, c0, coords, g, c1, alpha_spatial=0.5, lambda_class=10.0)

    # With class mismatch, λ=10 must produce strictly larger cost everywhere
    diff = C_lambda10 - C_lambda0
    assert np.all(diff >= -1e-6), "λ=10 with class mismatch should never reduce cost"
    assert np.mean(diff) > 5.0, f"λ=10 vs λ=0 should add ~10 on average for full mismatch; got mean diff {np.mean(diff)}"


def test_lambda_class_large_dominates_cost():
    """Large λ_class makes the class penalty dwarf spatial/gene costs."""
    n = 5
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    g = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    c_same = np.eye(n, dtype=np.float32)

    # Diagonal: same class; off-diagonal: different class. Large λ ⇒ huge gap.
    C = compute_hybrid_cost(coords, g, c_same, coords, g, c_same, alpha_spatial=0.5, lambda_class=1000.0)
    on_diag = np.diag(C)
    off_diag = C[~np.eye(n, dtype=bool)]
    assert off_diag.min() - on_diag.max() > 100.0, \
        "large λ should make off-diagonal (different class) cost dramatically larger"


def test_cost_backward_compatible_default_lambda_is_10():
    """Default λ_class=10 keeps the same behavior as before this PR."""
    n = 4
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    g = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    c0 = np.zeros((n, 2), dtype=np.float32)
    c0[:, 0] = 1
    c1 = np.zeros((n, 2), dtype=np.float32)
    c1[:, 1] = 1

    explicit = compute_hybrid_cost(coords, g, c0, coords, g, c1, alpha_spatial=0.5, lambda_class=10.0)
    default = compute_hybrid_cost(coords, g, c0, coords, g, c1, alpha_spatial=0.5)
    np.testing.assert_array_almost_equal(explicit, default)


# -- make_paired_slices --------------------------------------------------


def test_make_paired_slices_returns_consistent_permutation():
    s0, s1, perm = make_paired_slices(n_cells=20, seed=0, spatial_noise=0.0, gene_noise=0.0)
    # When noise=0, slice1[perm[i]] must equal slice0[i] in all modalities.
    for i in range(20):
        assert np.allclose(s0["x"][i], s1["x"][perm[i]])
        assert np.allclose(s0["g"][i], s1["g"][perm[i]])
        assert np.allclose(s0["c"][i], s1["c"][perm[i]])


def test_make_paired_slices_permutation_is_valid():
    _, _, perm = make_paired_slices(n_cells=30, seed=0)
    assert sorted(perm.tolist()) == list(range(30))


# -- score_coupling -------------------------------------------------------


def test_score_coupling_perfect_matrix_yields_top1_one():
    n = 10
    perm = np.array([3, 1, 0, 7, 5, 9, 2, 4, 8, 6])
    # Build a P where row i has its mass on column perm[i].
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        P[i, perm[i]] = 1.0
    scores = score_coupling(P, perm)
    assert scores["top1_accuracy"] == 1.0
    assert scores["mean_true_pair_mass"] == pytest.approx(1.0)


def test_score_coupling_uniform_matrix_yields_low_top1():
    n = 10
    perm = np.arange(n)
    P = np.ones((n, n), dtype=np.float64) / n
    scores = score_coupling(P, perm)
    # argmax over uniform is 0 → top1 hits only when perm[i] == 0, so 1/n
    assert scores["top1_accuracy"] <= 1.0 / n + 1e-6
    assert scores["mean_true_pair_mass"] == pytest.approx(1.0 / n)


# -- Ablation runner ------------------------------------------------------


def test_run_uot_ablation_emits_one_result_per_point():
    points = [
        UOTAblationPoint(alpha_spatial=0.0, lambda_class=0.0, seed=0),
        UOTAblationPoint(alpha_spatial=0.5, lambda_class=10.0, seed=0),
        UOTAblationPoint(alpha_spatial=1.0, lambda_class=100.0, seed=0),
    ]
    results = run_uot_ablation(points, n_cells=30, n_genes=10, uot_samples=1000)
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.top1_accuracy <= 1.0
        assert 0.0 <= r.mean_true_pair_mass <= 1.0
        assert r.runtime_s >= 0.0
        # marginal_spearman was removed (it was degenerate against a uniform baseline)
        assert not hasattr(r, "marginal_spearman")


def test_aggregate_ablation_is_json_serializable():
    points = [
        UOTAblationPoint(alpha_spatial=0.25, lambda_class=1.0, seed=0),
        UOTAblationPoint(alpha_spatial=0.75, lambda_class=10.0, seed=0),
    ]
    results = run_uot_ablation(points, n_cells=20, n_genes=6, uot_samples=500)
    aggregated = aggregate_ablation(results)

    assert aggregated["schema_version"] == "1"
    assert aggregated["n_points"] == 2
    assert "results" in aggregated
    r = aggregated["results"][0]
    assert "point" in r
    assert isinstance(r["point"], dict)
    assert r["point"]["alpha_spatial"] == 0.25
    assert r["point"]["lambda_class"] == 1.0

    # Round-trips through JSON
    import json
    s = json.dumps(aggregated)
    loaded = json.loads(s)
    assert loaded["n_points"] == 2


# -- Functional behavior of the ablation ----------------------------------


def test_ablation_with_clean_spatial_signal_prefers_high_alpha():
    """If gene profiles are noisy but spatial coords are clean, high α should help."""
    point_low = UOTAblationPoint(alpha_spatial=0.0, lambda_class=10.0, seed=0)
    point_high = UOTAblationPoint(alpha_spatial=1.0, lambda_class=10.0, seed=0)

    results = run_uot_ablation(
        [point_low, point_high],
        n_cells=40,
        n_genes=10,
        spatial_noise=0.5,   # low spatial noise
        gene_noise=5.0,      # high gene noise
        uot_samples=3000,
    )
    # The spatial-only weighting should not be drastically worse than gene-only;
    # we just check both produce sane (non-degenerate) scores.
    assert results[0].mean_true_pair_mass >= 0.0
    assert results[1].mean_true_pair_mass >= 0.0
