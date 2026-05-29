"""Regression tests for gene-expression fidelity metrics.

Issue #130 — the benchmark's headline "Gene Pearson" was computed as the
Pearson correlation between the per-gene *mean* expression of the whole
reconstructed slice and the per-gene mean of the whole truth slice. That
collapses each slice to a single bulk profile, so it is invariant to where
cells are placed: a spatially-shuffled (or near-constant) reconstruction can
score ~1.0 even though per-cell fidelity is near zero.

`gene_pearson_fidelity` returns that bulk number *clearly labeled* alongside
spatially-matched per-cell and per-gene Pearson / RMSE. These tests pin the
distinction: a shuffle that preserves bulk means must score high on the bulk
metric but low on the cell-level metric.
"""

from __future__ import annotations

import numpy as np

from aether_3d.benchmarks.metrics import gene_pearson_fidelity


def test_gene_pearson_is_cell_level():
    """A spatially-shuffled reconstruction with identical bulk means must score
    high on the bulk slice-mean Pearson but low on the per-cell / per-gene
    Pearson — documenting that the bulk metric is insensitive to layout."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 16
    truth_X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)

    # Permute the per-cell expression rows but keep every cell at its original
    # position. Bulk per-gene means are permutation-invariant (=> identical),
    # while each cell now carries a different cell's expression.
    perm = rng.permutation(n_cells)
    recon_X = truth_X[perm]

    res = gene_pearson_fidelity(
        X_recon=recon_X,
        coords_recon=coords,
        X_truth=truth_X,
        coords_truth=coords,
    )

    # Bulk metric is fooled: ~1.0 despite a scrambled reconstruction.
    assert res["bulk_slice_mean_pearson"] > 0.99, res

    # Cell-level metrics correctly report poor reconstruction (~0).
    assert res["per_cell_gene_pearson"] < 0.5, res
    assert res["per_gene_pearson"] < 0.5, res
    assert (
        res["per_cell_gene_pearson"] < res["bulk_slice_mean_pearson"]
    ), "per-cell Pearson must not exceed the bulk metric for a shuffle"
    assert res["per_cell_gene_rmse"] > 0.0


def test_gene_pearson_fidelity_perfect_reconstruction_scores_high():
    """An exact reconstruction must score high on bulk AND cell-level metrics
    and have ~zero RMSE."""
    rng = np.random.default_rng(1)
    truth_X = rng.normal(size=(40, 12)).astype(np.float32)
    coords = rng.uniform(0, 50, size=(40, 2)).astype(np.float32)

    res = gene_pearson_fidelity(
        X_recon=truth_X.copy(),
        coords_recon=coords.copy(),
        X_truth=truth_X,
        coords_truth=coords,
    )

    assert res["bulk_slice_mean_pearson"] > 0.99, res
    assert res["per_cell_gene_pearson"] > 0.99, res
    assert res["per_gene_pearson"] > 0.99, res
    assert res["per_cell_gene_rmse"] < 1e-5, res


def test_gene_pearson_fidelity_empty_returns_nan():
    empty = np.zeros((0, 5), dtype=np.float32)
    coords_empty = np.zeros((0, 2), dtype=np.float32)
    truth = np.zeros((4, 5), dtype=np.float32)
    coords = np.zeros((4, 2), dtype=np.float32)
    res = gene_pearson_fidelity(empty, coords_empty, truth, coords)
    assert all(np.isnan(v) for v in res.values())
