"""Regression tests for benchmark / contract metrics.

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

Issue #133 — `betti0_stability` was an integer component-count ratio
(min/max). For any reasonably dense planar slice the k-NN graph collapses
to one component, so a fully *collapsed* reconstruction (every cell at
one location) scored 1.0 — "perfect" topology preservation for a
clearly broken output.
"""

from __future__ import annotations

import numpy as np

from aether_3d.benchmarks.metrics import (
    gene_pearson_fidelity,
    voxel_cosine_similarity,
)
from aether_3d.benchmarks.topology import betti_zero_stability


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


def test_betti0_collapsed_not_one() -> None:
    """A constant-output reconstruction must NOT score ~1.0.

    Truth: a dense 2D blob with non-trivial spatial extent.
    Recon: every cell mapped to the same point (degenerate collapse).
    The metric must clearly separate this from a faithful identity
    reconstruction.
    """
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.uniform(-10.0, 10.0, size=(n, 2)).astype(np.float32)

    # Collapsed reconstruction: every point at (0, 0).
    collapsed = np.zeros_like(truth)

    score_collapsed = betti_zero_stability(truth, collapsed, k=6)
    score_identity = betti_zero_stability(truth, truth.copy(), k=6)

    # Identity reconstruction is still ~1.0 (sanity check).
    assert score_identity > 0.95, (
        f"identity reconstruction should score near 1.0; got {score_identity}"
    )

    # Collapsed reconstruction must clearly differ from identity.
    assert score_collapsed < 0.05, (
        "collapsed reconstruction must NOT score ~1.0 on betti0_stability; "
        f"got {score_collapsed}"
    )
    assert score_collapsed < score_identity - 0.5, (
        f"collapsed ({score_collapsed}) should be much worse than identity "
        f"({score_identity})"
    )


def _grid_cloud(
    seed: int, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """A reproducible 3D point cloud with a spatially-structured cell-type field.

    Cell type is assigned from the x-octant so that local composition varies
    across space — exactly the signal voxel cosine similarity is meant to read.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n, 3)).astype(np.float64)
    # 4 spatially-banded cell types along x => non-uniform local composition.
    labels = np.array(
        [f"T{int(x // 25)}" for x in coords[:, 0]], dtype=object
    )
    return coords, labels


def test_voxel_cosine_identical_is_one():
    """Identical truth/recon point sets => mean voxel cosine == 1.0."""
    coords, labels = _grid_cloud(seed=0)
    score = voxel_cosine_similarity(
        coords, labels, coords.copy(), labels.copy(), n_bins=5
    )
    assert isinstance(score, float)
    assert abs(score - 1.0) < 1e-9, score


def test_voxel_cosine_spatial_scramble_scores_lower():
    """Scrambling cell-type labels across space (global proportions preserved)
    must score markedly lower on the voxel-local metric, even though the global
    `celltype_distribution_cosine` stays ~1.0 — pinning the local/global split.
    """
    from aether_3d.benchmarks.metrics import celltype_distribution_cosine

    coords, labels = _grid_cloud(seed=1)
    rng = np.random.default_rng(99)
    scrambled = labels[rng.permutation(labels.shape[0])]

    voxel_score = voxel_cosine_similarity(
        coords, labels, coords.copy(), scrambled, n_bins=5
    )
    # Global proportion cosine is permutation-invariant => essentially 1.0.
    global_score = celltype_distribution_cosine(
        labels.tolist(), scrambled.tolist()
    )

    assert isinstance(voxel_score, float)
    assert global_score > 0.999, global_score
    assert voxel_score < 0.9, voxel_score
    assert voxel_score < global_score - 0.05, (voxel_score, global_score)


def test_voxel_cosine_downsample_reconstruct_is_sensible():
    """A down-sample => reconstruct toy: a faithful (subsampled) reconstruction
    should score high, while a spatially-shifted reconstruction scores lower.
    """
    coords, labels = _grid_cloud(seed=2, n=600)
    rng = np.random.default_rng(7)
    keep = rng.choice(coords.shape[0], size=300, replace=False)
    recon_coords = coords[keep]
    recon_labels = labels[keep]

    faithful = voxel_cosine_similarity(
        coords, labels, recon_coords, recon_labels, n_bins=4
    )
    # Shift recon by ~half the domain along x => cell types land in wrong voxels.
    shifted_coords = recon_coords.copy()
    shifted_coords[:, 0] = shifted_coords[:, 0] + 50.0
    shifted = voxel_cosine_similarity(
        coords, labels, shifted_coords, recon_labels, n_bins=4
    )

    assert isinstance(faithful, float) and isinstance(shifted, float)
    assert faithful > 0.95, faithful
    assert shifted < faithful, (shifted, faithful)


def test_voxel_cosine_return_per_voxel_and_empty():
    """`return_per_voxel` yields the per-voxel array; empty input => NaN."""
    coords, labels = _grid_cloud(seed=3, n=200)
    out = voxel_cosine_similarity(
        coords, labels, coords.copy(), labels.copy(), n_bins=3,
        return_per_voxel=True,
    )
    assert isinstance(out, tuple)
    mean_cos, per_voxel = out
    assert per_voxel.ndim == 1 and per_voxel.size > 0
    assert abs(mean_cos - float(np.mean(per_voxel))) < 1e-12

    empty = np.zeros((0, 3), dtype=np.float64)
    nan_score = voxel_cosine_similarity(empty, [], coords, labels, n_bins=3)
    assert isinstance(nan_score, float) and np.isnan(nan_score)
