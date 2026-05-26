"""
Wave 2 — cellular neighborhood co-localization matrix from 3D k-NN counts.

For every ordered pair (A, B) of cell classes, compute the fraction of class-A
cells that have at least one class-B neighbor within k nearest neighbors in 3D.
Significance asterisks come from a permutation test that shuffles class labels
n_permutations times and checks where the observed fraction lies in the null
distribution.

Outputs:
  <out_dir>/neighborhood_matrix.png  (heatmap + significance asterisks)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.spatial import cKDTree

from scripts.visualize._plot_utils import class_from_onehot


def _coloc_matrix(
    coords: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> np.ndarray:
    """Build observed co-localization matrix via k-NN in 3D.

    M[i, j] = fraction of class-i neighbors that are class-j.
    """
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k + 1)
    idxs = idxs[:, 1:]  # drop self
    unique = sorted(np.unique(classes).tolist())
    name_to_i = {c: i for i, c in enumerate(unique)}
    n_cls = len(unique)
    M = np.zeros((n_cls, n_cls), dtype=np.float64)
    counts = np.zeros(n_cls, dtype=np.int64)
    for ci in range(len(coords)):
        a = classes[ci]
        if a not in name_to_i:
            continue
        ai = name_to_i[a]
        counts[ai] += 1
        neigh_classes = classes[idxs[ci]]
        for b in np.unique(neigh_classes):
            if b in name_to_i:
                M[ai, name_to_i[b]] += 1.0
    M = M / np.maximum(counts[:, None], 1)
    return M


def _permutation_test(
    coords: np.ndarray,
    classes: np.ndarray,
    k: int,
    obs: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Compute two-sided p-values from a permutation null distribution.

    Returns p-value matrix of same shape as obs.
    """
    rng = np.random.default_rng(seed)
    n_cls = obs.shape[0]
    # Collect null distribution per cell (i, j)
    null_samples = np.zeros((n_permutations, n_cls, n_cls), dtype=np.float64)
    for p in range(n_permutations):
        shuffled = classes.copy()
        rng.shuffle(shuffled)
        null_samples[p] = _coloc_matrix(coords, shuffled, k)

    # Two-sided p-value: fraction of null abs-deviations >= observed abs-deviation
    null_mean = null_samples.mean(axis=0)
    obs_dev = np.abs(obs - null_mean)
    null_dev = np.abs(null_samples - null_mean)
    pvals = np.zeros_like(obs)
    for i in range(n_cls):
        for j in range(n_cls):
            pvals[i, j] = float((null_dev[:, i, j] >= obs_dev[i, j]).mean())
    return pvals


def render_neighborhood_matrix(
    volume: AnnData,
    out_path: Path,
    k: int = 10,
    n_permutations: int = 100,
) -> None:
    """Render 3D NN co-localization matrix with permutation significance.

    Args:
        volume: Reconstructed AnnData volume with obsm['spatial_3d'].
        out_path: Output path for the PNG figure.
        k: Number of nearest neighbors (default: 10).
        n_permutations: Number of label-shuffle permutations (default: 100).
    """
    if "spatial_3d" not in volume.obsm:
        return
    classes = class_from_onehot(volume)
    if classes is None:
        return
    coords = np.asarray(volume.obsm["spatial_3d"])

    # Subsample if very large for performance
    if coords.shape[0] > 6000:
        rng = np.random.default_rng(42)
        idx = rng.choice(coords.shape[0], 6000, replace=False)
        coords = coords[idx]
        classes = classes[idx]

    obs = _coloc_matrix(coords, classes, k)
    pvals = _permutation_test(coords, classes, k, obs, n_permutations, seed=42)

    unique = sorted(np.unique(classes).tolist())
    n_cls = len(unique)

    fig, ax = plt.subplots(
        figsize=(0.55 * n_cls + 2.5, 0.55 * n_cls + 2.0),
    )
    im = ax.imshow(obs, cmap="viridis", vmin=0, vmax=max(obs.max(), 0.001))
    ax.set_xticks(range(n_cls), unique, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_cls), unique, fontsize=7)
    ax.set_xlabel("neighbor class")
    ax.set_ylabel("anchor class")
    ax.set_title(
        f"3D cellular co-localization (k={k}, n_perm={n_permutations})\n"
        "observed fraction + significance asterisks",
        fontsize=9,
    )
    for i in range(n_cls):
        for j in range(n_cls):
            pv = float(pvals[i, j])
            if pv < 0.001:
                label = "***"
            elif pv < 0.01:
                label = "**"
            elif pv < 0.05:
                label = "*"
            else:
                continue
            # Choose text color for contrast against the heatmap
            val = float(obs[i, j])
            text_color = "white" if val < 0.5 * obs.max() else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.045, label="fraction")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
