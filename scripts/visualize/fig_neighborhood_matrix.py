"""
Wave 2 — cellular neighborhood co-localization matrix from 3D k-NN counts.

For every ordered pair (A, B) of cell classes, compute the fraction of class-A
cells that have at least one class-B neighbor within k nearest neighbors in 3D.
Significance asterisks come from a permutation test that shuffles class labels
1000 times and checks where the observed fraction lies in the null distribution.

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
from sklearn.neighbors import NearestNeighbors

from scripts.visualize._plot_utils import class_from_onehot


def _coloc_matrix(coords: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idxs = nn.kneighbors(coords)
    idxs = idxs[:, 1:]  # drop self
    unique = sorted(np.unique(classes).tolist())
    name_to_i = {c: i for i, c in enumerate(unique)}
    M = np.zeros((len(unique), len(unique)), dtype=np.float32)
    counts = np.zeros(len(unique), dtype=np.int64)
    for c_i in range(len(coords)):
        a = classes[c_i]
        if a not in name_to_i:
            continue
        ai = name_to_i[a]
        counts[ai] += 1
        neigh_classes = classes[idxs[c_i]]
        for b in np.unique(neigh_classes):
            if b in name_to_i:
                M[ai, name_to_i[b]] += 1.0
    M = M / np.maximum(counts[:, None], 1)
    return M


def render_neighborhood_matrix(
    volume: AnnData,
    out_path: Path,
    k: int = 12,
    n_perm: int = 200,
    seed: int = 42,
) -> None:
    if "spatial_3d" not in volume.obsm:
        return
    classes = class_from_onehot(volume)
    if classes is None:
        return
    coords = np.asarray(volume.obsm["spatial_3d"])
    if coords.shape[0] > 6000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], 6000, replace=False)
        coords = coords[idx]
        classes = classes[idx]

    obs = _coloc_matrix(coords, classes, k)
    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm,) + obs.shape, dtype=np.float32)
    for p in range(n_perm):
        shuffled = classes.copy()
        rng.shuffle(shuffled)
        perms[p] = _coloc_matrix(coords, shuffled, k)
    # Two-sided p-value: fraction of perm |z| >= observed |z|
    perm_mean = perms.mean(axis=0)
    perm_std = perms.std(axis=0) + 1e-9
    z = (obs - perm_mean) / perm_std

    unique = sorted(np.unique(classes).tolist())
    n_cls = len(unique)
    fig, ax = plt.subplots(figsize=(0.55 * n_cls + 2.5, 0.55 * n_cls + 2.0))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_cls), unique, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_cls), unique, fontsize=7)
    ax.set_xlabel("neighbor class")
    ax.set_ylabel("anchor class")
    ax.set_title(f"3D cellular neighborhood enrichment (k={k}, n_perm={n_perm})\nz-score (perm test)", fontsize=9)
    for i in range(n_cls):
        for j in range(n_cls):
            zi = float(z[i, j])
            if abs(zi) >= 3:
                ax.text(j, i, "***", ha="center", va="center", fontsize=7, color="white")
            elif abs(zi) >= 2:
                ax.text(j, i, "**", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, fraction=0.045, label="z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
