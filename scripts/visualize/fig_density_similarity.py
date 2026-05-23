"""
Wave 1 — per-cell-class 3D KDE density similarity between reconstructed volume
and the input slice stack.

For each cell class:
  1. Build a coarse 3D voxel histogram from the reconstructed volume (`obsm['spatial_3d']`)
  2. Build the same from the stacked input slices (`obsm['spatial']` + `obs['z_coord']`)
  3. Report cosine similarity between the two voxel-histogram vectors

Outputs:
  <out_dir>/density_similarity_bars.png  (grouped bar chart, similarity per class + per-class voxel counts)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors


def _voxel_histogram(coords: np.ndarray, bins: List[np.ndarray]) -> np.ndarray:
    H, _ = np.histogramdd(coords, bins=bins)
    return H.ravel()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def render_density_similarity(
    volume: AnnData,
    input_slices: List[AnnData],
    out_path: Path,
    n_voxels: int = 12,
) -> None:
    if not input_slices:
        return
    classes = class_from_onehot(volume)
    if classes is None:
        return
    coords_rec = np.asarray(volume.obsm["spatial_3d"])

    # Build input-stack coords + class array
    coords_in_list, classes_in_list = [], []
    for s in input_slices:
        if "spatial" not in s.obsm:
            continue
        xy = np.asarray(s.obsm["spatial"])[:, :2]
        if "z_coord" in s.obs:
            z = np.asarray(s.obs["z_coord"], dtype=float).reshape(-1, 1)
        else:
            z = np.zeros((xy.shape[0], 1))
        coords_in_list.append(np.hstack([xy, z]))
        if "cell_class" in s.obs:
            classes_in_list.append(s.obs["cell_class"].astype(str).to_numpy())
        else:
            classes_in_list.append(np.array(["?"] * xy.shape[0]))
    if not coords_in_list:
        return
    coords_in = np.concatenate(coords_in_list, axis=0)
    classes_in = np.concatenate(classes_in_list, axis=0)

    # Shared voxel grid across both
    lo = np.minimum(coords_rec.min(axis=0), coords_in.min(axis=0))
    hi = np.maximum(coords_rec.max(axis=0), coords_in.max(axis=0))
    bins = [np.linspace(lo[d], hi[d] + 1e-6, n_voxels + 1) for d in range(3)]

    common = sorted(set(np.unique(classes)) & set(np.unique(classes_in)))
    sims: Dict[str, float] = {}
    rec_cells: Dict[str, int] = {}
    in_cells: Dict[str, int] = {}
    for c in common:
        mr = (classes == c)
        mi = (classes_in == c)
        if mr.sum() < 5 or mi.sum() < 5:
            continue
        hr = _voxel_histogram(coords_rec[mr], bins)
        hi_ = _voxel_histogram(coords_in[mi], bins)
        sims[c] = _cosine(hr, hi_)
        rec_cells[c] = int(mr.sum())
        in_cells[c] = int(mi.sum())

    if not sims:
        # Class labels differ between reconstruction and input (e.g. reconstruction
        # uses argmax indices of predicted one-hot, input uses biological names).
        # Compare pooled densities so the figure still conveys overall 3D shape match.
        hr = _voxel_histogram(coords_rec, bins)
        hi_ = _voxel_histogram(coords_in, bins)
        sims["all"] = _cosine(hr, hi_)
        rec_cells["all"] = int(coords_rec.shape[0])
        in_cells["all"] = int(coords_in.shape[0])

    names = list(sims.keys())
    palette = stable_categorical_colors(np.array(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    bars = ax1.bar(names, [sims[n] for n in names], color=[palette[n] for n in names])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Cosine similarity (reconstructed vs input KDE)")
    ax1.set_title("3D density-shape similarity per cell class")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    for b, v in zip(bars, sims.values()):
        ax1.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)

    x = np.arange(len(names))
    w = 0.4
    ax2.bar(x - w / 2, [rec_cells[n] for n in names], w, label="reconstructed", color="#4C72B0")
    ax2.bar(x + w / 2, [in_cells[n] for n in names], w, label="input", color="#C44E52")
    ax2.set_xticks(x, names)
    ax2.set_title("Cell count per class")
    ax2.set_ylabel("Cells")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
