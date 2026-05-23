"""
Wave 3 — side-by-side rendering of input vs Aether3D vs naive-2.5D baseline.

Three rows of XY/XZ projections (input stack, Aether3D, naive baseline) with
class colouring. Pure visual comparison — no pass/fail gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors


def _scatter(ax, xy: np.ndarray, classes: np.ndarray, palette: dict, title: str) -> None:
    for c in np.unique(classes):
        m = (classes == c)
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], c=palette.get(str(c), "#888"), s=2)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def render_recon_vs_25d(
    input_slices: List[AnnData],
    aether_volume: AnnData,
    naive_volume: AnnData,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))

    # Input stack: flatten all input cells, color by class
    in_xy_list, in_z_list, in_cls_list = [], [], []
    for s in input_slices:
        if "spatial" not in s.obsm:
            continue
        in_xy_list.append(np.asarray(s.obsm["spatial"])[:, :2])
        z = float(s.obs["z_coord"].iloc[0]) if "z_coord" in s.obs else 0.0
        in_z_list.append(np.full(s.n_obs, z))
        in_cls_list.append(s.obs["cell_class"].astype(str).to_numpy() if "cell_class" in s.obs else np.array(["?"] * s.n_obs))
    in_xy = np.concatenate(in_xy_list); in_z = np.concatenate(in_z_list); in_cls = np.concatenate(in_cls_list)
    in_palette = stable_categorical_colors(in_cls)
    _scatter(axes[0, 0], in_xy, in_cls, in_palette, "Input stack — XY (all cells)")
    _scatter(axes[0, 1], np.column_stack([in_xy[:, 0], in_z]), in_cls, in_palette, "Input stack — XZ")

    # Aether3D
    a_xyz = np.asarray(aether_volume.obsm["spatial_3d"])
    a_cls = class_from_onehot(aether_volume)
    a_cls = a_cls if a_cls is not None else np.array(["?"] * aether_volume.n_obs)
    a_palette = stable_categorical_colors(a_cls)
    _scatter(axes[1, 0], a_xyz[:, :2], a_cls, a_palette, f"Aether3D — XY ({aether_volume.n_obs:,} cells)")
    _scatter(axes[1, 1], np.column_stack([a_xyz[:, 0], a_xyz[:, 2]]), a_cls, a_palette, "Aether3D — XZ")

    # Naive 2.5D
    n_xyz = np.asarray(naive_volume.obsm["spatial_3d"])
    n_cls = naive_volume.obs["cell_class"].astype(str).to_numpy() if "cell_class" in naive_volume.obs else np.array(["?"] * naive_volume.n_obs)
    n_palette = stable_categorical_colors(n_cls)
    _scatter(axes[2, 0], n_xyz[:, :2], n_cls, n_palette, f"Naive 2.5D baseline — XY ({naive_volume.n_obs:,} cells)")
    _scatter(axes[2, 1], np.column_stack([n_xyz[:, 0], n_xyz[:, 2]]), n_cls, n_palette, "Naive 2.5D baseline — XZ")

    fig.suptitle(
        "Input stack vs Aether3D vs naive identity-preserving 2.5D\n"
        "Naive baseline is a lower bound — NOT a reproduction of any published 2.5D method.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
