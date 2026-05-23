"""
Wave 3 — cell-type marker-gene dot plot, input stack vs reconstructed volume.

Two stacked panels (top: input cells, bottom: reconstructed cells) of mean
expression dot size + colour for the same marker set across cell classes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, to_dense


def _dot_metrics(adata: AnnData, classes: np.ndarray, marker_idx: List[int]):
    X = to_dense(adata.X)[:, marker_idx]
    unique = sorted(np.unique(classes).tolist())
    mean = np.zeros((len(unique), len(marker_idx)), dtype=np.float32)
    pct = np.zeros((len(unique), len(marker_idx)), dtype=np.float32)
    for ci, c in enumerate(unique):
        mask = (classes == c)
        if mask.sum() == 0:
            continue
        Xc = X[mask]
        mean[ci] = Xc.mean(axis=0)
        pct[ci] = (Xc > 0).mean(axis=0)
    return unique, mean, pct


def _render_dot(ax, classes_sorted, mean: np.ndarray, pct: np.ndarray, marker_names: List[str], title: str, vmin: float, vmax: float) -> None:
    ax.set_title(title, fontsize=9)
    for ci in range(mean.shape[0]):
        for mi in range(mean.shape[1]):
            size = 20 + 280 * pct[ci, mi]
            color_val = mean[ci, mi]
            ax.scatter(mi, ci, s=size, c=[color_val], cmap="viridis", vmin=vmin, vmax=vmax,
                       edgecolor="#222", linewidth=0.4)
    ax.set_xticks(range(len(marker_names)), marker_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes_sorted)), classes_sorted, fontsize=7)
    ax.set_xlim(-0.5, len(marker_names) - 0.5)
    ax.set_ylim(-0.5, len(classes_sorted) - 0.5)
    ax.invert_yaxis()


def render_celltype_dotplot(
    volume: AnnData,
    input_slices: List[AnnData],
    out_path: Path,
    markers: Optional[List[str]] = None,
    n_default: int = 10,
) -> None:
    if not input_slices:
        return
    rec_classes = class_from_onehot(volume)
    if rec_classes is None:
        return
    in_X_list, in_cls_list = [], []
    for s in input_slices:
        if "cell_class" not in s.obs:
            continue
        in_X_list.append(to_dense(s.X))
        in_cls_list.append(s.obs["cell_class"].astype(str).to_numpy())
    if not in_X_list:
        return
    in_X = np.concatenate(in_X_list, axis=0)
    in_classes = np.concatenate(in_cls_list)

    common = [g for g in volume.var_names if g in input_slices[0].var_names]
    if not common:
        return
    if markers is None:
        # default = top-variance from the input stack
        in_idx = [list(input_slices[0].var_names).index(g) for g in common]
        var = in_X[:, in_idx].var(axis=0)
        order = np.argsort(var)[::-1][:n_default]
        markers = [common[i] for i in order]

    rec_mark_idx = [list(volume.var_names).index(g) for g in markers if g in volume.var_names]
    in_mark_idx = [list(input_slices[0].var_names).index(g) for g in markers if g in input_slices[0].var_names]
    if not rec_mark_idx or len(rec_mark_idx) != len(in_mark_idx):
        return

    in_unique, in_mean, in_pct = _dot_metrics(
        AnnData(X=in_X), in_classes, in_mark_idx,
    )
    rec_unique, rec_mean, rec_pct = _dot_metrics(volume, rec_classes, rec_mark_idx)

    vmin = float(min(in_mean.min(), rec_mean.min()))
    vmax = float(max(in_mean.max(), rec_mean.max()))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(0.4 * len(markers) + 2, 0.35 * (len(in_unique) + len(rec_unique)) + 2),
    )
    _render_dot(axes[0], in_unique, in_mean, in_pct, markers, "Input stack", vmin, vmax)
    _render_dot(axes[1], rec_unique, rec_mean, rec_pct, markers, "Aether3D reconstruction", vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, fraction=0.04, label="mean expression")
    fig.suptitle("Cell-class × marker dot plot (size = fraction expressing, colour = mean expression)", fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
