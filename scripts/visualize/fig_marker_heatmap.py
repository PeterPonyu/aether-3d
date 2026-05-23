"""
Wave 2 — cell-type x marker-gene mean-expression heatmap.

For each cell class (resolved via class_from_onehot), compute mean expression
of a small marker panel. Show two heatmaps side-by-side: input-stack vs
reconstructed-volume. Markers default to the top-variance genes shared between
input and reconstruction; an explicit panel can be supplied for biological
specificity.

Outputs:
  <out_dir>/marker_heatmap.png
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


def _mean_panel(adata: AnnData, classes: np.ndarray, gene_idx: List[int]) -> np.ndarray:
    X = to_dense(adata.X)[:, gene_idx]
    unique = sorted(np.unique(classes).tolist())
    out = np.zeros((len(unique), len(gene_idx)), dtype=np.float32)
    for ci, c in enumerate(unique):
        mask = (classes == c)
        if mask.sum() == 0:
            continue
        out[ci] = X[mask].mean(axis=0)
    return out


def render_marker_heatmap(
    volume: AnnData,
    input_slices: List[AnnData],
    out_path: Path,
    markers: Optional[List[str]] = None,
    n_default_markers: int = 12,
) -> None:
    if not input_slices:
        return
    rec_classes = class_from_onehot(volume)
    if rec_classes is None:
        return
    # Stack input cells + their classes
    in_X_list, in_cls_list = [], []
    for s in input_slices:
        if "cell_class" not in s.obs:
            continue
        in_X_list.append(to_dense(s.X))
        in_cls_list.append(s.obs["cell_class"].astype(str).to_numpy())
    if not in_X_list:
        return
    in_X = np.concatenate(in_X_list, axis=0)
    in_classes = np.concatenate(in_cls_list, axis=0)

    common_genes = [g for g in volume.var_names if g in input_slices[0].var_names]
    if not common_genes:
        return
    if markers is None:
        # top-variance genes from the input stack as defaults
        in_var_idx_full = [list(input_slices[0].var_names).index(g) for g in common_genes]
        variances = in_X[:, in_var_idx_full].var(axis=0)
        order = np.argsort(variances)[::-1][:n_default_markers]
        markers = [common_genes[i] for i in order]

    rec_idx = [list(volume.var_names).index(g) for g in markers if g in volume.var_names]
    in_idx = [list(input_slices[0].var_names).index(g) for g in markers if g in input_slices[0].var_names]
    if not rec_idx or len(rec_idx) != len(in_idx):
        return

    rec_panel = _mean_panel(volume, rec_classes, rec_idx)
    in_panel = _mean_panel(
        AnnData(X=in_X, var=input_slices[0].var.iloc[:in_X.shape[1]] if hasattr(input_slices[0], "var") else None),
        in_classes, in_idx,
    )

    rec_classes_sorted = sorted(np.unique(rec_classes).tolist())
    in_classes_sorted = sorted(np.unique(in_classes).tolist())
    vmin = float(min(rec_panel.min(), in_panel.min()))
    vmax = float(max(rec_panel.max(), in_panel.max()))

    fig, (ax_in, ax_rec) = plt.subplots(1, 2, figsize=(2 + 0.4 * len(markers) * 2, 0.4 * max(len(rec_classes_sorted), len(in_classes_sorted)) + 1.5))
    for ax, panel, classes_sorted, title in [
        (ax_in, in_panel, in_classes_sorted, "Input stack"),
        (ax_rec, rec_panel, rec_classes_sorted, "Aether3D reconstruction"),
    ]:
        im = ax.imshow(panel, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(markers)), markers, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes_sorted)), classes_sorted, fontsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Cell-type × marker mean expression: input stack vs reconstruction", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
