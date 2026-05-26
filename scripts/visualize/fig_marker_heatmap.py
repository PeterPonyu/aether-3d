"""
Wave 2 — cell-type × marker-gene mean-expression heatmap.

For each cell class (resolved via class_from_onehot), compute mean expression
of canonical marker genes. If input_slices are provided, renders as a 2-row
heatmap (input stack on top, reconstructed volume below). Otherwise renders
a single-row heatmap (reconstructed only).

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


def _mean_panel(
    adata: AnnData,
    classes: np.ndarray,
    gene_idx: List[int],
) -> np.ndarray:
    """Compute per-class mean expression for the given gene indices."""
    X = to_dense(adata.X)[:, gene_idx]
    unique = sorted(np.unique(classes).tolist())
    out = np.zeros((len(unique), len(gene_idx)), dtype=np.float32)
    for ci, c in enumerate(unique):
        mask = classes == c
        if mask.sum() == 0:
            continue
        out[ci] = X[mask].mean(axis=0)
    return out


def _canonical_marker_idx(
    volume: AnnData,
    input_slices: List[AnnData],
    n_markers: int = 12,
) -> Optional[List[int]]:
    """Select top-variance gene indices shared across volume and input slices."""
    if not input_slices:
        return None
    ref_var_names = list(input_slices[0].var_names)
    common_genes = [g for g in volume.var_names if g in ref_var_names]
    if not common_genes:
        return None
    # Use variance of input stack across common genes to pick markers
    in_X_list = [to_dense(s.X) for s in input_slices]
    in_X = np.concatenate(in_X_list, axis=0)
    in_var_idx_full = [ref_var_names.index(g) for g in common_genes]
    variances = in_X[:, in_var_idx_full].var(axis=0)
    order = np.argsort(variances)[::-1][:n_markers]
    markers = [common_genes[i] for i in order]
    rec_idx = [list(volume.var_names).index(g) for g in markers]
    return rec_idx


def render_marker_heatmap(
    volume: AnnData,
    out_path: Path,
    input_slices: Optional[List[AnnData]] = None,
) -> None:
    """Render cell-type × marker-gene mean-expression heatmap.

    Args:
        volume: Reconstructed AnnData volume.
        out_path: Output path for the PNG figure.
        input_slices: Optional list of input slice AnnData objects.
            If provided, renders a 2-row heatmap (input top, reconstructed
            bottom).  Otherwise renders a single-row heatmap.
    """
    rec_classes = class_from_onehot(volume)
    if rec_classes is None:
        return

    has_input = input_slices is not None and len(input_slices) > 0
    marker_idx: Optional[List[int]] = None
    in_panel: Optional[np.ndarray] = None
    in_classes_sorted: List[str] = []

    if has_input:
        marker_idx = _canonical_marker_idx(volume, input_slices)
        if marker_idx is None:
            return
        # Build input stack
        in_X_list: List[np.ndarray] = []
        in_cls_list: List[np.ndarray] = []
        for s in input_slices:
            if "cell_class" not in s.obs:
                continue
            in_X_list.append(to_dense(s.X))
            in_cls_list.append(s.obs["cell_class"].astype(str).to_numpy())
        if not in_X_list:
            return
        in_X = np.concatenate(in_X_list, axis=0)
        in_classes = np.concatenate(in_cls_list, axis=0)
        in_marker_idx = [
            list(input_slices[0].var_names).index(g)
            for g in [str(volume.var_names[i]) for i in marker_idx]
        ]
        in_panel = _mean_panel(
            AnnData(X=in_X),
            in_classes,
            in_marker_idx,
        )
        in_classes_sorted = sorted(np.unique(in_classes).tolist())
    else:
        # No input slices: pick top-variance genes from the volume itself
        X_vol = to_dense(volume.X)
        variances = X_vol.var(axis=0)
        n_markers = min(12, volume.n_vars)
        top_idx = np.argsort(variances)[::-1][:n_markers]
        marker_idx = top_idx.tolist()

    rec_panel = _mean_panel(volume, rec_classes, marker_idx)
    rec_classes_sorted = sorted(np.unique(rec_classes).tolist())

    # Gather gene labels
    marker_names = [str(volume.var_names[i]) for i in marker_idx]

    # Determine vmin/vmax
    if in_panel is not None:
        vmin = float(min(rec_panel.min(), in_panel.min()))
        vmax = float(max(rec_panel.max(), in_panel.max()))
    else:
        vmin = float(rec_panel.min())
        vmax = float(rec_panel.max())

    n_rows = 2 if in_panel is not None else 1
    n_cols = 1
    height_per_row = 0.4 * max(len(rec_classes_sorted), len(in_classes_sorted)) + 1.0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 + 0.4 * len(marker_names), height_per_row * n_rows),
        squeeze=False,
    )

    # Row 0: input (if available)
    if in_panel is not None:
        ax = axes[0, 0]
        im = ax.imshow(in_panel, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(
            range(len(marker_names)), marker_names, rotation=60, ha="right", fontsize=7
        )
        ax.set_yticks(range(len(in_classes_sorted)), in_classes_sorted, fontsize=7)
        ax.set_title("Input stack", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)

    # Row 1 (or 0): reconstructed
    row_idx = 0 if in_panel is None else 1
    ax = axes[row_idx, 0]
    im = ax.imshow(rec_panel, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(
        range(len(marker_names)), marker_names, rotation=60, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(rec_classes_sorted)), rec_classes_sorted, fontsize=7)
    ax.set_title("Aether3D reconstruction", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)

    title = (
        "Cell-type × marker mean expression: input stack vs reconstruction"
        if in_panel is not None
        else "Cell-type × marker mean expression: reconstruction"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
