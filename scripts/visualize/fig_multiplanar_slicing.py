"""
Wave 3 — multi-planar virtual cross-sections of the reconstructed volume.

For each of three orthogonal planes (coronal Z-fixed, sagittal X-fixed,
horizontal Y-fixed), take a thin band centered at the midpoint and render
two views: (a) cells coloured by class, (b) cells coloured by a marker gene.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, select_markers_by_group, stable_categorical_colors, to_dense


def render_multiplanar_slicing(volume: AnnData, out_path: Path, marker: Optional[str] = None) -> None:
    coords = np.asarray(volume.obsm["spatial_3d"])
    classes = class_from_onehot(volume)
    if classes is None:
        classes = np.array(["?"] * volume.n_obs)
    palette = stable_categorical_colors(classes)

    if marker is None:
        # pick a class-marker if possible
        if "cell_class" in volume.obs:
            ms = select_markers_by_group(volume, "cell_class", n_per_group=1)
            for cls_markers in ms.values():
                if cls_markers:
                    marker = cls_markers[0]
                    break
    if marker is None and volume.n_vars > 0:
        marker = str(volume.var_names[int(np.argmax(to_dense(volume.X).var(axis=0)))])

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    mids = 0.5 * (mins + maxs)
    bands = 0.04 * (maxs - mins)

    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    plane_specs = [
        ("Coronal (Z=mid)", 2, mids[2], bands[2], (0, 1)),
        ("Sagittal (X=mid)", 0, mids[0], bands[0], (1, 2)),
        ("Horizontal (Y=mid)", 1, mids[1], bands[1], (0, 2)),
    ]
    gene_idx = list(volume.var_names).index(marker) if marker in volume.var_names else 0
    gene_vals = to_dense(volume.X)[:, gene_idx]
    vmax = float(np.percentile(gene_vals, 99) + 1e-9)

    for r, (title, axis, center, band, (xi, yi)) in enumerate(plane_specs):
        m = np.abs(coords[:, axis] - center) <= band
        ax_cls = axes[r, 0]
        ax_gene = axes[r, 1]
        for c in np.unique(classes):
            mask = m & (classes == c)
            if mask.sum() == 0:
                continue
            ax_cls.scatter(coords[mask, xi], coords[mask, yi], c=palette[str(c)], s=3)
        ax_cls.set_title(f"{title}\nclass colouring ({int(m.sum())} cells in band)", fontsize=9)
        ax_cls.set_xticks([]); ax_cls.set_yticks([])
        ax_cls.set_aspect("equal", adjustable="datalim")

        sc_h = ax_gene.scatter(coords[m, xi], coords[m, yi], c=gene_vals[m], s=3, cmap="viridis", vmin=0, vmax=vmax)
        ax_gene.set_title(f"{title}\n{marker} expression", fontsize=9)
        ax_gene.set_xticks([]); ax_gene.set_yticks([])
        ax_gene.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc_h, ax=ax_gene, fraction=0.04, label=str(marker))
    fig.suptitle(f"Multi-planar virtual cross-sections of the reconstructed volume", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
