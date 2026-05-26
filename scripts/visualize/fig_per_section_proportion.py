"""
Wave 2 — per-input-section cell-type stacked-bar grid.

If the volume has ``obs['source_slice']``, groups cells by source slice,
computes cell-type proportions per slice, and renders a small-multiples
stacked-bar grid.  Otherwise falls back to a single overall stacked bar.

Outputs:
  <out_dir>/per_section_proportion.png
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


def _proportion(
    labels: np.ndarray,
    classes_sorted: List[str],
) -> np.ndarray:
    """Compute normalized proportion of each class in labels."""
    total = max(len(labels), 1)
    return np.array(
        [float((labels == c).sum()) / total for c in classes_sorted],
        dtype=np.float64,
    )


def render_per_section_proportion(
    volume: AnnData,
    out_path: Path,
) -> None:
    """Render per-slice cell-type proportion stacked-bar grid.

    Args:
        volume: Reconstructed AnnData volume. If ``obs['source_slice']``
            is present, cells are grouped by slice. Otherwise overall
            proportions are shown as a single stacked bar.
        out_path: Output path for the PNG figure.
    """
    classes = class_from_onehot(volume)
    if classes is None:
        return
    palette = stable_categorical_colors(classes)
    classes_sorted = sorted(np.unique(classes).tolist())

    if "source_slice" in volume.obs:
        slice_ids = volume.obs["source_slice"].astype(str).tolist()
        slice_unique = sorted(set(slice_ids), key=lambda x: (len(x), x))
        n_slices = len(slice_unique)

        fig, axes = plt.subplots(
            1,
            n_slices,
            figsize=(max(2.0 * n_slices, 4), 4.5),
            squeeze=False,
            sharey=True,
        )
        for si, sid in enumerate(slice_unique):
            mask = np.array([s == sid for s in slice_ids])
            labels = classes[mask]
            prop = _proportion(labels, classes_sorted)
            bottom = 0.0
            ax = axes[0, si]
            for ci, c in enumerate(classes_sorted):
                ax.bar(
                    0,
                    prop[ci],
                    bottom=bottom,
                    color=palette[str(c)],
                    width=0.7,
                    label=str(c) if si == 0 else None,
                )
                bottom += prop[ci]
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            ax.set_title(f"Slice {sid}\n({mask.sum()} cells)", fontsize=8)
        if classes_sorted:
            axes[0, 0].legend(
                fontsize=6,
                loc="center left",
                bbox_to_anchor=(-1.1, 0.5),
            )
        fig.suptitle("Per-section cell-type proportion", fontsize=10)
    else:
        # Single overall stacked bar
        prop = _proportion(classes, classes_sorted)
        fig, ax = plt.subplots(figsize=(2.5, 4.5))
        bottom = 0.0
        for ci, c in enumerate(classes_sorted):
            ax.bar(
                0,
                prop[ci],
                bottom=bottom,
                color=palette[str(c)],
                width=0.7,
                label=str(c),
            )
            bottom += prop[ci]
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_title(f"Overall\n({len(classes)} cells)", fontsize=8)
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.05, 0.5))
        fig.suptitle("Overall cell-type proportion", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
