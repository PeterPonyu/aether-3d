"""
Wave 2 — per-input-section cell-type stacked-bar grid.

For each input slice (real or synthetic), render a normalized stacked bar of
cell-type proportions; arrange one row of bars for input vs a matching row
where each input slice's Z-position maps to the nearest reconstructed Z band.

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


def _proportion(labels: np.ndarray, classes_sorted: List[str]) -> np.ndarray:
    total = len(labels) or 1
    return np.array([float((labels == c).sum()) / total for c in classes_sorted], dtype=np.float32)


def render_per_section_proportion(
    volume: AnnData, input_slices: List[AnnData], out_path: Path
) -> None:
    if not input_slices:
        return
    # Gather classes from input slices
    in_class_arrs = []
    in_z = []
    for s in input_slices:
        if "cell_class" not in s.obs:
            continue
        in_class_arrs.append(s.obs["cell_class"].astype(str).to_numpy())
        z = s.obs["z_coord"].iloc[0] if "z_coord" in s.obs else 0.0
        in_z.append(float(z))
    if not in_class_arrs:
        return
    all_in = np.concatenate(in_class_arrs)
    rec_classes = class_from_onehot(volume)
    rec_palette = stable_categorical_colors(rec_classes if rec_classes is not None else np.array(["?"]))
    in_palette = stable_categorical_colors(all_in)

    # Map input classes to reconstructed Z bands by Z proximity
    if "spatial_3d" in volume.obsm and rec_classes is not None:
        rec_z = np.asarray(volume.obsm["spatial_3d"])[:, 2]
        z_bands = np.linspace(rec_z.min(), rec_z.max(), len(input_slices) + 1)
    else:
        z_bands = None

    n = len(in_class_arrs)
    in_classes_sorted = sorted(np.unique(all_in).tolist())
    rec_classes_sorted = sorted(np.unique(rec_classes).tolist()) if rec_classes is not None else []

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5), squeeze=False, sharey=True)
    for i, (labels, z) in enumerate(zip(in_class_arrs, in_z)):
        prop = _proportion(labels, in_classes_sorted)
        bottom = 0.0
        for ci, c in enumerate(in_classes_sorted):
            axes[0, i].bar(0, prop[ci], bottom=bottom, color=in_palette[c], width=0.7, label=str(c) if i == 0 else None)
            bottom += prop[ci]
        axes[0, i].set_xticks([])
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_title(f"Input slice {i}\n(z={z:.1f})", fontsize=8)

    for i, z in enumerate(in_z):
        if z_bands is None or rec_classes is None:
            axes[1, i].set_axis_off()
            continue
        # Take the band whose center is closest to this input Z
        band_centers = 0.5 * (z_bands[:-1] + z_bands[1:])
        b = int(np.argmin(np.abs(band_centers - z)))
        mask = (np.asarray(volume.obsm["spatial_3d"])[:, 2] >= z_bands[b]) & \
               (np.asarray(volume.obsm["spatial_3d"])[:, 2] < z_bands[b + 1])
        labels = rec_classes[mask]
        prop = _proportion(labels, rec_classes_sorted)
        bottom = 0.0
        for ci, c in enumerate(rec_classes_sorted):
            axes[1, i].bar(0, prop[ci], bottom=bottom, color=rec_palette[c], width=0.7, label=str(c) if i == 0 else None)
            bottom += prop[ci]
        axes[1, i].set_xticks([])
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f"Reconstructed band\n[{z_bands[b]:.1f}, {z_bands[b+1]:.1f}]", fontsize=8)

    if in_classes_sorted:
        axes[0, 0].legend(fontsize=6, loc="center left", bbox_to_anchor=(-1.1, 0.5))
    if rec_classes_sorted:
        axes[1, 0].legend(fontsize=6, loc="center left", bbox_to_anchor=(-1.1, 0.5))
    fig.suptitle("Per-section cell-type proportion: input slices (top) vs Aether3D Z-band (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
