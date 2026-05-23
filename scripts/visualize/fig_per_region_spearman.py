"""
Wave 3 — per-region Spearman scatter of cell-type proportions.

Splits the reconstructed volume into spatial "regions" (3D voxel bins),
computes the cell-class proportion vector per region, and compares to the
proportion vector of the nearest input slice (per region — assign each region
to its nearest input slice by Z). Scatter of all (region, class) values
input vs reconstructed; Spearman R reported in the title.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.stats import spearmanr

from scripts.visualize._plot_utils import class_from_onehot


def render_per_region_spearman(
    volume: AnnData, input_slices: List[AnnData], out_path: Path, n_bins: int = 6
) -> None:
    if not input_slices:
        return
    classes = class_from_onehot(volume)
    if classes is None:
        return
    coords = np.asarray(volume.obsm["spatial_3d"])
    z = coords[:, 2]
    z_bins = np.linspace(z.min(), z.max(), n_bins + 1)

    # Map input slice z to nearest reconstructed bin
    slice_zs = np.array([
        float(s.obs["z_coord"].iloc[0]) if "z_coord" in s.obs else i * 10.0
        for i, s in enumerate(input_slices)
    ])

    rec_unique = sorted(np.unique(classes).tolist())
    in_class_arrs = [s.obs["cell_class"].astype(str).to_numpy() if "cell_class" in s.obs else None
                     for s in input_slices]
    in_unique_all = sorted({c for arr in in_class_arrs if arr is not None for c in np.unique(arr)})
    if not in_unique_all:
        return

    # Build per-region proportion vectors using REC classes
    rec_props = np.zeros((n_bins, len(rec_unique)), dtype=np.float32)
    for b in range(n_bins):
        in_band = (z >= z_bins[b]) & (z < z_bins[b + 1])
        labels = classes[in_band]
        total = max(len(labels), 1)
        for ci, c in enumerate(rec_unique):
            rec_props[b, ci] = float((labels == c).sum()) / total
    # Build per-region proportion using INPUT classes for matched slice
    in_props = np.zeros((n_bins, len(in_unique_all)), dtype=np.float32)
    band_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    for b in range(n_bins):
        slice_i = int(np.argmin(np.abs(slice_zs - band_centers[b])))
        arr = in_class_arrs[slice_i]
        if arr is None:
            continue
        total = max(len(arr), 1)
        for ci, c in enumerate(in_unique_all):
            in_props[b, ci] = float((arr == c).sum()) / total

    # Spearman correlation per band between flatten input and reconstructed proportions
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    rs = []
    for b in range(n_bins):
        x = in_props[b]
        y_padded = np.zeros(len(in_unique_all), dtype=np.float32)
        # If rec classes overlap input names, copy in; otherwise align by descending sort
        for ci, c in enumerate(in_unique_all):
            if c in rec_unique:
                y_padded[ci] = rec_props[b, rec_unique.index(c)]
        ax.scatter(x, y_padded, s=12, alpha=0.75, label=f"Z band {b}")
        if x.std() > 1e-9 and y_padded.std() > 1e-9:
            r, _ = spearmanr(x, y_padded)
            rs.append(r)
    mean_r = float(np.nanmean(rs)) if rs else float("nan")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=0.8)
    ax.set_xlabel("Input slice cell-class proportion")
    ax.set_ylabel("Aether3D Z-band cell-class proportion")
    ax.set_title(f"Per-region cell-class proportions (mean Spearman R = {mean_r:.3f})")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
