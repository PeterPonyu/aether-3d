"""
Wave 2 — Z-axis density of each cell class within a radius of an anchor class.

For an anchor class chosen as the largest non-empty class (or supplied
explicitly), compute the 3D nearest neighbours of every anchor cell, count
cells of each class along Z within a configurable radius, and plot smoothed
density curves per class.

Outputs:
  <out_dir>/z_density_anchored_<anchor>.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors


def render_z_density_anchored(
    volume: AnnData,
    out_path_template: str,
    anchor_class: Optional[str] = None,
    radius_um: float = 20.0,
    n_z_bins: int = 24,
) -> Optional[str]:
    """Returns the path string written, or None if it bailed."""
    if "spatial_3d" not in volume.obsm:
        return None
    classes = class_from_onehot(volume)
    if classes is None:
        return None
    coords = np.asarray(volume.obsm["spatial_3d"])
    unique = sorted(np.unique(classes).tolist())
    if anchor_class is None:
        counts = {c: int((classes == c).sum()) for c in unique}
        anchor_class = max(counts, key=counts.get)
    if anchor_class not in unique:
        return None

    anchor_mask = (classes == anchor_class)
    if anchor_mask.sum() == 0:
        return None

    nn = NearestNeighbors(radius=radius_um).fit(coords)
    _, idxs = nn.radius_neighbors(coords[anchor_mask], return_distance=True)

    z = coords[:, 2]
    bins = np.linspace(z.min(), z.max(), n_z_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    densities = {c: np.zeros(n_z_bins, dtype=np.float32) for c in unique}
    for neigh in idxs:
        for ni in neigh:
            c = classes[ni]
            b = min(int(np.searchsorted(bins, z[ni], side="right") - 1), n_z_bins - 1)
            if 0 <= b < n_z_bins:
                densities[c][b] += 1.0
    for c in densities:
        densities[c] = gaussian_filter1d(densities[c], sigma=1.0)

    palette = stable_categorical_colors(np.array(unique))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c in unique:
        ax.plot(centers, densities[c], label=str(c), color=palette[c], linewidth=1.5)
    ax.set_xlabel("Reconstructed Z")
    ax.set_ylabel(f"Cells within {radius_um:.0f} of {anchor_class} anchor")
    ax.set_title(f"Cellular Z-density anchored on class '{anchor_class}'")
    ax.legend(fontsize=7, loc="best")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path = Path(out_path_template.format(anchor=str(anchor_class)))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name
