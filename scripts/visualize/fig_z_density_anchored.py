"""
Wave 2 — Z-axis density of each cell class within a radius of an anchor class.

For a chosen anchor cell class, find all cells within `radius` µm of any anchor
cell, then compute density curves along the Z-axis for every cell class.
If the requested anchor class is not present, the first available class is
auto-selected and noted in the title.

Outputs:
  <out_dir>/z_density_anchored.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors


def render_z_density_anchored(
    volume: AnnData,
    out_path: Path,
    anchor_class: str = "Endothelial",
    radius: float = 20.0,
) -> None:
    """Render Z-density curves for each cell class within radius of anchor.

    Args:
        volume: Reconstructed AnnData volume with obsm['spatial_3d'].
        out_path: Output path for the PNG figure.
        anchor_class: Cell class to use as spatial anchor (default: "Endothelial").
            If not present in the data, the first available class is used.
        radius: Search radius in µm around each anchor cell (default: 20.0).
    """
    if "spatial_3d" not in volume.obsm:
        return
    classes = class_from_onehot(volume)
    if classes is None:
        return
    coords = np.asarray(volume.obsm["spatial_3d"])
    unique = sorted(np.unique(classes).tolist())

    # Auto-select anchor if requested class is missing
    actual_anchor = anchor_class
    if anchor_class not in unique:
        if not unique:
            return
        actual_anchor = unique[0]

    anchor_mask = classes == actual_anchor
    if anchor_mask.sum() == 0:
        return

    # Build KDTree and query all cells within radius of ANY anchor cell
    tree = cKDTree(coords)
    anchor_coords = coords[anchor_mask]
    # query_ball_point returns list of arrays; union them
    neighbor_sets = tree.query_ball_point(anchor_coords, r=radius)
    all_neighbor_indices = np.unique(np.concatenate(neighbor_sets))

    if len(all_neighbor_indices) == 0:
        return

    # Compute Z density for each cell class on the neighbor subset
    z = coords[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    n_z_bins = 24
    bins = np.linspace(z_min, z_max, n_z_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    densities: dict[str, np.ndarray] = {}
    for c in unique:
        densities[c] = np.zeros(n_z_bins, dtype=np.float64)

    for ni in all_neighbor_indices:
        c = str(classes[ni])
        b = min(int(np.searchsorted(bins, z[ni], side="right") - 1), n_z_bins - 1)
        if 0 <= b < n_z_bins:
            densities[c][b] += 1.0

    # Smooth with gaussian kernel
    for c in densities:
        densities[c] = gaussian_filter1d(densities[c], sigma=1.0)

    palette = stable_categorical_colors(np.array(unique))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c in unique:
        ax.plot(
            centers, densities[c], label=str(c), color=palette[str(c)], linewidth=1.5
        )
    ax.set_xlabel("Reconstructed Z")
    ax.set_ylabel(f"Cells within {radius:.0f} µm of {actual_anchor} anchor")
    title = f"Cellular Z-density anchored on class '{actual_anchor}'"
    if actual_anchor != anchor_class:
        title += (
            f"\n(requested '{anchor_class}' not found; auto-selected '{actual_anchor}')"
        )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
