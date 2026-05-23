"""
Wave 1 — 6-row Z-strata scatter grid.

For six evenly-spaced Z values across the reconstructed thickness, render a 2D
XY scatter of virtual cells within a thin band around that Z, coloured by cell
class. When input slices are provided, the figure includes a second row showing
the nearest input slice at each Z for direct visual comparison.

Outputs:
  <out_dir>/multi_z_slice_grid.png
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors


def render_multi_z_slice_grid(
    volume: AnnData, input_slices: Optional[List[AnnData]], out_path: Path, n_z: int = 6
) -> None:
    coords = np.asarray(volume.obsm["spatial_3d"])
    z = coords[:, 2]
    z_targets = np.linspace(z.min(), z.max(), n_z)
    band = (z.max() - z.min()) * 0.05

    classes = class_from_onehot(volume)
    if classes is None:
        classes = np.array(["all"] * volume.n_obs)
    palette = stable_categorical_colors(np.array(classes))

    n_rows = 2 if input_slices else 1
    fig, axes = plt.subplots(n_rows, n_z, figsize=(2.6 * n_z, 2.8 * n_rows), squeeze=False)

    for ci, t in enumerate(z_targets):
        mask = (z >= t - band) & (z <= t + band)
        ax = axes[0, ci]
        for c in np.unique(classes):
            m = mask & (classes == c)
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], c=palette[c], s=2)
        ax.set_title(f"Reconstructed\nZ={t:.1f}  (n={int(mask.sum())})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    if input_slices:
        slice_zs = np.array([float(s.obs["z_coord"].iloc[0]) if "z_coord" in s.obs else i * 10.0
                             for i, s in enumerate(input_slices)])
        for ci, t in enumerate(z_targets):
            nearest = int(np.argmin(np.abs(slice_zs - t)))
            s = input_slices[nearest]
            xy = np.asarray(s.obsm["spatial"])[:, :2]
            cls = s.obs["cell_class"].astype(str).to_numpy() if "cell_class" in s.obs else np.array(["?"] * s.n_obs)
            in_palette = stable_categorical_colors(cls)
            ax = axes[1, ci]
            for c in np.unique(cls):
                m = (cls == c)
                ax.scatter(xy[m, 0], xy[m, 1], c=in_palette.get(c, "#888"), s=2)
            ax.set_title(f"Input slice {nearest}\nZ={slice_zs[nearest]:.1f}  (n={s.n_obs})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Z-strata scatter grid — reconstructed vs nearest input slice", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
