"""
Wave 1 — per-gene Moran's I on reconstructed volume vs input stack.

Spatial weights: 1/(1 + d) over k nearest neighbours in the 3D coordinates.
Inline numpy implementation; no PySAL dependency.

Outputs:
  <out_dir>/morans_i_scatter.png  (scatter input vs reconstructed Moran's I, with Pearson R)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from scripts.visualize._plot_utils import to_dense


def morans_i_per_gene(coords: np.ndarray, X: np.ndarray, k: int = 8) -> np.ndarray:
    n, g = X.shape
    if n < k + 1:
        return np.full(g, np.nan)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists, idxs = nn.kneighbors(coords)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    weights = 1.0 / (1.0 + dists)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = weights / row_sums  # row-normalized

    Xc = X - X.mean(axis=0, keepdims=True)
    var = (Xc ** 2).mean(axis=0)
    safe = var > 1e-12
    out = np.full(g, np.nan)
    if not safe.any():
        return out

    neighbor_vals = Xc[idxs]  # (n, k, g)
    weighted = (neighbor_vals * W[..., None]).sum(axis=1)  # (n, g)
    numerator = (Xc * weighted).sum(axis=0)
    denominator = (Xc ** 2).sum(axis=0)
    denom_safe = denominator > 1e-12
    ok = safe & denom_safe
    out[ok] = numerator[ok] / denominator[ok]
    return out


def render_morans_i_scatter(
    volume: AnnData, input_slices: List[AnnData], out_path: Path, max_genes: int = 200, k: int = 8
) -> None:
    if not input_slices:
        return
    rec_coords = np.asarray(volume.obsm["spatial_3d"])
    rec_X = to_dense(volume.X).astype(np.float32)

    in_coords_list, in_X_list = [], []
    for s in input_slices:
        if "spatial" not in s.obsm:
            continue
        xy = np.asarray(s.obsm["spatial"])[:, :2]
        z = np.asarray(s.obs.get("z_coord", 0.0), dtype=float).reshape(-1, 1) if "z_coord" in s.obs else np.zeros((xy.shape[0], 1))
        in_coords_list.append(np.hstack([xy, z]))
        in_X_list.append(to_dense(s.X).astype(np.float32))
    if not in_coords_list:
        return
    in_coords = np.concatenate(in_coords_list, axis=0)
    in_X = np.concatenate(in_X_list, axis=0)

    # Restrict to common gene set
    common = [g for g in volume.var_names if g in input_slices[0].var_names]
    if not common:
        return
    if len(common) > max_genes:
        rng = np.random.default_rng(42)
        common = list(rng.choice(common, max_genes, replace=False))

    rec_idx = [list(volume.var_names).index(g) for g in common]
    in_idx = [list(input_slices[0].var_names).index(g) for g in common]

    I_rec = morans_i_per_gene(rec_coords, rec_X[:, rec_idx], k=k)
    I_in = morans_i_per_gene(in_coords, in_X[:, in_idx], k=k)

    ok = ~(np.isnan(I_rec) | np.isnan(I_in))
    if ok.sum() < 3:
        return
    r, _ = pearsonr(I_rec[ok], I_in[ok])

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.scatter(I_in[ok], I_rec[ok], s=8, color="#4C72B0", alpha=0.7)
    lim_min = float(np.nanmin([I_in[ok].min(), I_rec[ok].min()]))
    lim_max = float(np.nanmax([I_in[ok].max(), I_rec[ok].max()]))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#888", linewidth=0.8)
    ax.set_xlabel("Moran's I — input stack")
    ax.set_ylabel("Moran's I — Aether3D reconstruction")
    ax.set_title(f"Per-gene Moran's I (n={int(ok.sum())} genes, R={r:.3f})")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
