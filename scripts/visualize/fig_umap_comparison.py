"""
Wave 3 — UMAP comparison: input stack vs reconstructed volume.

Concatenates input cells and reconstructed virtual cells in a single embedding
space (top HVGs shared between them), runs PCA → UMAP, then renders two
panels coloured by (a) source = input vs reconstructed and (b) class label
when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from scripts.visualize._plot_utils import class_from_onehot, stable_categorical_colors, to_dense


def render_umap_comparison(
    volume: AnnData, input_slices: List[AnnData], out_path: Path, max_per_source: int = 3000
) -> None:
    if not input_slices:
        return
    common = [g for g in volume.var_names if g in input_slices[0].var_names]
    if len(common) < 5:
        return

    rng = np.random.default_rng(0)

    # Volume sample
    v_idx_full = np.arange(volume.n_obs)
    if volume.n_obs > max_per_source:
        v_idx_full = rng.choice(volume.n_obs, max_per_source, replace=False)
    rec_X = to_dense(volume.X)[v_idx_full][:, [list(volume.var_names).index(g) for g in common]]
    rec_cls = class_from_onehot(volume)
    rec_cls = rec_cls[v_idx_full] if rec_cls is not None else np.array(["?"] * len(v_idx_full))

    # Input stack sample
    in_X_list, in_cls_list = [], []
    per_slice = max(50, max_per_source // max(len(input_slices), 1))
    for s in input_slices:
        n = min(per_slice, s.n_obs)
        idx = rng.choice(s.n_obs, n, replace=False)
        X = to_dense(s.X)[idx][:, [list(s.var_names).index(g) for g in common if g in s.var_names]]
        in_X_list.append(X)
        cls = s.obs["cell_class"].astype(str).to_numpy()[idx] if "cell_class" in s.obs else np.array(["?"] * n)
        in_cls_list.append(cls)
    in_X = np.concatenate(in_X_list, axis=0)
    in_cls = np.concatenate(in_cls_list)

    combined_X = np.concatenate([in_X, rec_X], axis=0).astype(np.float32)
    source = np.array(["input"] * in_X.shape[0] + ["reconstructed"] * rec_X.shape[0])
    classes = np.concatenate([in_cls, rec_cls])

    a = AnnData(X=combined_X)
    a.obs["source"] = source
    a.obs["class"] = classes
    sc.pp.pca(a, n_comps=min(30, combined_X.shape[1] - 1, combined_X.shape[0] - 1))
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15, key_added="_cmp")
    sc.tl.umap(a, neighbors_key="_cmp")
    coords = np.asarray(a.obsm["X_umap"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, color_key in zip(axes, ["source", "class"]):
        cats = a.obs[color_key].astype(str)
        palette = stable_categorical_colors(cats)
        for c in cats.unique():
            m = (cats == c).to_numpy()
            ax.scatter(coords[m, 0], coords[m, 1], s=3, color=palette[c], label=str(c))
        ax.set_title(f"UMAP — colour: {color_key}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if len(cats.unique()) <= 8:
            ax.legend(fontsize=7, markerscale=1.5, loc="best", frameon=False)
    fig.suptitle("UMAP: input stack vs Aether3D-reconstructed (joint embedding)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
