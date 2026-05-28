"""Shared plotting helpers for the Aether3D biology figure pack."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData


CATEGORICAL_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def stable_categorical_colors(values) -> Dict[str, str]:
    cats = pd.Categorical(values).categories.tolist()
    return {str(c): CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, c in enumerate(cats)}


def select_markers_by_group(
    adata: AnnData, group_key: str, n_per_group: int = 1
) -> Dict[str, List[str]]:
    X = to_dense(adata.X)
    gene_names = list(adata.var_names)
    if group_key not in adata.obs:
        return {}
    groups = adata.obs[group_key].astype(str)
    out: Dict[str, List[str]] = {}
    overall_mean = X.mean(axis=0)
    for g in groups.unique():
        mask = (groups == g).to_numpy()
        if mask.sum() < 2:
            continue
        group_mean = X[mask].mean(axis=0)
        score = group_mean - overall_mean
        idx = np.argsort(score)[-n_per_group:][::-1]
        out[g] = [gene_names[i] for i in idx]
    return out


def class_from_onehot(adata: AnnData) -> Optional[np.ndarray]:
    """Resolve a categorical cell-class array from either obs['cell_class']
    or obsm['cell_class_vel'] (one-hot vector predicted by the model)."""
    if "cell_class" in adata.obs:
        return adata.obs["cell_class"].astype(str).to_numpy()
    if "cell_class_vel" in adata.obsm:
        arr = np.asarray(adata.obsm["cell_class_vel"])
        idx = arr.argmax(axis=1)
        return np.array([f"C{int(i)}" for i in idx])
    return None
