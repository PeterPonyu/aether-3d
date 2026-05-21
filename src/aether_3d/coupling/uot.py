"""
Unbalanced Optimal Transport (UOT) for slice-to-slice cell coupling in Aether3D.

Clean, retyped implementation of the hybrid cost (spatial + gene + class) + 
unbalanced Sinkhorn that was in the original DeepSpatial uot_solver.py.

No original code copied — only the mathematical idea is preserved.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

try:
    import ot
    _HAS_POT = True
except ImportError:
    _HAS_POT = False
    ot = None


def compute_hybrid_cost(
    x0: np.ndarray,
    g0: np.ndarray,
    c0: np.ndarray,
    x1: np.ndarray,
    g1: np.ndarray,
    c1: np.ndarray,
    alpha_spatial: float = 0.5,
) -> np.ndarray:
    """
    Hybrid cost matrix for UOT between two slices.

    Cost = alpha * spatial_dist + (1-alpha) * gene_cosine_dist + 10 * class_mismatch
    """
    eps = 1e-9

    # Spatial (Euclidean) - with fallback if POT not present
    if _HAS_POT:
        cost_spatial = ot.dist(x0, x1, metric="euclidean")
    else:
        cost_spatial = np.sqrt(((x0[:, None] - x1[None, :])**2).sum(axis=2))
    smax = cost_spatial.max()
    cost_spatial = cost_spatial / (smax + eps) if smax > 0 else cost_spatial

    # Gene (cosine) - with fallback
    if _HAS_POT:
        cost_gene = ot.dist(g0, g1, metric="cosine")
    else:
        # Simple cosine distance
        g0n = g0 / (np.linalg.norm(g0, axis=1, keepdims=True) + eps)
        g1n = g1 / (np.linalg.norm(g1, axis=1, keepdims=True) + eps)
        cost_gene = 1 - g0n @ g1n.T
    gmax = cost_gene.max()
    cost_gene = cost_gene / (gmax + eps) if gmax > 0 else cost_gene

    # Class penalty (one-hot)
    cost_class = np.clip(1.0 - np.dot(c0, c1.T), 0, 1) * 10.0

    C = alpha_spatial * cost_spatial + (1 - alpha_spatial) * cost_gene + cost_class
    return C


def compute_uot_coupling(
    cost: np.ndarray,
    reg: float = 0.8,
    tau: float = 0.05,
    n_samples: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unbalanced OT coupling using POT's unbalanced Sinkhorn.
    Falls back to uniform random pairing if POT is not installed.
    """
    n0, n1 = cost.shape

    if not _HAS_POT:
        # Fallback for environments without POT
        src = np.random.randint(0, n0, n_samples)
        tgt = np.random.randint(0, n1, n_samples)
        weights = np.ones(n_samples) / n_samples
        return src, tgt, weights

    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    P = ot.sinkhorn_unbalanced(a, b, cost, reg, tau)

    flat_P = P.ravel()
    flat_P = flat_P / flat_P.sum()
    idx = np.random.choice(n0 * n1, size=n_samples, p=flat_P)

    src = idx // n1
    tgt = idx % n1
    weights = flat_P[idx]
    return src, tgt, weights
