"""
Unbalanced Optimal Transport (UOT) for slice-to-slice cell coupling in Aether3D.

Fully GPU-accelerated implementation of the hybrid cost (spatial + gene + class) +
unbalanced Sinkhorn solver in PyTorch, with backward compatible NumPy CPU fallbacks.
"""

from __future__ import annotations

import warnings
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
    x0: np.ndarray | torch.Tensor,
    g0: np.ndarray | torch.Tensor,
    c0: np.ndarray | torch.Tensor,
    x1: np.ndarray | torch.Tensor,
    g1: np.ndarray | torch.Tensor,
    c1: np.ndarray | torch.Tensor,
    alpha_spatial: float = 0.5,
    lambda_class: float = 10.0,
) -> np.ndarray | torch.Tensor:
    """
    Hybrid cost matrix for UOT between two slices.

    Cost = alpha_spatial * spatial_dist
         + (1 - alpha_spatial) * gene_cosine_dist
         + lambda_class * class_mismatch

    alpha_spatial ∈ [0, 1] trades off spatial vs gene; lambda_class scales the
    same-cell-type-prior penalty (default 10, retained for backward compat).

    Supports both NumPy arrays and PyTorch Tensors.
    """
    if isinstance(x0, torch.Tensor):
        return compute_hybrid_cost_pytorch(
            x0, g0, c0, x1, g1, c1, alpha_spatial, lambda_class
        )

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
    cost_class = np.clip(1.0 - np.dot(c0, c1.T), 0, 1) * lambda_class

    C = alpha_spatial * cost_spatial + (1 - alpha_spatial) * cost_gene + cost_class
    return C


def compute_hybrid_cost_pytorch(
    x0: torch.Tensor,
    g0: torch.Tensor,
    c0: torch.Tensor,
    x1: torch.Tensor,
    g1: torch.Tensor,
    c1: torch.Tensor,
    alpha_spatial: float = 0.5,
    lambda_class: float = 10.0,
) -> torch.Tensor:
    """
    Compute hybrid cost matrix on PyTorch (supporting GPU acceleration).
    """
    eps = 1e-9
    
    # Spatial (Euclidean) distance
    dist_spatial = torch.cdist(x0, x1, p=2)
    smax = dist_spatial.max()
    cost_spatial = dist_spatial / (smax + eps) if smax > 0 else dist_spatial

    # Gene (cosine) distance
    g0n = g0 / (torch.norm(g0, dim=1, keepdim=True) + eps)
    g1n = g1 / (torch.norm(g1, dim=1, keepdim=True) + eps)
    cost_gene = 1.0 - torch.matmul(g0n, g1n.T)
    gmax = cost_gene.max()
    cost_gene = cost_gene / (gmax + eps) if gmax > 0 else cost_gene

    # Class penalty (one-hot overlap)
    cost_class = torch.clamp(1.0 - torch.matmul(c0, c1.T), 0.0, 1.0) * lambda_class

    C = alpha_spatial * cost_spatial + (1.0 - alpha_spatial) * cost_gene + cost_class
    return C


def compute_uot_coupling(
    cost: np.ndarray | torch.Tensor,
    reg: float = 0.8,
    tau: float = 0.05,
    n_samples: int = 50000,
    rng: np.random.Generator | None = None,
    torch_generator: torch.Generator | None = None,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Unbalanced OT coupling using unbalanced Sinkhorn.
    Automatically routes to GPU solver if cost is a PyTorch tensor, otherwise uses POT on CPU.
    """
    if isinstance(cost, torch.Tensor):
        return compute_uot_coupling_pytorch(
            cost, reg, tau, n_samples, generator=torch_generator
        )

    n0, n1 = cost.shape
    if n0 == 0 or n1 == 0:
        raise ValueError(f"UOT coupling requires non-empty cost axes; got {cost.shape}")
    rng = rng or np.random.default_rng()

    if not _HAS_POT:
        warnings.warn(
            "POT is unavailable; using the PyTorch Sinkhorn UOT fallback instead "
            "of random source/target pairings.",
            RuntimeWarning,
            stacklevel=2,
        )
        generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        src_t, tgt_t, weights_t = compute_uot_coupling_pytorch(
            torch.as_tensor(cost, dtype=torch.float32),
            reg=reg,
            tau=tau,
            n_samples=n_samples,
            generator=generator,
        )
        return src_t.cpu().numpy(), tgt_t.cpu().numpy(), weights_t.cpu().numpy()

    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    P = ot.sinkhorn_unbalanced(a, b, cost, reg, tau)

    flat_P = P.ravel()
    flat_P = flat_P / flat_P.sum()
    idx = rng.choice(n0 * n1, size=n_samples, p=flat_P)

    src = idx // n1
    tgt = idx % n1
    weights = flat_P[idx]
    return src, tgt, weights


def compute_uot_coupling_pytorch(
    cost: torch.Tensor,
    reg: float = 0.8,
    tau: float = 0.05,
    n_samples: int = 50000,
    max_iter: int = 1000,
    tol: float = 1e-6,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unbalanced OT coupling solver on GPU/CPU in PyTorch.
    Mathematical parity with POT's sinkhorn_unbalanced.
    """
    n0, n1 = cost.shape
    if n0 == 0 or n1 == 0:
        raise ValueError(f"UOT coupling requires non-empty cost axes; got {tuple(cost.shape)}")
    device = cost.device
    dtype = cost.dtype
    solver_cost = cost
    if cost.dtype in (torch.float16, torch.bfloat16, torch.float32):
        underflow_fraction = torch.mean(((-cost / reg) < -80).float()).item()
        if underflow_fraction > 0:
            warnings.warn(
                "compute_uot_coupling_pytorch detected Sinkhorn kernel underflow risk; "
                "solving in float64 for numerical stability.",
                RuntimeWarning,
                stacklevel=2,
            )
            solver_cost = cost.to(torch.float64)
            dtype = solver_cost.dtype

    a_t = torch.ones(n0, device=device, dtype=dtype) / n0
    b_t = torch.ones(n1, device=device, dtype=dtype) / n1

    # In POT, K = exp(-cost / reg) * (a * b).  Promote underflow-prone
    # fp32 inputs to float64 above so lower-reg OT does not silently collapse.
    K = torch.exp(-solver_cost / reg) * (a_t.unsqueeze(1) * b_t.unsqueeze(0))

    u = torch.ones(n0, device=device, dtype=dtype)
    v = torch.ones(n1, device=device, dtype=dtype)
    fi = tau / (tau + reg)

    for i in range(max_iter):
        uprev = u.clone()
        vprev = v.clone()

        Kv = torch.matmul(K, v)
        u = (a_t / torch.clamp(Kv, min=1e-12)) ** fi
        Ktu = torch.matmul(K.T, u)
        v = (b_t / torch.clamp(Ktu, min=1e-12)) ** fi

        # Converge check
        err_u = torch.max(torch.abs(u - uprev)) / max(torch.max(torch.abs(u)), torch.max(torch.abs(uprev)), 1.0)
        err_v = torch.max(torch.abs(v - vprev)) / max(torch.max(torch.abs(v)), torch.max(torch.abs(vprev)), 1.0)
        err = 0.5 * (err_u + err_v)
        if err < tol:
            break

    P = u.unsqueeze(1) * K * v.unsqueeze(0)

    # Sample pairs
    flat_P = P.view(-1)
    flat_P_sum = flat_P.sum()
    if flat_P_sum > 0:
        flat_P = flat_P / flat_P_sum
    else:
        flat_P = torch.ones_like(flat_P) / flat_P.numel()

    idx = torch.multinomial(
        flat_P, num_samples=n_samples, replacement=True, generator=generator
    )
    src = torch.div(idx, n1, rounding_mode='floor')
    tgt = idx % n1
    weights = flat_P[idx]

    return src, tgt, weights.to(cost.dtype)
