"""
Unbalanced Optimal Transport (UOT) for slice-to-slice cell coupling in Aether3D.

Fully GPU-accelerated implementation of the hybrid cost (spatial + gene + class) +
unbalanced Sinkhorn solver in PyTorch, with backward compatible NumPy CPU fallbacks.
"""

from __future__ import annotations

import warnings
from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

try:
    import ot
    _HAS_POT = True
except ImportError:
    _HAS_POT = False
    ot = None


def compute_hybrid_cost(
    x0: npt.NDArray[Any] | torch.Tensor,
    g0: npt.NDArray[Any] | torch.Tensor,
    c0: npt.NDArray[Any] | torch.Tensor,
    x1: npt.NDArray[Any] | torch.Tensor,
    g1: npt.NDArray[Any] | torch.Tensor,
    c1: npt.NDArray[Any] | torch.Tensor,
    alpha_spatial: float = 0.5,
    lambda_class: float = 10.0,
) -> npt.NDArray[Any] | torch.Tensor:
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
        # By contract all six inputs share a backend; isinstance narrows x0
        # only, so cast the rest to satisfy the all-Tensor PyTorch overload.
        return compute_hybrid_cost_pytorch(
            x0,
            cast(torch.Tensor, g0),
            cast(torch.Tensor, c0),
            cast(torch.Tensor, x1),
            cast(torch.Tensor, g1),
            cast(torch.Tensor, c1),
            alpha_spatial,
            lambda_class,
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
    return np.asarray(C)


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
    cost: npt.NDArray[Any] | torch.Tensor,
    reg: float = 0.8,
    tau: float = 0.05,
    n_samples: int = 50000,
    rng: np.random.Generator | None = None,
    torch_generator: torch.Generator | None = None,
) -> Tuple[
    npt.NDArray[Any] | torch.Tensor,
    npt.NDArray[Any] | torch.Tensor,
    npt.NDArray[Any] | torch.Tensor,
]:
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


def compute_uot_plan_pytorch(
    cost: torch.Tensor,
    reg: float = 0.8,
    tau: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Solve the unbalanced Sinkhorn UOT and return the *normalized* transport
    plan ``P`` (``P.sum() == 1``) of shape ``(n0, n1)``.

    Mathematical parity with POT's ``ot.sinkhorn_unbalanced`` on the normalized
    plan. The solve runs in the log domain (see below) so it stays finite in
    the small-``reg`` / large-cost regimes where the exp-domain recursion
    collapses.
    """
    n0, n1 = cost.shape
    if n0 == 0 or n1 == 0:
        raise ValueError(f"UOT coupling requires non-empty cost axes; got {tuple(cost.shape)}")
    device = cost.device
    dtype = cost.dtype

    # Log-domain stabilized unbalanced Sinkhorn.
    #
    # The exp-domain recursion u=(a/Kv)^fi with K=exp(-cost/reg)*(a*b) underflows
    # whenever -cost/reg is strongly negative (small reg, or the lambda_class=10
    # term added to a normalized cost): the kernel rounds to 0 and the plan
    # silently collapses to ~uniform. We instead keep the dual potentials in log
    # space and combine the log-kernel with torch.logsumexp, which is invariant
    # to a per-row/column constant shift and so never underflows. This replaces
    # the float64 promotion band-aid from issue #23 (issue #134).
    #
    # Mapping to the exp-domain form: with log_u=log(u), log_v=log(v) and
    # M = -cost/reg, log_a = log(a), log_b = log(b),
    #   K = exp(M) * (a * b),   u = (a / (K v))^fi
    # becomes the logsumexp updates below. The final transport plan
    # P = u * K * v is recovered as exp(log P); only the *normalized* plan is
    # used downstream (sampling + weights), so we subtract max(log P) before the
    # single exponentiation to keep it finite in float32.
    log_a = torch.full((n0,), -float(np.log(n0)), device=device, dtype=dtype)
    log_b = torch.full((n1,), -float(np.log(n1)), device=device, dtype=dtype)
    log_K = -cost / reg  # log of the cost kernel, excluding the a*b prefactor
    fi = tau / (tau + reg)

    log_u = torch.zeros(n0, device=device, dtype=dtype)
    log_v = torch.zeros(n1, device=device, dtype=dtype)

    for i in range(max_iter):
        log_u_prev = log_u.clone()
        log_v_prev = log_v.clone()

        # log(K v) = log_a_i + logsumexp_j(log_K_ij + log_b_j + log_v_j)
        log_u = -fi * torch.logsumexp(
            log_K + (log_b + log_v).unsqueeze(0), dim=1
        )
        # log(K^T u) = log_b_j + logsumexp_i(log_K_ij + log_a_i + log_u_i)
        log_v = -fi * torch.logsumexp(
            log_K + (log_a + log_u).unsqueeze(1), dim=0
        )

        # Converge check (absolute change in the log potentials)
        err_u = torch.max(torch.abs(log_u - log_u_prev))
        err_v = torch.max(torch.abs(log_v - log_v_prev))
        err = 0.5 * (err_u + err_v)
        if err < tol:
            break

    log_P = (
        log_u.unsqueeze(1)
        + log_K
        + log_a.unsqueeze(1)
        + log_b.unsqueeze(0)
        + log_v.unsqueeze(0)
    )
    # Shift before exponentiating: a global constant cancels under the
    # normalization below, so this only guards the final exp from underflow.
    log_P = log_P - log_P.max()
    P = torch.exp(log_P)

    total = P.sum()
    if total > 0:
        P = P / total
    else:  # pragma: no cover - the max-shift above keeps at least one entry == 1
        P = torch.ones_like(P) / P.numel()
    return P


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
    Solves the log-domain stabilized unbalanced Sinkhorn (mathematical parity
    with POT's sinkhorn_unbalanced) and samples ``n_samples`` cell pairs from
    the resulting normalized transport plan.
    """
    n1 = cost.shape[1]
    flat_P = compute_uot_plan_pytorch(cost, reg, tau, max_iter, tol).view(-1)

    idx = torch.multinomial(
        flat_P, num_samples=n_samples, replacement=True, generator=generator
    )
    src = torch.div(idx, n1, rounding_mode='floor')
    tgt = idx % n1
    weights = flat_P[idx]

    return src, tgt, weights.to(cost.dtype)
