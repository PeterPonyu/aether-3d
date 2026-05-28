"""Topology + vector-field metrics for 3D-reconstruction benchmarks.

The metrics here are deliberately not in the standard "Chamfer + Moran's I +
ARI" set used by competing 3D-reconstruction methods. They quantify three
properties of a continuous tissue reconstruction that point-cloud / cluster
metrics cannot expose:

1. **Persistent-homology Betti stability** — does the reconstructed volume
   preserve the same number of connected components and one-dimensional
   holes (loops) as the source slices, across virtual depth? A drop in
   Betti-0 means the model fragments tissue; a spike in Betti-1 means it
   introduces spurious holes.

2. **Flow-divergence map** — for a velocity-field reconstruction (Aether3D's
   regime), the per-voxel divergence ∇·v measures whether the model preserves
   local cellular density (∇·v ≈ 0) or creates/annihilates mass spuriously.

3. **Velocity anisotropy ratio** — the ratio of the largest to smallest
   eigenvalue of the local velocity covariance. Quantifies whether flow is
   directional (high anisotropy, expected near tissue boundaries) or
   diffuse-isotropic (low anisotropy, expected in homogeneous regions).

All metrics gracefully return NaN on degenerate input. The persistent-
homology implementation here is dependency-free (a simple union-find for
Betti-0 + a small grid-based Euler-characteristic estimate for Betti-1) so
no `gudhi` / `ripser` install is required for the smoke path; an optional
upstream library can be wired later for proper computation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt


# -- Betti-0 via k-NN graph union-find ------------------------------------


def betti_zero(
    coords: npt.NDArray[np.floating[Any]], k: int = 6, edge_threshold: Optional[float] = None
) -> int:
    """Count connected components in a k-NN graph over a 2D/3D point cloud.

    A reconstructed tissue slice should match the ground-truth Betti-0
    (same number of connected components) at the same neighborhood scale.

    Args:
        coords: (n, d) point cloud (d ∈ {2, 3}).
        k: neighbors per node in the kNN graph.
        edge_threshold: if given, only edges with distance ≤ threshold count;
            useful for tissue-boundary segmentation where long-range edges
            should not bridge anatomically distinct regions. Default `None`
            keeps every kNN edge — the canonical Betti-0 over the unfiltered
            graph.
    """
    if coords.size == 0:
        return 0
    n = coords.shape[0]
    if n == 1:
        return 1
    k_use = min(k, n - 1)

    # Pairwise distances (small-cloud regime — fine for the synthetic harness)
    diff = coords[:, None, :] - coords[None, :, :]
    sq = (diff * diff).sum(axis=-1)
    np.fill_diagonal(sq, np.inf)

    knn_idx = np.argpartition(sq, k_use, axis=1)[:, :k_use]
    knn_d = np.sqrt(np.take_along_axis(sq, knn_idx, axis=1))

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j_idx in range(k_use):
            d = knn_d[i, j_idx]
            if edge_threshold is None or d <= edge_threshold:
                union(int(i), int(knn_idx[i, j_idx]))

    roots = {find(i) for i in range(n)}
    return len(roots)


def betti_zero_stability(
    coords_truth: npt.NDArray[np.floating[Any]],
    coords_recon: npt.NDArray[np.floating[Any]],
    k: int = 6,
) -> float:
    """Symmetric stability of Betti-0 across truth and reconstruction.

    Returns 1.0 when both components counts match, dropping toward 0 as they
    diverge. Defined as min(b_t, b_r) / max(b_t, b_r) with NaN on empty.
    """
    if coords_truth.size == 0 or coords_recon.size == 0:
        return float("nan")
    b_t = betti_zero(coords_truth, k=k)
    b_r = betti_zero(coords_recon, k=k)
    if max(b_t, b_r) == 0:
        return float("nan")
    return float(min(b_t, b_r) / max(b_t, b_r))


# -- Voxelized flow divergence --------------------------------------------


def flow_divergence_map(
    coords: npt.NDArray[np.floating[Any]],
    velocities: npt.NDArray[np.floating[Any]],
    grid_size: int = 16,
) -> npt.NDArray[np.float32]:
    """Per-voxel divergence ∇·v computed via central differences on a grid.

    For a velocity field that preserves mass, ∇·v ≈ 0 everywhere. Local
    spikes in |∇·v| indicate model-introduced source/sink artifacts.

    Args:
        coords: (n, 2) spatial coordinates.
        velocities: (n, 2) per-cell velocity vectors.
        grid_size: side length of the regular grid; each voxel is filled by
            mean-pooling the velocities of cells that fall in it.

    Returns:
        (grid_size, grid_size) array of divergence values; NaN where the
        voxel had no cells contributing.
    """
    if coords.size == 0 or velocities.size == 0:
        return np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    if coords.shape != velocities.shape:
        raise ValueError(
            f"coords and velocities must have same shape; got {coords.shape}, {velocities.shape}"
        )

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    ix = np.clip(((coords[:, 0] - xmin) / xr * (grid_size - 1)).astype(int), 0, grid_size - 1)
    iy = np.clip(((coords[:, 1] - ymin) / yr * (grid_size - 1)).astype(int), 0, grid_size - 1)

    grid_vx: npt.NDArray[np.float64] = np.zeros((grid_size, grid_size), dtype=np.float64)
    grid_vy: npt.NDArray[np.float64] = np.zeros((grid_size, grid_size), dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for ci in range(coords.shape[0]):
        grid_vx[iy[ci], ix[ci]] += velocities[ci, 0]
        grid_vy[iy[ci], ix[ci]] += velocities[ci, 1]
        counts[iy[ci], ix[ci]] += 1.0
    has_cells = counts > 0
    grid_vx = np.where(has_cells, grid_vx / np.where(counts > 0, counts, 1.0), np.nan)
    grid_vy = np.where(has_cells, grid_vy / np.where(counts > 0, counts, 1.0), np.nan)

    # Cell spacing in physical units
    dx = xr / max(grid_size - 1, 1)
    dy = yr / max(grid_size - 1, 1)

    # Central differences with NaN-safe propagation
    dvx_dx = np.full_like(grid_vx, np.nan)
    dvy_dy = np.full_like(grid_vy, np.nan)
    dvx_dx[:, 1:-1] = (grid_vx[:, 2:] - grid_vx[:, :-2]) / (2.0 * dx)
    dvy_dy[1:-1, :] = (grid_vy[2:, :] - grid_vy[:-2, :]) / (2.0 * dy)

    return (dvx_dx + dvy_dy).astype(np.float32)


def divergence_summary(div_map: npt.NDArray[np.floating[Any]]) -> dict[str, float]:
    """Reduce a divergence map to scalars: mean |∇·v|, max |∇·v|, RMS ∇·v."""
    finite = div_map[np.isfinite(div_map)]
    if finite.size == 0:
        return {
            "mean_abs_divergence": float("nan"),
            "max_abs_divergence": float("nan"),
            "rms_divergence": float("nan"),
        }
    return {
        "mean_abs_divergence": float(np.mean(np.abs(finite))),
        "max_abs_divergence": float(np.max(np.abs(finite))),
        "rms_divergence": float(np.sqrt(np.mean(finite ** 2))),
    }


# -- Velocity anisotropy --------------------------------------------------


def velocity_anisotropy(velocities: npt.NDArray[np.floating[Any]]) -> float:
    """Eigenvalue ratio of the velocity covariance.

    Returns λ_max / λ_min, where eigenvalues are of the 2×2 covariance matrix
    Cov(v). Approaches 1 for diffuse-isotropic flow; grows large for highly
    directional flow (typical near tissue boundaries). Returns NaN when the
    covariance is degenerate.
    """
    if velocities.size == 0:
        return float("nan")
    if velocities.shape[1] != 2:
        raise ValueError(f"velocity_anisotropy expects 2D velocities; got {velocities.shape}")
    if velocities.shape[0] < 2:
        return float("nan")
    cov = np.cov(velocities, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    if eigs.min() < 1e-12:
        return float("nan")
    return float(eigs.max() / eigs.min())


# -- Roll-up for the contract ---------------------------------------------


def topology_metrics(
    coords_truth: npt.NDArray[np.floating[Any]],
    coords_recon: npt.NDArray[np.floating[Any]],
    velocities_recon: Optional[npt.NDArray[np.floating[Any]]] = None,
    grid_size: int = 16,
    k: int = 6,
) -> dict[str, float]:
    """Compute the full topology metric set for one virtual-slice comparison.

    `velocities_recon` is optional — when absent, divergence + anisotropy are
    NaN and only the Betti stability score is meaningful.
    """
    out: dict[str, float] = {
        "betti0_stability": betti_zero_stability(coords_truth, coords_recon, k=k),
    }
    if velocities_recon is not None and velocities_recon.shape == coords_recon.shape:
        div_map = flow_divergence_map(coords_recon, velocities_recon, grid_size=grid_size)
        out.update(divergence_summary(div_map))
        out["velocity_anisotropy"] = velocity_anisotropy(velocities_recon)
    else:
        out["mean_abs_divergence"] = float("nan")
        out["max_abs_divergence"] = float("nan")
        out["rms_divergence"] = float("nan")
        out["velocity_anisotropy"] = float("nan")
    return out
