"""3D graph topology metrics for vascular-network / connectivity analysis.

Implements 3D-graph topology metrics for vascular-network connectivity
analysis. Metric helpers are pure graph-theory functions on a
``(coords_3d, edges)`` input and deterministic on synthetic graphs.

All functions are NumPy-only; no graph library dependency. An edge list
is expected as ``edges: Iterable[(u, v)]`` with integer node indices; we
treat the graph as undirected and dedupe ``(u, v) == (v, u)`` internally.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt


__all__ = [
    "degree_histogram",
    "edge_length_distribution",
    "branch_point_count",
    "hub_centrality",
]


def _normalize_edges(edges: Iterable[tuple[int, int]], n_nodes: int) -> npt.NDArray[np.int64]:
    """Return an (E, 2) sorted-pair edge array, deduped and self-loop-free."""
    pairs = set()
    for u, v in edges:
        if u == v:
            continue
        if not (0 <= u < n_nodes) or not (0 <= v < n_nodes):
            raise ValueError(f"edge ({u}, {v}) out of range for n_nodes={n_nodes}")
        pairs.add((min(u, v), max(u, v)))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def degree_histogram(
    n_nodes: int,
    edges: Iterable[tuple[int, int]],
    max_degree: int | None = None,
) -> dict[str, Any]:
    """Per-degree count of nodes; isolated nodes carry degree 0.

    Args:
        n_nodes: number of graph nodes.
        edges: iterable of ``(u, v)`` pairs (undirected; self-loops dropped).
        max_degree: clip the histogram at this degree (counts above this
            cap collapse into the final bin). If None, use observed maximum.

    Returns:
        ``{"degrees": (D,) int array, "counts": (D,) int array,
           "mean_degree", "median_degree", "n_isolated"}``.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
    e = _normalize_edges(edges, n_nodes)
    deg = np.zeros(n_nodes, dtype=np.int64)
    if e.size:
        np.add.at(deg, e[:, 0], 1)
        np.add.at(deg, e[:, 1], 1)
    cap = int(deg.max()) if max_degree is None else int(max_degree)
    clipped = np.clip(deg, 0, cap)
    counts = np.bincount(clipped, minlength=cap + 1)
    return {
        "degrees": np.arange(cap + 1, dtype=np.int64),
        "counts": counts,
        "mean_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
        "n_isolated": int((deg == 0).sum()),
    }


def edge_length_distribution(
    coords_3d: npt.NDArray[np.floating[Any]],
    edges: Iterable[tuple[int, int]],
    n_bins: int = 30,
) -> dict[str, Any]:
    """Histogram of Euclidean edge lengths for a 3D graph.

    Args:
        coords_3d: (N, 3) node coordinates.
        edges: iterable of ``(u, v)`` pairs.
        n_bins: number of histogram bins (>= 1).

    Returns:
        ``{"lengths", "bin_edges", "counts", "mean", "median", "min", "max"}``.
        ``lengths`` is the raw (E,) array; the histogram fields are bin-level.
    """
    c = np.asarray(coords_3d, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords_3d must be (N, 3), got {c.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    e = _normalize_edges(edges, c.shape[0])
    if e.shape[0] == 0:
        return {
            "lengths": np.zeros(0, dtype=np.float64),
            "bin_edges": np.zeros(n_bins + 1, dtype=np.float64),
            "counts": np.zeros(n_bins, dtype=np.int64),
            "mean": float("nan"), "median": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    lengths = np.linalg.norm(c[e[:, 0]] - c[e[:, 1]], axis=1)
    counts, bin_edges = np.histogram(lengths, bins=n_bins)
    return {
        "lengths": lengths,
        "bin_edges": bin_edges,
        "counts": counts.astype(np.int64),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
    }


def branch_point_count(
    n_nodes: int,
    edges: Iterable[tuple[int, int]],
    branch_degree: int = 3,
) -> dict[str, Any]:
    """Count nodes of degree >= branch_degree (branch points) and degree == 1
    (tip / endpoint) on an undirected graph.

    Args:
        n_nodes: number of graph nodes.
        edges: iterable of ``(u, v)`` pairs.
        branch_degree: minimum degree to count as a branch point (default 3).

    Returns:
        ``{"n_branch_points", "n_endpoints", "branch_node_ids", "endpoint_node_ids"}``.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
    if branch_degree < 2:
        raise ValueError(f"branch_degree must be >= 2, got {branch_degree}")
    e = _normalize_edges(edges, n_nodes)
    deg = np.zeros(n_nodes, dtype=np.int64)
    if e.size:
        np.add.at(deg, e[:, 0], 1)
        np.add.at(deg, e[:, 1], 1)
    branch_mask = deg >= branch_degree
    endpoint_mask = deg == 1
    return {
        "n_branch_points": int(branch_mask.sum()),
        "n_endpoints": int(endpoint_mask.sum()),
        "branch_node_ids": np.where(branch_mask)[0],
        "endpoint_node_ids": np.where(endpoint_mask)[0],
    }


def hub_centrality(
    n_nodes: int,
    edges: Iterable[tuple[int, int]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Identify the top-k highest-degree nodes (hubs) plus per-node degree
    centrality (degree / (n_nodes - 1)).

    This is a deliberately simple centrality definition; the goal here is
    to give downstream composers a hub overlay without pulling in a graph
    library. Betweenness or eigenvector centrality is out of scope and is
    deferred until a real vascular volume justifies the dependency.

    Args:
        n_nodes: number of graph nodes.
        edges: iterable of ``(u, v)`` pairs.
        top_k: number of top-degree nodes to surface.

    Returns:
        ``{"degree_centrality": (N,) float, "top_hub_ids": (K,) int,
           "top_hub_degrees": (K,) int}``.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be > 0, got {n_nodes}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    e = _normalize_edges(edges, n_nodes)
    deg = np.zeros(n_nodes, dtype=np.int64)
    if e.size:
        np.add.at(deg, e[:, 0], 1)
        np.add.at(deg, e[:, 1], 1)
    denom = max(1, n_nodes - 1)
    centrality = deg.astype(np.float64) / denom
    k = min(top_k, n_nodes)
    top_idx = np.argsort(deg)[-k:][::-1]
    return {
        "degree_centrality": centrality,
        "top_hub_ids": top_idx.astype(np.int64),
        "top_hub_degrees": deg[top_idx].astype(np.int64),
    }
