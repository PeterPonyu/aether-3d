"""Domain spatial-coherence metrics: CHAOS and PAS (Dong 2025, field-standard).

These two scores quantify how spatially coherent a set of *domain labels* is
over a tissue point cloud — independent of any ground-truth labelling. They are
the field-standard spatial-domain quality scores used in the SpatialDLPFC /
Dong 2025 benchmarks, complementing the geometry + molecular metrics already in
``metrics.py``.

- **CHAOS** (Spatial Chaos score): the mean, over all cells, of the Euclidean
  distance to the nearest *same-label* neighbour, averaged across labels. A
  spatially coherent labelling places same-label cells close together, so a
  LOWER CHAOS is better. Coordinates are min-max normalised per-axis so the
  score is scale-free.

- **PAS** (Percentage of Abnormal Spots): the fraction of cells whose own label
  disagrees with the majority label among its ``k`` nearest spatial neighbours.
  A coherent labelling has few abnormal spots, so a LOWER PAS is better.

Both gracefully return NaN on degenerate input (empty / single-cluster / too
few cells) so a reviewer can tell missing data apart from a real value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["chaos_score", "pas_score"]


def _normalize_coords(coords: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    c = np.asarray(coords, dtype=np.float64)
    span = c.max(axis=0) - c.min(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    normalized: npt.NDArray[np.float64] = (c - c.min(axis=0)) / span
    return normalized


def chaos_score(
    coords: npt.NDArray[np.floating[Any]],
    labels: Sequence[Any],
) -> float:
    """Spatial Chaos score of a domain labelling (lower = more coherent).

    For each label, the mean distance from every cell of that label to its
    nearest same-label neighbour is computed on min-max normalised coordinates;
    the per-label means are then averaged. Returns NaN when there are fewer than
    two cells or fewer than one usable label.
    """
    c = np.asarray(coords, dtype=np.float64)
    lab = np.asarray(labels)
    if c.ndim != 2 or c.shape[0] != lab.shape[0]:
        raise ValueError(f"coords {c.shape} incompatible with labels {lab.shape}")
    if c.shape[0] < 2:
        return float("nan")

    c = _normalize_coords(c)
    try:
        from scipy.spatial import cKDTree

        def _nn_same_label(sub: npt.NDArray[np.float64]) -> float:
            if sub.shape[0] < 2:
                return float("nan")
            d, _ = cKDTree(sub).query(sub, k=2)
            return float(np.mean(d[:, 1]))
    except ImportError:  # pragma: no cover - scipy present in env dl

        def _nn_same_label(sub: npt.NDArray[np.float64]) -> float:
            if sub.shape[0] < 2:
                return float("nan")
            diff = sub[:, None, :] - sub[None, :, :]
            sq = (diff * diff).sum(axis=-1)
            np.fill_diagonal(sq, np.inf)
            return float(np.mean(np.sqrt(sq.min(axis=1))))

    per_label: list[float] = []
    for lbl in np.unique(lab):
        sub = c[lab == lbl]
        val = _nn_same_label(sub)
        if not np.isnan(val):
            per_label.append(val)
    if not per_label:
        return float("nan")
    return float(np.mean(per_label))


def pas_score(
    coords: npt.NDArray[np.floating[Any]],
    labels: Sequence[Any],
    k: int = 10,
) -> float:
    """Percentage of Abnormal Spots (lower = more coherent).

    A cell is "abnormal" when its own label differs from the most common label
    among its ``k`` nearest spatial neighbours (self excluded). Returns the
    fraction of abnormal cells, or NaN when there are too few cells.
    """
    c = np.asarray(coords, dtype=np.float64)
    lab = np.asarray(labels)
    if c.ndim != 2 or c.shape[0] != lab.shape[0]:
        raise ValueError(f"coords {c.shape} incompatible with labels {lab.shape}")
    n = c.shape[0]
    if n < 2:
        return float("nan")
    k_use = min(k, n - 1)

    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(c).query(c, k=k_use + 1)
        nbr_idx = np.asarray(idx[:, 1:], dtype=np.int64)
    except ImportError:  # pragma: no cover - scipy present in env dl
        diff = c[:, None, :] - c[None, :, :]
        sq = (diff * diff).sum(axis=-1)
        np.fill_diagonal(sq, np.inf)
        nbr_idx = np.argpartition(sq, k_use, axis=1)[:, :k_use]

    abnormal = 0
    for i in range(n):
        nbr_labels = lab[nbr_idx[i]]
        vals, counts = np.unique(nbr_labels, return_counts=True)
        majority = vals[int(np.argmax(counts))]
        if majority != lab[i]:
            abnormal += 1
    return float(abnormal / n)
