"""Maximal common spatial region across multi-section slices.

Computes the maximal common spatial region across multi-section slices.
When sections of a 3D volume have different lateral footprints, downstream
metrics need to restrict comparisons to the spatial region covered by
*all* sections; this helper computes that mask.

The implementation is pure NumPy: compute each section's axis-aligned
bounding box (AABB), intersect them, then mark per-section points that
fall inside the intersection. A ``shrink`` margin can be used to add a
safety inset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "maximal_common_region",
]


def maximal_common_region(
    coords_per_section: Sequence[np.ndarray],
    shrink: float = 0.0,
) -> dict[str, Any]:
    """Identify the maximal axis-aligned region common to every section.

    Args:
        coords_per_section: list of (N_s, 2) arrays — xy coordinates per
            section (z is implicit in list order; only xy matters here).
        shrink: optional inward margin (same units as coords). The
            intersection AABB is contracted by this amount on each side
            so points exactly on the boundary are excluded.

    Returns:
        ``{
            "bbox": (xmin, xmax, ymin, ymax),
            "masks": tuple of bool arrays, one per section, True where
                that section's point falls inside the common AABB,
            "n_kept_per_section": int array of counts,
        }``.

    Raises:
        ValueError: empty input, non-2D coordinate arrays, or empty
            intersection (shrink larger than the common region).
    """
    sections = [np.asarray(c, dtype=np.float64) for c in coords_per_section]
    if len(sections) == 0:
        raise ValueError("coords_per_section must contain >= 1 section")
    for i, s in enumerate(sections):
        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError(
                f"section {i} must be (N, 2), got {s.shape}"
            )
        if s.shape[0] == 0:
            raise ValueError(f"section {i} is empty")

    xmin = max(float(s[:, 0].min()) for s in sections)
    xmax = min(float(s[:, 0].max()) for s in sections)
    ymin = max(float(s[:, 1].min()) for s in sections)
    ymax = min(float(s[:, 1].max()) for s in sections)

    xmin += shrink
    xmax -= shrink
    ymin += shrink
    ymax -= shrink
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "common AABB is empty after applying shrink; either sections do "
            "not overlap or shrink is too large"
        )

    masks: list[np.ndarray] = []
    n_kept: list[int] = []
    for s in sections:
        m = (
            (s[:, 0] >= xmin)
            & (s[:, 0] <= xmax)
            & (s[:, 1] >= ymin)
            & (s[:, 1] <= ymax)
        )
        masks.append(m)
        n_kept.append(int(m.sum()))

    return {
        "bbox": (xmin, xmax, ymin, ymax),
        "masks": tuple(masks),
        "n_kept_per_section": np.array(n_kept, dtype=np.int64),
    }
