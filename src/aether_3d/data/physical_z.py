"""Physical inter-slice z-coordinate resolution for serial-slice stacks.

Issue #222: every real-data code path used to inject a synthetic
``z_coord = idx * 10.0`` (a uniform, arbitrary 10-unit spacing) instead of the
genuine physical inter-section distance recorded in the data. For the Moffitt
2018 MERFISH hypothalamus stack the real anterior-posterior coordinate is the
``Bregma`` position in mm (~0.05 mm spacing between sections); the cached
baseline slices expose it as ``obs['slice_id']`` (a string such as ``"0.04"``).
That 200x/20x scale mismatch corrupts every physical-spacing-dependent metric
and figure (voxel cosine, Z-density curves, depth-along-Z trajectories,
multi-planar slicing, flow-divergence / velocity-anisotropy).

``resolve_slice_z`` derives each slice's z from physical metadata when present
and otherwise falls back to a CONFIGURABLE spacing (never a hard-coded 10),
returning an explicit ``z_is_physical`` flag so downstream code/figures can gate
physical-distance claims.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import anndata as ad
import numpy as np

# obs columns that, when present and numeric (or numeric-string) and constant
# within a slice, carry the genuine physical z / anterior-posterior coordinate.
# Order = preference. ``Bregma`` is the canonical Moffitt 2018 field; the cached
# baseline slices store the same mm value under ``slice_id``.
PHYSICAL_Z_OBS_FIELDS: tuple[str, ...] = (
    "Bregma",
    "bregma",
    "z_coord_um",
    "z_um",
    "slice_id",
    "z",
    "Z",
    "Centroid_Z",
    "center_z",
)

# obsm matrix whose 3rd column is a native physical z (e.g. squidpy spatial3d).
PHYSICAL_Z_OBSM_KEYS: tuple[str, ...] = ("spatial3d",)


def _slice_physical_z(adata: ad.AnnData) -> float | None:
    """Return the single physical z value for ``adata`` if one is available.

    A field qualifies only if it is constant across the slice's cells (a serial
    section sits at one physical depth) and parses to a finite float.
    """
    for field in PHYSICAL_Z_OBS_FIELDS:
        if field in adata.obs.columns:
            vals = adata.obs[field]
            try:
                numeric = np.asarray(vals.astype(str).to_numpy(), dtype=str)
                numeric = numeric.astype(np.float64)
            except (ValueError, TypeError):
                continue
            uniq = np.unique(numeric[np.isfinite(numeric)])
            if uniq.size == 1:
                return float(uniq[0])
            # Non-constant within the slice: not a per-section depth label.
            # Fall through to the next candidate field.

    for key in PHYSICAL_Z_OBSM_KEYS:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                col = arr[:, 2].astype(np.float64)
                uniq = np.unique(col[np.isfinite(col)])
                if uniq.size == 1:
                    return float(uniq[0])
                # A native per-cell 3D z: use the section's mean depth.
                if uniq.size > 1:
                    return float(np.nanmean(col))
    return None


def resolve_slice_z(
    slices: Sequence[ad.AnnData],
    fallback_spacing: float = 10.0,
) -> tuple[list[float], bool]:
    """Resolve a physical z value per serial slice.

    Parameters
    ----------
    slices:
        Ordered list of AnnData serial sections (index order == stack order).
    fallback_spacing:
        Spacing (in the dataset's physical unit) used ONLY when no physical
        metadata is found; the synthetic ladder becomes ``idx * fallback_spacing``.
        This is configurable (data-card / parameter driven) — never hard-coded
        to 10 at the call site.

    Returns
    -------
    (z_values, z_is_physical):
        ``z_values`` is a list of floats (one per slice, in order). When every
        slice exposes a physical z field, ``z_is_physical`` is True and the
        returned values are the genuine physical coordinates. Otherwise the
        function emits a ``UserWarning`` and returns the synthetic fallback
        ladder with ``z_is_physical=False``.
    """
    if len(slices) == 0:
        return [], False

    physical = [_slice_physical_z(a) for a in slices]
    if all(z is not None for z in physical):
        return [float(z) for z in physical], True  # type: ignore[arg-type]

    warnings.warn(
        "No physical inter-slice z metadata (e.g. obs['Bregma'] / obs['slice_id'] "
        "/ obsm['spatial3d']) found on at least one slice; falling back to a "
        f"synthetic ladder idx*{fallback_spacing}. Physical-spacing-dependent "
        "metrics/figures must be treated as DESCRIPTIVE (z_is_physical=False).",
        UserWarning,
        stacklevel=2,
    )
    z_values = [float(i) * float(fallback_spacing) for i in range(len(slices))]
    return z_values, False
