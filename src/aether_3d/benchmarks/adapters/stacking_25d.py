"""Clean-room 2.5D virtual-slice *stacking* baseline.

This is the canonical "2.5D" reconstruction strategy used as the comparison
floor for continuous-3D methods: a virtual slice at depth ``z`` is built by
*stacking* the two bracketing observed slices onto the target plane, with each
observed plane contributing a depth-proportional, identity-preserving subset of
its real cells. There is no learned model and no gene-wise smoothing — every
virtual cell carries the *exact* expression vector and 2D position of a real
measured cell, so the baseline reflects pure stacking/interpolation geometry and
its quality gap to a continuous method is attributable to the method, not to
artifacts introduced on the baseline side.

Algorithm (deterministic, shuffle-invariant, always available):
    For a target depth ``z`` bracketed by visible slices ``a`` (``z_a``) and
    ``b`` (``z_b``), the depth weight ``w = (z - z_a) / (z_b - z_a)`` sets how
    many cells each plane stacks onto the virtual plane: ``k_a = (1 - w) * n_a``
    from ``a`` and ``k_b = w * n_b`` from ``b``. Cells are selected by an
    evenly-spaced decimation over a content-deterministic spatial ordering
    (``lexsort`` on the 2D coordinates), which keeps the subset spatially
    representative and independent of input row order. Each selected cell keeps
    its real expression vector and XY position; its z is overwritten to the
    target depth. When only one bracketing slice is visible the method degrades
    to stacking that single plane.

Reference (clean-room reimplementation of the *2.5D virtual-slice stacking*
strategy; no third-party code imported):
    Lin, S. et al. "Bridging the dimensional gap from planar spatial
    transcriptomics to 3D cell atlases." Nat. Methods 23, 360-372 (2025).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter


def _spatial_decimation(coords: npt.NDArray[np.floating[Any]], k: int) -> npt.NDArray[np.int64]:
    """Deterministic, content-keyed selection of ``k`` evenly-spaced cells.

    Cells are ordered by ``lexsort`` on their 2D coordinates (so the result is
    invariant to input row order) and ``k`` indices are sampled at even strides
    across that ordering, keeping the subset spatially representative.
    """
    n = int(coords.shape[0])
    k = max(1, min(k, n))
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    picks = np.linspace(0, n - 1, k).round().astype(np.int64)
    return np.asarray(order[picks], dtype=np.int64)


def _stack_plane(
    src: ad.AnnData,
    k: int,
    z_target: float,
    spatial_key: str,
    z_key: str,
) -> ad.AnnData:
    """Stack ``k`` identity-preserving cells from ``src`` onto the ``z`` plane."""
    xy = np.asarray(src.obsm[spatial_key], dtype=np.float32)[:, :2]
    sel = _spatial_decimation(xy, k)
    X_sel_raw = src.X[sel]
    if hasattr(X_sel_raw, "toarray"):
        X_sel_raw = X_sel_raw.toarray()
    X_sel = np.asarray(X_sel_raw, dtype=np.float32)
    xy_sel = xy[sel]
    z_col = np.full((xy_sel.shape[0], 1), float(z_target), dtype=np.float32)

    out = ad.AnnData(X=X_sel)
    out.var_names = src.var_names
    out.obs[z_key] = np.full((xy_sel.shape[0],), float(z_target), dtype=np.float32)
    out.obsm[spatial_key] = xy_sel
    out.obsm["spatial_3d"] = np.hstack([xy_sel, z_col])
    return out


class Stacking25DAdapter(VolumeBaseAdapter):
    """Clean-room 2.5D virtual-slice stacking baseline (always available)."""

    name = "stacking-2.5d"

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        if not visible:
            raise RuntimeError("stacking-2.5d baseline needs at least one visible slice")

        truth_z = inp.truth_z_values()
        spatial_key = inp.spatial_key
        z_key = inp.z_key

        # Visible slice z-anchors (mean physical z per slice), sorted by z.
        visible_z: list[float] = []
        for s in visible:
            z_col = s.obs[z_key].astype(float).values if z_key in s.obs else None
            visible_z.append(float(np.mean(z_col)) if (z_col is not None and len(z_col)) else 0.0)
        order = np.argsort(visible_z)
        sorted_slices = [visible[i] for i in order]
        sorted_z = np.asarray([visible_z[i] for i in order], dtype=np.float32)

        if not truth_z:
            n_genes = visible[0].n_vars
            empty = ad.AnnData(X=np.zeros((0, n_genes), dtype=np.float32))
            empty.var_names = visible[0].var_names
            empty.obsm[spatial_key] = np.zeros((0, 2), dtype=np.float32)
            empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
            empty.obs[z_key] = np.zeros((0,), dtype=np.float32)
            return empty

        out_slices: list[ad.AnnData] = []
        for z_target in truth_z:
            if len(sorted_slices) == 1:
                # Single visible plane: stack it whole onto the target depth.
                out_slices.append(
                    _stack_plane(sorted_slices[0], sorted_slices[0].n_obs, z_target, spatial_key, z_key)
                )
                continue

            # Find the bracketing pair (z_a <= z_target <= z_b); clamp at ends.
            if z_target <= sorted_z[0]:
                a_idx, b_idx = 0, 1
            elif z_target >= sorted_z[-1]:
                a_idx, b_idx = len(sorted_z) - 2, len(sorted_z) - 1
            else:
                b_idx = int(np.searchsorted(sorted_z, z_target))
                a_idx = b_idx - 1

            a, b = sorted_slices[a_idx], sorted_slices[b_idx]
            z_a, z_b = float(sorted_z[a_idx]), float(sorted_z[b_idx])
            denom = (z_b - z_a) if abs(z_b - z_a) > 1e-9 else 1.0
            w = float(np.clip((float(z_target) - z_a) / denom, 0.0, 1.0))

            # Depth-proportional stacking: plane a contributes (1-w) of its cells,
            # plane b contributes w of its cells, onto the shared virtual plane.
            k_a = int(round((1.0 - w) * a.n_obs))
            k_b = int(round(w * b.n_obs))
            stacked: list[ad.AnnData] = []
            if k_a >= 1 or k_b == 0:
                stacked.append(_stack_plane(a, max(k_a, 1), z_target, spatial_key, z_key))
            if k_b >= 1 or k_a == 0:
                stacked.append(_stack_plane(b, max(k_b, 1), z_target, spatial_key, z_key))
            out_slices.append(sc.concat(stacked, axis=0, join="outer"))

        return sc.concat(out_slices, axis=0, join="outer")
