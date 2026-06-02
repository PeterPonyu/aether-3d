"""Clean-room 2.5D stacking reconstruction baseline (#221).

This is a license-clean, brand-independent re-implementation of the canonical
"2.5D virtual-slice" reconstruction *idea* used as a prior-art baseline for
continuous-3D methods. It is NOT a port of, and contains no source or class
names from, any published 2.5D package (e.g. the SpatialZ / DeepSpatial line of
work). Only the mathematical intent — interpolating virtual cells between two
bracketing physical slices by 2D nearest-neighbour pairing — is reproduced, per
the brand-independence boundary in SPATIAL-OMICS-REFORM.md.

Algorithm (identity-preserving 2.5D stacking):
    For each held-out (truth) physical depth ``z*`` the adapter finds the two
    visible slices that bracket ``z*`` in physical z (the *resolved* z from the
    contract's ``truth_z_values()`` / per-slice z column — never an ``idx * 10``
    proxy, honouring the physical-z work of #222). The mixing weight is
    ``w = (z* - z_lo) / (z_hi - z_lo)``. The virtual population is assembled by
    *borrowing whole cells*: each lower-slice cell is carried forward with
    probability ``1 - w`` (keeping its real expression vector and 2D coords),
    and the complementary fraction is drawn from the upper slice via 2D
    nearest-neighbour lookup. Every virtual cell therefore inherits the EXACT
    measured expression of a real cell (no gene-wise smoothing); only the cell
    *population* is interpolated across depth, and z is stamped to ``z*``.

This identity-preserving design distinguishes the baseline from the gene-blend
``linear-interp`` adapter and the single-slice ``nearest-slice`` adapter, so the
three together bracket the 2.5D family. It is deliberately simple — any quality
gap against a learned continuous reconstruction (Aether3D) reflects the method,
not the baseline's machinery. It is always available (pure NumPy / AnnData) and
is wired into the holdout contrast in
``scripts/e2e/validate_holdout_slice.py``.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter


class Stack25DAdapter(VolumeBaseAdapter):
    """License-clean 2.5D virtual-slice stacking baseline.

    Brand-independent, always-available reconstruction: virtual cells at each
    held-out depth are produced by 2D nearest-neighbour pairing + linear blend
    between the two bracketing visible slices at the resolved physical z.
    """

    name = "stack-2.5d"

    def __init__(self, device: str = "cpu", seed: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.seed = seed

    def _check_available(self) -> tuple[bool, str]:
        # Pure NumPy / AnnData — no external package required.
        return True, ""

    def _empty_like(self, template: ad.AnnData, inp: VolumeAdapterInput) -> ad.AnnData:
        empty = ad.AnnData(X=np.zeros((0, template.n_vars), dtype=np.float32))
        empty.var_names = template.var_names
        empty.obsm[inp.spatial_key] = np.zeros((0, 2), dtype=np.float32)
        empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
        empty.obs[inp.z_key] = np.zeros((0,), dtype=np.float32)
        return empty

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        if len(visible) < 2:
            raise RuntimeError("2.5D stacking baseline needs at least two visible slices")

        # Resolved physical z of held-out (truth) slices — issue #222: use the
        # contract's resolved z, never idx*spacing.
        truth_z = inp.virtual_z if inp.virtual_z is not None else inp.truth_z_values()
        if not truth_z:
            return self._empty_like(visible[0], inp)

        # Visible-slice physical z anchors (mean of per-cell z column).
        visible_z = []
        for s in visible:
            z_col = s.obs[inp.z_key].astype(float).values if inp.z_key in s.obs else None
            visible_z.append(
                float(np.mean(z_col)) if (z_col is not None and len(z_col)) else 0.0
            )
        order = np.argsort(visible_z)
        sorted_slices = [visible[i] for i in order]
        sorted_z = np.asarray([visible_z[i] for i in order], dtype=np.float32)

        rng = np.random.default_rng(inp.seed if inp.seed is not None else self.seed)
        out_slices: list[ad.AnnData] = []
        for z_target in truth_z:
            z_target = float(z_target)
            if np.isnan(z_target):
                continue

            # Bracket z_target by physical z; clamp/extrapolate at the ends.
            if z_target <= sorted_z[0]:
                lo_idx, hi_idx = 0, 1
            elif z_target >= sorted_z[-1]:
                lo_idx, hi_idx = len(sorted_z) - 2, len(sorted_z) - 1
            else:
                hi_idx = int(np.searchsorted(sorted_z, z_target))
                lo_idx = hi_idx - 1

            lo, hi = sorted_slices[lo_idx], sorted_slices[hi_idx]
            z_lo, z_hi = float(sorted_z[lo_idx]), float(sorted_z[hi_idx])
            denom = (z_hi - z_lo) if abs(z_hi - z_lo) > 1e-9 else 1.0
            w = float(np.clip((z_target - z_lo) / denom, 0.0, 1.0))

            xy_lo = np.asarray(lo.obsm[inp.spatial_key], dtype=np.float32)
            xy_hi = np.asarray(hi.obsm[inp.spatial_key], dtype=np.float32)

            # Deterministic order over the lower slice (coords-sorted, not row
            # order) for reproducibility independent of input row layout.
            order_lo = np.lexsort((xy_lo[:, 1], xy_lo[:, 0]))
            xy_lo = xy_lo[order_lo]
            X_lo = self._dense(lo.X[order_lo])
            lab_lo = (
                lo.obs[inp.label_key].astype(str).values[order_lo]
                if inp.label_key and inp.label_key in lo.obs
                else None
            )

            # Identity-preserving population interpolation: borrow whole cells.
            # A fraction ``w`` of the virtual population is drawn from the upper
            # slice (via 2D NN of each lower cell), the rest stay on the lower
            # slice. Each kept cell carries its real expression vector verbatim.
            n = xy_lo.shape[0]
            from_hi = rng.random(n) < w
            X_v = X_lo.copy()
            xy_v = xy_lo.copy()
            lab_v = None if lab_lo is None else lab_lo.copy()
            if from_hi.any():
                dists = ((xy_lo[from_hi, None, :] - xy_hi[None, :, :]) ** 2).sum(axis=2)
                nn = np.argmin(dists, axis=1)
                X_v[from_hi] = self._dense(hi.X[nn])
                xy_v[from_hi] = xy_hi[nn]
                if lab_v is not None and inp.label_key in hi.obs:
                    lab_v[from_hi] = hi.obs[inp.label_key].astype(str).values[nn]

            z_col = np.full((xy_v.shape[0], 1), z_target, dtype=np.float32)
            v = ad.AnnData(X=X_v.astype(np.float32))
            v.var_names = lo.var_names
            v.obs[inp.z_key] = z_target
            v.obsm[inp.spatial_key] = xy_v.astype(np.float32)
            v.obsm["spatial_3d"] = np.hstack([xy_v, z_col]).astype(np.float32)
            if lab_v is not None:
                v.obs[inp.label_key] = lab_v
            out_slices.append(v)

        if not out_slices:
            return self._empty_like(visible[0], inp)
        return sc.concat(out_slices, axis=0, join="outer")

    @staticmethod
    def _dense(X: Any) -> np.ndarray:
        return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


# Back-compat alias: the prior public name remains importable, but it now points
# at the brand-independent clean-room implementation (the old SpatialZ-source
# stub is removed). The wrapper keeps the historical ``name`` so any pinned
# result key referencing it stays stable.
class SpatialZAdapter(Stack25DAdapter):
    """Deprecated alias for :class:`Stack25DAdapter` (kept for back-compat)."""

    name = "stack-2.5d"
