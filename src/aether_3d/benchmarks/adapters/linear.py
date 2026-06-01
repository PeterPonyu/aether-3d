"""Linear-interpolation baseline: each virtual depth's cells are interpolated
linearly between the two bracketing visible slices.

Mid-grade 2.5D baseline; the standard linear-interpolation reference for
virtual-slice comparisons. Always available.

Cells from each side are paired by spatial nearest-neighbor in 2D (the only
honest signal available without ground-truth correspondences); gene
expression + spatial coordinates are interpolated by the weight
w = (z_target - z_a) / (z_b - z_a). The pairing is deterministic in the
cell content (not the input row order), so reported baseline numbers no
longer depend on how rows happen to be laid out in the input AnnData
(issue #147).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter


class LinearInterpAdapter(VolumeBaseAdapter):
    name = "linear-interp"

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        if len(visible) < 2:
            raise RuntimeError("linear-interp baseline needs at least two visible slices")

        truth_z = inp.truth_z_values()
        if not truth_z:
            n_genes = visible[0].n_vars
            empty = ad.AnnData(X=np.zeros((0, n_genes), dtype=np.float32))
            empty.var_names = visible[0].var_names
            empty.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
            empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
            empty.obs[inp.z_key] = np.zeros((0,), dtype=np.float32)
            return empty

        # Sort visible slices by z
        visible_z = []
        for s in visible:
            z_col = s.obs[inp.z_key].astype(float).values if inp.z_key in s.obs else None
            visible_z.append(float(np.mean(z_col)) if (z_col is not None and len(z_col)) else 0.0)
        order = np.argsort(visible_z)
        sorted_slices = [visible[i] for i in order]
        sorted_z = [visible_z[i] for i in order]
        sorted_z_arr = np.asarray(sorted_z, dtype=np.float32)

        out_slices: list[ad.AnnData] = []
        for z_target in truth_z:
            # Find the bracketing pair (z_a < z_target < z_b). Extrapolate if outside.
            if z_target <= sorted_z_arr[0]:
                a_idx, b_idx = 0, 1
            elif z_target >= sorted_z_arr[-1]:
                a_idx, b_idx = len(sorted_z_arr) - 2, len(sorted_z_arr) - 1
            else:
                b_idx = int(np.searchsorted(sorted_z_arr, z_target))
                a_idx = b_idx - 1

            a, b = sorted_slices[a_idx], sorted_slices[b_idx]
            z_a, z_b = float(sorted_z_arr[a_idx]), float(sorted_z_arr[b_idx])
            denom = (z_b - z_a) if abs(z_b - z_a) > 1e-9 else 1.0
            w = (float(z_target) - z_a) / denom

            n_pair = min(a.n_obs, b.n_obs)

            # Ground-truth-style pairing: iterate the smaller slice in a
            # coords-deterministic order, then for each cell pick the nearest
            # neighbor in the other slice by 2D Euclidean distance. This makes
            # the output a function of cell *content*, not input row order
            # (issue #147).
            xy_a_full = np.asarray(a.obsm[inp.spatial_key], dtype=np.float32)
            xy_b_full = np.asarray(b.obsm[inp.spatial_key], dtype=np.float32)

            order_a = np.lexsort((xy_a_full[:, 1], xy_a_full[:, 0]))[:n_pair]
            dists = ((xy_a_full[order_a, None, :] - xy_b_full[None, :, :]) ** 2).sum(axis=2)
            nearest_b = np.argmin(dists, axis=1)

            X_a_raw = a.X[order_a]
            if hasattr(X_a_raw, "toarray"):
                X_a_raw = X_a_raw.toarray()
            X_a = np.asarray(X_a_raw, dtype=np.float32)
            X_b_raw = b.X[nearest_b]
            if hasattr(X_b_raw, "toarray"):
                X_b_raw = X_b_raw.toarray()
            X_b = np.asarray(X_b_raw, dtype=np.float32)
            X_v = (1.0 - w) * X_a + w * X_b

            xy_a = xy_a_full[order_a]
            xy_b = xy_b_full[nearest_b]
            xy_v = (1.0 - w) * xy_a + w * xy_b
            z_col = np.full((n_pair, 1), float(z_target), dtype=np.float32)

            v = ad.AnnData(X=X_v)
            v.var_names = a.var_names
            v.obs[inp.z_key] = float(z_target)
            v.obsm[inp.spatial_key] = xy_v
            v.obsm["spatial_3d"] = np.hstack([xy_v, z_col])
            out_slices.append(v)

        volume = sc.concat(out_slices, axis=0, join="outer")
        return volume
