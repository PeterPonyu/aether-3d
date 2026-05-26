"""Linear-interpolation baseline: each virtual depth's cells are interpolated
linearly between the two bracketing visible slices.

Mid-grade 2.5D baseline; the standard linear-interpolation reference for
virtual-slice comparisons. Always available.

Cells from each side are paired by simple index up to min(n_a, n_b), and
gene expression + spatial coordinates are interpolated by the weight
w = (z_target - z_a) / (z_b - z_a).
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
            X_a = a.X[:n_pair]
            if hasattr(X_a, "toarray"):
                X_a = X_a.toarray()
            X_a = np.asarray(X_a, dtype=np.float32)
            X_b = b.X[:n_pair]
            if hasattr(X_b, "toarray"):
                X_b = X_b.toarray()
            X_b = np.asarray(X_b, dtype=np.float32)
            X_v = (1.0 - w) * X_a + w * X_b

            xy_a = np.asarray(a.obsm[inp.spatial_key][:n_pair], dtype=np.float32)
            xy_b = np.asarray(b.obsm[inp.spatial_key][:n_pair], dtype=np.float32)
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
