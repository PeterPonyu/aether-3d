"""Nearest-slice baseline: each virtual depth is filled by copying the closest
*visible* physical slice's cells, with their z-coords overwritten to the
target depth.

Lowest possible 2.5D baseline. Always available.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter


def _empty_volume_like(template: ad.AnnData, inp: VolumeAdapterInput) -> ad.AnnData:
    n_genes = template.n_vars
    empty = ad.AnnData(X=np.zeros((0, n_genes), dtype=np.float32))
    empty.var_names = template.var_names
    empty.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
    empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
    empty.obs[inp.z_key] = np.zeros((0,), dtype=np.float32)
    return empty


class NearestSliceAdapter(VolumeBaseAdapter):
    name = "nearest-slice"

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        if not visible:
            raise RuntimeError("nearest-slice baseline needs at least one visible slice")

        truth_z = inp.truth_z_values()
        if not truth_z:
            return _empty_volume_like(visible[0], inp)

        # Visible slice z-anchors
        visible_z = []
        for s in visible:
            z_col = s.obs[inp.z_key].astype(float).values if inp.z_key in s.obs else None
            if z_col is None or len(z_col) == 0:
                visible_z.append(0.0)
            else:
                visible_z.append(float(np.mean(z_col)))
        visible_z_arr = np.asarray(visible_z, dtype=np.float32)

        out_slices: list[ad.AnnData] = []
        for z_target in truth_z:
            nearest_idx = int(np.argmin(np.abs(visible_z_arr - z_target)))
            copy = visible[nearest_idx].copy()
            copy.obs[inp.z_key] = float(z_target)
            # Stamp 3D coords
            xy = np.asarray(copy.obsm[inp.spatial_key], dtype=np.float32)
            zcol = np.full((xy.shape[0], 1), float(z_target), dtype=np.float32)
            copy.obsm["spatial_3d"] = np.hstack([xy, zcol])
            out_slices.append(copy)

        volume = sc.concat(out_slices, axis=0, join="outer")
        return volume
