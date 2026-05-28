"""InterpolAI adapter — optical-flow interpolation between adjacent slices.

Reference:
    Joshi, S., Forjaz, A., Han, K. et al. "InterpolAI: deep learning-based
    optical flow interpolation and restoration of biomedical images for
    improved 3D tissue mapping." Nat. Methods 22, 1556-1567 (2025).
    https://www.nature.com/articles/s41592-025-02712-4

Adapts InterpolAI's optical-flow frame interpolation (designed for biomedical
image stacks) to ST by treating each gene's spatial expression as a separate
"channel". The adapter records its specific input requirement (a gene-grid
representation of each slice) and reports unavailable when InterpolAI's
upstream library isn't installed.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter

_INSTALL_HINT = (
    "interpolai-not-installed: install via the InterpolAI release artifact "
    "(see Nature Methods supplement / GitHub); adapter expects an `interpolai` "
    "importable module"
)


class InterpolAIAdapter(VolumeBaseAdapter):
    name = "interpolai"

    def __init__(
        self,
        device: str = "cpu",
        grid_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.grid_size = grid_size

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("interpolai", "InterpolAI", "interpol_ai"):
            try:
                importlib.import_module(candidate)
                self._module_name = candidate
                return True, ""
            except ImportError:
                continue
        return False, _INSTALL_HINT

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        import importlib

        interpolai = importlib.import_module(
            getattr(self, "_module_name", "interpolai")
        )

        # InterpolAI's canonical API in the released package is a function
        # `interpolate(frames, n_intermediate)` returning the interpolated stack.
        interpolate = getattr(interpolai, "interpolate", None)
        if interpolate is None:
            # Try the model class form
            model_cls = None
            for cls_name in ("InterpolAI", "FrameInterpolator", "Model"):
                model_cls = getattr(interpolai, cls_name, None)
                if model_cls is not None:
                    break
            if model_cls is None:
                raise RuntimeError(
                    "interpolai module does not expose interpolate() or a model class"
                )

            def _interp(frames: np.ndarray, n_intermediate: int) -> np.ndarray:
                m = model_cls(device=self.device)
                return np.asarray(m.interpolate(frames, n_intermediate=n_intermediate))

            interpolate = _interp

        truth_z = inp.truth_z_values()

        # Sort visible slices by z so the interpolator gets monotonic input
        visible_z = []
        for s in visible:
            zc = s.obs[inp.z_key].astype(float).values if inp.z_key in s.obs else None
            visible_z.append(float(np.mean(zc)) if (zc is not None and len(zc)) else 0.0)
        order = np.argsort(visible_z)
        sorted_slices = [visible[i] for i in order]

        # Render each slice as a (G, H, W) gene grid; InterpolAI interpolates
        # between consecutive grids, producing n_intermediate frames per pair.
        frames = np.stack(
            [_slice_to_grid(s, inp.spatial_key, self.grid_size) for s in sorted_slices]
        )
        n_intermediate = max(1, len(truth_z))
        interpolated = interpolate(frames, n_intermediate=n_intermediate)
        # interpolated shape: (n_frames + n_intermediate*(n_frames-1), G, H, W)

        out_slices: list[ad.AnnData] = []
        for ti, z_target in enumerate(truth_z):
            grid = interpolated[ti % len(interpolated)]
            v = _grid_to_slice(grid, sorted_slices[0].var_names, z_target, inp.z_key, inp.spatial_key)
            out_slices.append(v)

        if not out_slices:
            n_genes = visible[0].n_vars
            empty = ad.AnnData(X=np.zeros((0, n_genes), dtype=np.float32))
            empty.var_names = visible[0].var_names
            empty.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
            empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
            empty.obs[inp.z_key] = np.zeros((0,), dtype=np.float32)
            return empty
        return sc.concat(out_slices, axis=0, join="outer")


def _slice_to_grid(adata: ad.AnnData, spatial_key: str, grid_size: int) -> np.ndarray:
    """Render an ST slice as a (n_genes, H, W) gene grid via mean-pooled binning."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if coords.shape[0] == 0:
        return np.zeros((adata.n_vars, grid_size, grid_size), dtype=np.float32)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    ix = np.clip(((coords[:, 0] - xmin) / xr * (grid_size - 1)).astype(int), 0, grid_size - 1)
    iy = np.clip(((coords[:, 1] - ymin) / yr * (grid_size - 1)).astype(int), 0, grid_size - 1)

    grid = np.zeros((adata.n_vars, grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    for ci in range(coords.shape[0]):
        grid[:, iy[ci], ix[ci]] += X[ci]
        counts[iy[ci], ix[ci]] += 1.0
    counts = np.where(counts == 0, 1.0, counts)
    grid /= counts[None, :, :]
    return grid


def _grid_to_slice(
    grid: np.ndarray,
    var_names: pd.Index,
    z_value: float,
    z_key: str,
    spatial_key: str,
) -> ad.AnnData:
    """Inverse of _slice_to_grid: emit one virtual cell per grid voxel."""
    n_genes, H, W = grid.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    X = grid.reshape(n_genes, -1).T.astype(np.float32)
    nonzero = (X.sum(axis=1) > 0)
    X = X[nonzero]
    coords = coords[nonzero]
    n = X.shape[0]
    if n == 0:
        empty = ad.AnnData(X=np.zeros((0, n_genes), dtype=np.float32))
        empty.var_names = var_names
        empty.obsm[spatial_key] = np.zeros((0, 2), dtype=np.float32)
        empty.obsm["spatial_3d"] = np.zeros((0, 3), dtype=np.float32)
        empty.obs[z_key] = np.zeros((0,), dtype=np.float32)
        return empty

    v = ad.AnnData(X=X)
    v.var_names = var_names
    v.obsm[spatial_key] = coords
    v.obsm["spatial_3d"] = np.hstack([coords, np.full((n, 1), float(z_value), dtype=np.float32)])
    v.obs[z_key] = float(z_value)
    return v
