"""SpatialZ head-to-head adapter — the canonical 2.5D virtual-slice baseline
for continuous-3D comparisons.

Reference:
    Lin, S. et al. "Bridging the dimensional gap from planar spatial
    transcriptomics to 3D cell atlases." Nat. Methods 23, 360-372 (2025).
    https://github.com/senlin-lin/SpatialZ
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter

_INSTALL_HINT = (
    "spatialz-not-installed: clone https://github.com/senlin-lin/SpatialZ "
    "and install requirements; adapter expects a `spatialz` importable module"
)


class SpatialZAdapter(VolumeBaseAdapter):
    name = "spatialz"

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("spatialz", "SpatialZ", "spatial_z"):
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

        spatialz = importlib.import_module(getattr(self, "_module_name", "spatialz"))

        model_cls = None
        for cls_name in ("SpatialZ", "Reconstructor", "Model"):
            model_cls = getattr(spatialz, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError(
                "spatialz module does not expose SpatialZ / Reconstructor / Model class"
            )

        truth_z = inp.truth_z_values()
        model = model_cls(device=self.device)
        model.fit(visible)
        volume = model.predict(virtual_z=truth_z)

        if isinstance(volume, ad.AnnData):
            return volume

        # Otherwise assume the API returns a list of slices and concat.
        if isinstance(volume, (list, tuple)):
            return sc.concat(list(volume), axis=0, join="outer")

        raise RuntimeError(
            f"SpatialZ returned unsupported type {type(volume)}; expected AnnData or list"
        )
