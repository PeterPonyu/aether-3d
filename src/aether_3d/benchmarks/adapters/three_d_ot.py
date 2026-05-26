"""3d-OT adapter — geometry-aware OT alignment + 3D reconstruction.

Reference:
    Anonymous (dbjzs GitHub). "3d-OT: a deep geometry-aware framework for
    heterogeneous slices alignment of spatial multi-omics."
    Nat. Methods (2026), DOI 10.1038/s41592-026-03034-9.
    https://github.com/dbjzs/3d-OT

3d-OT uses PointNet++ + optimal transport with soft communication and
reports Chamfer distance as its primary spatial metric — perfectly aligned
with our `compute_volume_metrics` contract.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter

_INSTALL_HINT = (
    "three-d-ot-not-installed: clone https://github.com/dbjzs/3d-OT and install "
    "requirements; adapter expects a `three_d_ot` or `ot3d` importable module"
)


class ThreeDOTAdapter(VolumeBaseAdapter):
    name = "3d-ot"

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("three_d_ot", "ot3d", "threed_ot", "ThreeDOT"):
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

        ot3d = importlib.import_module(getattr(self, "_module_name", "three_d_ot"))

        model_cls = None
        for cls_name in ("ThreeDOT", "OT3D", "Reconstructor", "Model"):
            model_cls = getattr(ot3d, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError(
                "3d-OT module does not expose ThreeDOT / OT3D / Reconstructor / Model class"
            )

        truth_z = inp.truth_z_values()
        model = model_cls(device=self.device)
        model.fit(visible)
        volume = model.reconstruct(virtual_z=truth_z)

        if isinstance(volume, ad.AnnData):
            return volume
        if isinstance(volume, (list, tuple)):
            return sc.concat(list(volume), axis=0, join="outer")
        raise RuntimeError(
            f"3d-OT returned unsupported type {type(volume)}; expected AnnData or list"
        )
