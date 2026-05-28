"""ASIGN adapter — anatomy-aware 3D ST imputation using WSI + 1 ST slide.

Reference:
    Zhu, J. et al. "ASIGN: An Anatomy-aware Spatial Imputation Graphic
    Network for 3D Spatial Transcriptomics." CVPR 2025.
    https://arxiv.org/abs/2412.03026
    https://github.com/hrlblab/ASIGN

ASIGN's input regime differs from the rest: it takes 3D WSI sections + one
2D ST slide. The adapter accepts an optional `wsi_stack` option for the WSI
images; when missing, it records that as a structured unavailable reason
rather than silently falling back.
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import scanpy as sc

from ..contract import VolumeAdapterInput, VolumeBaseAdapter

_INSTALL_HINT = (
    "asign-not-installed: clone https://github.com/hrlblab/ASIGN and install "
    "requirements; adapter expects an `asign` importable module"
)


class ASIGNAdapter(VolumeBaseAdapter):
    name = "asign"

    def __init__(
        self,
        wsi_stack: Optional[list] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.wsi_stack = wsi_stack
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("asign", "ASIGN"):
            try:
                importlib.import_module(candidate)
                self._module_name = candidate
                break
            except ImportError:
                continue
        else:
            return False, _INSTALL_HINT

        if self.wsi_stack is None:
            return False, (
                "asign-requires-wsi-stack: pass wsi_stack=list[ndarray] (3D WSI "
                "sections aligned with the ST slices); ASIGN's input regime is "
                "WSI + 1 ST slide, not multi-slice ST"
            )
        return True, ""

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        import importlib

        asign = importlib.import_module(getattr(self, "_module_name", "asign"))

        model_cls = None
        for cls_name in ("ASIGN", "Imputer", "Model"):
            model_cls = getattr(asign, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError(
                "asign module does not expose ASIGN / Imputer / Model class"
            )

        truth_z = inp.truth_z_values()
        model = model_cls(device=self.device)
        model.fit(visible_slices=visible, wsi_stack=self.wsi_stack)
        volume = model.impute(virtual_z=truth_z)

        if isinstance(volume, ad.AnnData):
            return volume
        if isinstance(volume, (list, tuple)):
            return sc.concat(list(volume), axis=0, join="outer")
        raise RuntimeError(
            f"ASIGN returned unsupported type {type(volume)}; expected AnnData or list"
        )
