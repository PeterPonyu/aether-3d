"""Shared pytest fixtures for the Aether3D test suite.

Consolidates the synthetic serial-slice builder that previously lived as a
private ``_make_aether_slices`` copy in several test modules, and provides a
helper to inject a fake external-adapter module via ``sys.modules`` so the
adapter ``_reconstruct`` paths can be exercised without the real dependency
(issue #89).
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Callable, Iterator

import anndata as ad
import numpy as np
import pytest


@pytest.fixture
def make_aether_slices() -> Callable[..., list[ad.AnnData]]:
    """Factory for a stack of schema-valid synthetic serial slices.

    Each slice carries ``obs['cell_class']``, ``obs['z_coord']`` and
    ``obsm['spatial']`` — the keys ``Aether3DConfig`` defaults expect.
    """

    def _build(
        n_slices: int = 3,
        n_cells: int = 12,
        n_genes: int = 8,
        seed: int = 101,
    ) -> list[ad.AnnData]:
        rng = np.random.default_rng(seed)
        slices: list[ad.AnnData] = []
        for z in range(n_slices):
            adata = ad.AnnData(
                X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
                obs={
                    "cell_class": ["T", "B"] * (n_cells // 2)
                    + (["T"] if n_cells % 2 else []),
                    "z_coord": [float(z)] * n_cells,
                },
            )
            adata.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
            slices.append(adata)
        return slices

    return _build


@pytest.fixture
def stub_adapter_module() -> Iterator[Callable[[str, type], ModuleType]]:
    """Inject a fake module into ``sys.modules`` for the duration of a test.

    Returns a register function ``register(name, model_cls)`` that creates a
    module exposing ``model_cls`` (so e.g. an external-reconstruction adapter
    can import it and call ``fit``/``predict``). All injected modules are
    removed on teardown.
    """
    injected: list[str] = []

    def _register(name: str, model_cls: type) -> ModuleType:
        module = ModuleType(name)
        # Expose under the common class names adapters probe for.
        for attr in ("SpatialZ", "Reconstructor", "Model"):
            setattr(module, attr, model_cls)
        sys.modules[name] = module
        injected.append(name)
        return module

    yield _register

    for name in injected:
        sys.modules.pop(name, None)
