"""Cross-slice precondition validation for reconstruct_continuous_volume (#132).

A deliberately inconsistent slice stack must fail with a precise, named
``ValueError`` *before* any cdist / UOT / multinomial call, rather than
crashing deep inside the velocity field or silently corrupting the coupling
with non-finite inputs.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor


def _make_slices(n_slices: int = 3, n_cells: int = 12, n_genes: int = 8, seed: int = 0) -> list[ad.AnnData]:
    rng = np.random.default_rng(seed)
    slices: list[ad.AnnData] = []
    for z in range(n_slices):
        adata = ad.AnnData(
            X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
            obs={
                "cell_class": ["T", "B"] * (n_cells // 2),
                "z_coord": [float(z)] * n_cells,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
        slices.append(adata)
    return slices


def _reconstructor() -> AetherReconstructor:
    cfg = Aether3DConfig(seed=0, hidden_size=8, depth=1, num_heads=2, patch_size=4)
    recon = AetherReconstructor(cfg)
    recon.setup_data(_make_slices())  # valid stack to build the model
    return recon


def test_validate_accepts_consistent_stack() -> None:
    recon = _reconstructor()
    # Should not raise.
    recon._validate_slices(_make_slices())


def test_mismatched_gene_dimension_raises_named_error() -> None:
    recon = _reconstructor()
    stack = _make_slices(n_slices=2, n_genes=8)
    stack[1] = _make_slices(n_slices=1, n_genes=9, seed=1)[0]
    with pytest.raises(ValueError, match=r"slice 1 has 9 genes, expected 8"):
        recon.reconstruct_continuous_volume(stack, num_depths=3)


def test_missing_spatial_key_raises_named_error() -> None:
    recon = _reconstructor()
    stack = _make_slices(n_slices=2)
    del stack[1].obsm["spatial"]
    with pytest.raises(ValueError, match=r"slice 1 is missing obsm\['spatial'\]"):
        recon.reconstruct_continuous_volume(stack, num_depths=3)


def test_missing_label_key_raises_named_error() -> None:
    recon = _reconstructor()
    stack = _make_slices(n_slices=2)
    del stack[0].obs["cell_class"]
    with pytest.raises(ValueError, match=r"slice 0 is missing obs\['cell_class'\]"):
        recon.reconstruct_continuous_volume(stack, num_depths=3)


def test_nonfinite_coordinates_raise_named_error() -> None:
    recon = _reconstructor()
    stack = _make_slices(n_slices=2)
    coords = np.asarray(stack[1].obsm["spatial"]).copy()
    coords[0, 0] = np.nan
    stack[1].obsm["spatial"] = coords
    with pytest.raises(ValueError, match=r"slice 1 obsm\['spatial'\] contains non-finite"):
        recon.reconstruct_continuous_volume(stack, num_depths=3)


def test_nonfinite_expression_raises_named_error() -> None:
    recon = _reconstructor()
    stack = _make_slices(n_slices=2)
    X = np.asarray(stack[0].X).copy()
    X[1, 1] = np.inf
    stack[0].X = X
    with pytest.raises(ValueError, match=r"slice 0 expression matrix \(X\) contains non-finite"):
        recon.reconstruct_continuous_volume(stack, num_depths=3)
