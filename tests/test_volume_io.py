"""Export-contract / round-trip tests for 3D volumes (CLAIM_LEDGER row 3).

Evidence that a reconstructed volume survives an AnnData ``.h5ad`` round-trip
losslessly and stays Scanpy-compatible — the export-contract evidence the
AnnData/Scanpy/SpatialData interoperability claim requires. (Real-data export
round-trip remains a separate gate; these use synthetic + reconstructor output.)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from aether_3d.volume_io import (
    DEFAULT_Z_KEY,
    SPATIAL_3D_KEY,
    assert_volume_schema,
    read_volume,
    write_volume,
)


def _make_volume(n: int = 12, n_genes: int = 5, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 50.0, size=(n, 2)).astype(np.float32)
    z = rng.uniform(-2.0, 2.0, size=(n, 1)).astype(np.float32)
    vol = ad.AnnData(X=rng.normal(size=(n, n_genes)).astype(np.float32))
    vol.var_names = [f"GENE_{j:02d}" for j in range(n_genes)]
    vol.obsm["spatial"] = xy
    vol.obsm[SPATIAL_3D_KEY] = np.hstack([xy, z]).astype(np.float32)
    vol.obs[DEFAULT_Z_KEY] = z[:, 0]
    vol.obs["cell_type"] = ["A", "B"] * (n // 2)
    return vol


def test_roundtrip_is_lossless(tmp_path) -> None:
    vol = _make_volume()
    path = write_volume(vol, tmp_path / "vol.h5ad")
    assert path.exists()
    back = read_volume(path)

    assert back.n_obs == vol.n_obs and back.n_vars == vol.n_vars
    assert list(back.var_names) == list(vol.var_names)
    np.testing.assert_allclose(np.asarray(back.X), np.asarray(vol.X), rtol=0, atol=0)
    np.testing.assert_allclose(back.obsm[SPATIAL_3D_KEY], vol.obsm[SPATIAL_3D_KEY])
    np.testing.assert_allclose(back.obsm["spatial"], vol.obsm["spatial"])
    np.testing.assert_allclose(
        back.obs[DEFAULT_Z_KEY].to_numpy(), vol.obs[DEFAULT_Z_KEY].to_numpy()
    )


def test_write_rejects_missing_spatial_3d(tmp_path) -> None:
    vol = _make_volume()
    del vol.obsm[SPATIAL_3D_KEY]
    with pytest.raises(ValueError, match=SPATIAL_3D_KEY):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_write_rejects_wrong_shape(tmp_path) -> None:
    vol = _make_volume()
    vol.obsm[SPATIAL_3D_KEY] = vol.obsm[SPATIAL_3D_KEY][:, :2]  # (N,2), not (N,3)
    with pytest.raises(ValueError, match="N, 3"):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_write_rejects_non_finite(tmp_path) -> None:
    vol = _make_volume()
    coords = vol.obsm[SPATIAL_3D_KEY].copy()
    coords[0, 2] = np.nan
    vol.obsm[SPATIAL_3D_KEY] = coords
    with pytest.raises(ValueError, match="non-finite"):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_write_rejects_missing_z(tmp_path) -> None:
    vol = _make_volume()
    del vol.obs[DEFAULT_Z_KEY]
    with pytest.raises(ValueError, match=DEFAULT_Z_KEY):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_write_rejects_non_finite_X(tmp_path) -> None:
    vol = _make_volume()
    X = np.asarray(vol.X).copy()
    X[0, 0] = np.inf
    vol.X = X
    with pytest.raises(ValueError, match="X contains non-finite"):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_write_rejects_none_X(tmp_path) -> None:
    vol = _make_volume()
    vol.X = None
    with pytest.raises(ValueError, match="X is None"):
        write_volume(vol, tmp_path / "bad.h5ad")


def test_roundtripped_volume_is_scanpy_compatible(tmp_path) -> None:
    """A standard Scanpy preprocessing op runs on the round-tripped volume."""
    import scanpy as sc

    vol = _make_volume(n=20, n_genes=8)
    # Make X non-negative so normalize/log are well-defined (count-like).
    vol.X = np.abs(np.asarray(vol.X)).astype(np.float32)
    back = read_volume(write_volume(vol, tmp_path / "vol.h5ad"))

    sc.pp.normalize_total(back, target_sum=1e4)
    sc.pp.log1p(back)
    sc.pp.pca(back, n_comps=3)
    assert "X_pca" in back.obsm
    assert back.obsm["X_pca"].shape == (20, 3)
    # 3-D schema still holds after scanpy mutated the object.
    assert_volume_schema(back)


def _recon_input_stack(
    n_slices: int = 3, n_cells: int = 16, n_genes: int = 6, seed: int = 0
) -> list[ad.AnnData]:
    """Minimal valid serial stack for the reconstructor (self-contained: does not
    depend on the synthetic-field generator, which lives on a separate branch)."""
    rng = np.random.default_rng(seed)
    stack: list[ad.AnnData] = []
    for zi in range(n_slices):
        a = ad.AnnData(X=np.abs(rng.normal(size=(n_cells, n_genes))).astype(np.float32))
        a.obs["cell_type"] = ["A", "B"] * (n_cells // 2)
        a.obs["z"] = float(zi)
        a.obsm["spatial"] = rng.uniform(0.0, 50.0, size=(n_cells, 2)).astype(np.float32)
        stack.append(a)
    return stack


def test_reconstructed_volume_roundtrips(tmp_path) -> None:
    """A REAL reconstructor output survives the export round-trip (row-3 evidence)."""
    from aether_3d.config.aether_config import Aether3DConfig
    from aether_3d.core.aether_reconstructor import AetherReconstructor

    stack = _recon_input_stack(n_slices=3, n_cells=16, n_genes=6, seed=0)
    cfg = Aether3DConfig(
        spatial_key="spatial",
        z_key="z",
        label_key="cell_type",
        hidden_size=32,
        depth=2,
        num_heads=2,
        patch_size=4,
        num_workers=0,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(stack)
    volume = recon.reconstruct_continuous_volume(stack, num_depths=3, n_samples=16)

    # The reconstructor writes obs['z_3d'] + obsm['spatial_3d'] — the contract keys.
    assert_volume_schema(volume)
    back = read_volume(write_volume(volume, tmp_path / "recon.h5ad"))
    assert back.n_obs == volume.n_obs
    np.testing.assert_allclose(
        back.obsm[SPATIAL_3D_KEY], volume.obsm[SPATIAL_3D_KEY], rtol=1e-6, atol=1e-6
    )
