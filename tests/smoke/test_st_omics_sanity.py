"""Smoke / sanity tests using *very small* ST-omics data in real format.

Uses poisson integer counts + obsm['spatial'] + obs['z_coord'/'cell_class']
(not pure scRNA-style gaussian or normalized expression, no spatial coords).

These guard the aether_3d_serial_slice contract and raw-count expectations
from data cards / fetchers / recent #181 work. Keep n small so they run in <1s
with no GPU and no external downloads.
"""

from __future__ import annotations

import numpy as np

import anndata as ad

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.data.raw_counts import verify_raw_counts
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset


def _tiny_st_slices(n_slices: int = 2, n_spots: int = 8, n_genes: int = 6, seed: int = 42) -> list[ad.AnnData]:
    """Factory for tiny realistic ST serial slices following real data format."""
    rng = np.random.default_rng(seed)
    slices: list[ad.AnnData] = []
    for z in range(n_slices):
        X = rng.poisson(lam=1.7, size=(n_spots, n_genes)).astype(np.float32)
        obs = {
            "cell_class": (["A", "B"] * (n_spots // 2 + 1))[:n_spots],
            "z_coord": [float(z) * 10.0] * n_spots,
        }
        a = ad.AnnData(X=X, obs=obs)
        a.obsm["spatial"] = rng.uniform(low=0.0, high=200.0, size=(n_spots, 2)).astype(np.float32)
        slices.append(a)
    return slices


def test_tiny_st_data_is_raw_counts():
    """The tiny maker itself must produce raw-count-like .X (sanity for the fixture)."""
    slices = _tiny_st_slices()
    for i, a in enumerate(slices):
        check = verify_raw_counts(a.X)
        assert check.is_raw, f"slice {i} failed raw check: {check.reason}"
        assert a.obsm["spatial"].shape == (a.n_obs, 2)
        assert "z_coord" in a.obs and "cell_class" in a.obs


def test_serial_dataset_accepts_tiny_st_format():
    """SerialSliceTrajectoryDataset must accept and process proper small ST data without error."""
    slices = _tiny_st_slices(n_slices=2, n_spots=10, n_genes=5, seed=7)
    cfg = Aether3DConfig(
        patch_size=4,
        n_samples_base=4,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        seed=7,
    )
    ds = SerialSliceTrajectoryDataset(slices, cfg)
    assert len(ds) > 0
    item = ds[0]
    # Expected keys from __getitem__
    for k in ("x0", "g0", "c0", "z0", "x1", "g1", "c1", "z1"):
        assert k in item, f"missing key {k} in trajectory item"


def test_tiny_st_roundtrip_h5ad_preserves_format(tmp_path):
    """Write/read tiny ST via h5ad must preserve raw counts + spatial (mimics card data load)."""
    slices = _tiny_st_slices(n_slices=1, n_spots=5, n_genes=4, seed=99)
    p = tmp_path / "tiny_st.h5ad"
    slices[0].write_h5ad(p)
    reloaded = ad.read_h5ad(p)
    assert reloaded.X.dtype.kind == "f"  # float32 storage ok for small poisson
    check = verify_raw_counts(reloaded.X)
    assert check.is_raw
    assert "spatial" in reloaded.obsm
    assert "z_coord" in reloaded.obs
