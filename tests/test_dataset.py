"""Schema-validation regression tests for SerialSliceTrajectoryDataset.

Issue #86 — `SerialSliceTrajectoryDataset.__getitem__` read `obs[z_key]`
(`z_coord` by default), `obs[label_key]`, and `obsm[spatial_key]` per item
with no up-front validation. A slice missing one of these failed deep inside
data loading with a bare `KeyError`, long after `setup_data` "succeeded" and
often after training had started. The fix validates the required schema in
`__init__` and raises one clear `ValueError` naming the missing key and the
offending slice index.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset


def _slice(
    *,
    with_z: bool = True,
    with_label: bool = True,
    with_spatial: bool = True,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    obs: dict[str, list] = {}
    if with_label:
        obs["cell_class"] = ["T", "B", "T", "B"]
    a = ad.AnnData(X=rng.normal(size=(4, 5)).astype("float32"), obs=obs or None)
    if with_z:
        a.obs["z_coord"] = [0.0, 0.0, 0.0, 0.0]
    if with_spatial:
        a.obsm["spatial"] = rng.normal(size=(4, 2)).astype("float32")
    return a


def test_missing_z_coord_clear_error():
    """A slice missing obs['z_coord'] must raise a clear ValueError at
    construction (naming z_coord), not a lazy KeyError during iteration."""
    a0 = _slice(with_z=False)
    a1 = _slice(with_z=True)
    cfg = Aether3DConfig(patch_size=4)

    with pytest.raises(ValueError, match="z_coord"):
        SerialSliceTrajectoryDataset([a0, a1], cfg)


def test_missing_spatial_key_clear_error():
    a0 = _slice(with_spatial=False)
    a1 = _slice()
    cfg = Aether3DConfig(patch_size=4)

    with pytest.raises(ValueError, match="spatial"):
        SerialSliceTrajectoryDataset([a0, a1], cfg)


def test_missing_label_key_clear_error():
    a0 = _slice(with_label=False)
    a1 = _slice()
    cfg = Aether3DConfig(patch_size=4)

    with pytest.raises(ValueError, match="cell_class"):
        SerialSliceTrajectoryDataset([a0, a1], cfg)


def test_valid_slices_construct_without_error():
    """A fully-specified pair must still build trajectories normally."""
    cfg = Aether3DConfig(patch_size=4, n_samples_base=8)
    ds = SerialSliceTrajectoryDataset([_slice(seed=1), _slice(seed=2)], cfg)
    assert len(ds) > 0
    sample = ds[0]
    assert "z0" in sample and "x0" in sample and "c0" in sample
