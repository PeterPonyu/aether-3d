"""Regression: the openST/HNSCC (GSE251926) processed h5ad is consumable by the
holdout pipeline as a SECOND real serial-3D dataset (issues #294, #224).

The full processed h5ad (~8.9 GB) is not checked in, so every test skips cleanly
when it is absent. When present, they assert the contract the data card promises:
``load_serial_h5ad`` yields schema-valid per-section slices carrying RAW INTEGER
counts (from ``layers['raw']``), ``obsm['spatial']`` and ``obs['cell_class']``,
and that ``SerialSliceTrajectoryDataset`` constructs without ``ValueError``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENST_H5AD = (
    REPO_ROOT.parent
    / "data"
    / "processed"
    / "openst_hnscc_gse251926"
    / "serial_sections.h5ad"
)

pytestmark = pytest.mark.skipif(
    not OPENST_H5AD.exists(),
    reason=f"openST processed h5ad not present at {OPENST_H5AD}",
)


def _load_vhs():
    path = REPO_ROOT / "scripts" / "e2e" / "validate_holdout_slice.py"
    spec = importlib.util.spec_from_file_location("vhs_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_openst_loader_yields_schema_valid_integer_count_slices():
    vhs = _load_vhs()
    slices, paths, n_dropped, n_shared, sections, z_uniform = vhs.load_serial_h5ad(
        str(OPENST_H5AD), n_hvg=128, max_cells=200, section_window=(0, 3), seed=42
    )
    assert len(slices) == 3
    assert n_shared == 128 and n_dropped > 0
    assert [str(s) for s in sections]  # ordered, non-empty section ids
    for s in slices:
        x = np.asarray(s.X)
        assert np.all(x == np.round(x)), "X must be raw integer counts (layers['raw'])"
        assert "spatial" in s.obsm and s.obsm["spatial"].shape[1] == 2
        assert "cell_class" in s.obs and "z_coord" in s.obs


def test_openst_constructs_serial_dataset_without_valueerror():
    """The acceptance gate from issue #294."""
    vhs = _load_vhs()
    from aether_3d.config.aether_config import Aether3DConfig
    from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset

    slices, *_ = vhs.load_serial_h5ad(
        str(OPENST_H5AD), n_hvg=128, max_cells=200, section_window=(0, 3), seed=42
    )
    cfg = Aether3DConfig(seed=42, n_samples_base=40)
    ds = SerialSliceTrajectoryDataset(slices, cfg)  # must not raise
    assert len(ds) > 0
    assert len(ds.label_encoder.classes_) >= 2
