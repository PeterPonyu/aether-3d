"""Regression tests for issue #222 — physical inter-slice z spacing.

The pipeline previously injected a synthetic ``z_coord = idx * 10.0`` for every
real MERFISH slice, which corrupts all physical-spacing-dependent metrics and
figures (the Moffitt 2018 hypothalamus stack carries a real anterior-posterior
Bregma coordinate, ~0.05 mm spacing, exposed as ``obs['slice_id']`` in the
cached baseline slices). These tests pin the contract:

(a) when ``obs`` carries a physical z/Bregma field, the resolver uses those
    values (NOT idx*10) and reports ``z_is_physical=True``;
(b) when no physical field is present, the resolver falls back to a configurable
    spacing (NOT a hard-coded 10) and reports ``z_is_physical=False``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable when the suite runs from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aether_3d.data.physical_z import resolve_slice_z


def _slice(n=6, z_field=None, z_value=None, n_genes=4):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(n, n_genes)).astype(np.float32))
    a.obsm["spatial"] = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    if z_field is not None:
        a.obs[z_field] = [z_value] * n
    return a


def test_physical_z_from_bregma_obs_field():
    """When obs carries a physical z (Bregma) field, resolve_slice_z must use
    those exact values, not idx*spacing, and flag z_is_physical=True."""
    bregma = [0.04, 0.09, 0.14, 0.19, 0.24]
    slices = [_slice(z_field="Bregma", z_value=b) for b in bregma]

    z, is_physical = resolve_slice_z(slices, fallback_spacing=10.0)

    assert is_physical is True
    np.testing.assert_allclose(z, bregma)
    # It must NOT be the synthetic idx*10 ladder.
    assert not np.allclose(z, [0.0, 10.0, 20.0, 30.0, 40.0])


def test_physical_z_from_slice_id_field():
    """The cached MERFISH baseline stores the Bregma mm coordinate in
    obs['slice_id'] (as a string). resolve_slice_z must parse and use it."""
    slice_ids = ["0.04", "0.09", "0.14", "0.19", "0.24"]
    slices = []
    for sid in slice_ids:
        a = _slice()
        a.obs["slice_id"] = pd.Categorical([sid] * a.n_obs)
        slices.append(a)

    z, is_physical = resolve_slice_z(slices, fallback_spacing=10.0)

    assert is_physical is True
    np.testing.assert_allclose(z, [0.04, 0.09, 0.14, 0.19, 0.24])


def test_synthetic_fallback_sets_flag_false_and_uses_configurable_spacing():
    """With no physical field, the resolver falls back to idx*spacing using the
    CONFIGURABLE spacing (not a hard-coded 10) and flags z_is_physical=False."""
    slices = [_slice() for _ in range(4)]

    with pytest.warns(UserWarning):
        z, is_physical = resolve_slice_z(slices, fallback_spacing=2.5)

    assert is_physical is False
    np.testing.assert_allclose(z, [0.0, 2.5, 5.0, 7.5])


def test_fallback_spacing_is_not_hardcoded_ten():
    """Distinct fallback spacings must yield distinct z ladders (guards against a
    re-introduced hard-coded 10)."""
    slices = [_slice() for _ in range(3)]
    z1, _ = resolve_slice_z(slices, fallback_spacing=1.0)
    z2, _ = resolve_slice_z(slices, fallback_spacing=100.0)
    assert not np.allclose(z1, z2)


_BASELINE_DIR = Path(
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/baselines/serial3d_ref/merfish_mouse_hypothalamus"
)


@pytest.mark.skipif(
    not (_BASELINE_DIR / "merfish_0.h5ad").exists(),
    reason="cached real MERFISH baseline slices not present",
)
def test_real_merfish_loader_uses_physical_bregma_not_idx10():
    """End-to-end (issue #222 CI gate): the holdout loader on REAL MERFISH data
    must inject the physical Bregma mm z (range ~[0.04, 0.24]) and report
    z_is_physical=True — NOT the synthetic idx*10 ladder [0,10,20,30,40]."""
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "e2e" / "validate_holdout_slice.py"
    )
    spec = importlib.util.spec_from_file_location("_vhs", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    slices, _, _, _, z_is_physical = mod.load_real_merfish_slices(
        str(_BASELINE_DIR), n_slices=5, max_cells=200, seed=0
    )
    z = [float(s.obs["z_coord"].iloc[0]) for s in slices]

    assert z_is_physical is True
    # Real Moffitt Bregma mm spacing, NOT idx*10.
    assert max(z) < 1.0, f"z looks synthetic (idx*10), got {z}"
    assert not np.allclose(z, [0.0, 10.0, 20.0, 30.0, 40.0])
