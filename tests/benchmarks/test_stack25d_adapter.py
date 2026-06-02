"""Tests for the clean-room 2.5D stacking reconstruction adapter (#221).

The adapter is a license-clean, brand-independent 2.5D virtual-slice baseline
(SpatialZ-equivalent in role, NOT a port of any published source). Unlike the
former ``SpatialZAdapter`` stub — whose availability check always returned
False and was never wired into the holdout contrast — this baseline:

  * reports available with no external dependency,
  * produces a reconstruction of the expected shape on small synthetic slices,
  * honours the resolved physical z (uses ``inp.truth_z_values()``, not idx*10),
  * is selectable in the contrast path of ``validate_holdout_slice``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from aether_3d.benchmarks import VolumeAdapterInput
from aether_3d.benchmarks.adapters import Stack25DAdapter


def _make_synthetic_slice(z: float, n_cells: int = 25, n_genes: int = 12, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed + int(z))
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    a.obsm["spatial"] = coords
    a.obs["z"] = float(z)
    a.obs["cell_type"] = ["A" if i % 2 == 0 else "B" for i in range(n_cells)]
    return a


def _stack(zs: list[float], seed: int = 0) -> list[ad.AnnData]:
    return [_make_synthetic_slice(z, seed=seed) for z in zs]


def test_stack25d_reports_available_without_external_dep():
    ok, reason = Stack25DAdapter().is_available()
    assert ok, f"clean-room 2.5D baseline must be always-available; got {reason!r}"


def test_stack25d_produces_reconstruction_of_expected_shape():
    inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
    res = Stack25DAdapter().run(inp)
    assert res.status == "ok", res.status
    vol = res.volume_h5ad
    assert vol is not None
    # one held-out depth ⇒ exactly one virtual slice's worth of cells
    assert vol.n_obs > 0
    assert vol.n_vars == 12
    assert "spatial" in vol.obsm and vol.obsm["spatial"].shape[1] == 2
    assert "spatial_3d" in vol.obsm and vol.obsm["spatial_3d"].shape[1] == 3
    assert "z" in vol.obs


def test_stack25d_honours_resolved_physical_z_not_idx_times_ten():
    # Held-out slice sits at a non-idx*10 physical z; the virtual cells must be
    # stamped with the resolved truth z, never an idx*10 fallback.
    slices = _stack([0.0, 3.7, 9.1])
    inp = VolumeAdapterInput(slices=slices, held_out_indices=[1])
    z_target = inp.truth_z_values()[0]
    assert abs(z_target - 3.7) < 1e-5
    vol = Stack25DAdapter().run(inp).volume_h5ad
    assert vol is not None
    zs = np.unique(np.asarray(vol.obs["z"], dtype=np.float32))
    assert np.allclose(zs, z_target, atol=1e-4), (
        f"virtual cells must carry resolved z {z_target}, got {zs}"
    )
    # 3D coords' z column must match too.
    assert np.allclose(vol.obsm["spatial_3d"][:, 2], z_target, atol=1e-4)


def test_stack25d_inherits_real_cell_identity():
    # Identity-preserving: each virtual cell copies a real measured expression
    # vector from a visible slice (no gene-wise blending), so every row of the
    # output matches some row of a visible slice exactly.
    slices = _stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=slices, held_out_indices=[1])
    vol = Stack25DAdapter().run(inp).volume_h5ad
    assert vol is not None
    visible_rows = np.vstack([np.asarray(s.X, dtype=np.float32) for s in inp.visible_slices()])
    out_rows = np.asarray(vol.X, dtype=np.float32)
    for row in out_rows:
        matches = np.all(np.isclose(visible_rows, row, atol=1e-4), axis=1)
        assert matches.any(), "every virtual cell must copy a real visible-cell expression vector"


def test_stack25d_selectable_in_holdout_contrast_path():
    # The contrast loop in validate_holdout_slice must include the clean-room
    # 2.5D baseline by name.
    import scripts.e2e.validate_holdout_slice as vh

    slices = _stack([0.0, 1.0, 2.0])
    for s in slices:
        s.obs["z_coord"] = float(s.obs["z"].iloc[0])
        s.obs["cell_class"] = s.obs["cell_type"]
    held_idx = 1
    held = slices[held_idx]

    # virtual_slice stand-in: reuse a neighbour as a trivial "continuous" recon.
    virtual = slices[0].copy()
    contrast = vh.evaluate_25d_contrast(slices, held_idx, virtual, held, seed=0)
    assert "continuous" in contrast
    # The clean-room 2.5D baseline must appear as a contrast method.
    assert Stack25DAdapter().name in contrast, (
        f"clean-room 2.5D baseline must be wired into the contrast; got {list(contrast)}"
    )
    sc = contrast[Stack25DAdapter().name]
    assert "chamfer_distance" in sc or "status" in sc
