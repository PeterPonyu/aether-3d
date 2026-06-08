"""Real-data export round-trip test for 3D volumes (CLAIM_LEDGER row 3).

Complements ``tests/test_volume_io.py`` (synthetic + reconstructor) with the
missing **real-data** gate: a volume assembled from the real openST/HNSCC
GSE251926 leave-one-out reconstruction survives the ``volume_io`` write/read
contract losslessly and stays Scanpy-compatible.

The 1.55 GB ``reconstructed_volume.npz`` is a gitignored local artifact (results/*
is not committed), so this test SKIPS when it is absent — e.g. in CI. Run the
full-volume evidence pass with ``python -m scripts.e2e.export_roundtrip_real``.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.e2e.export_roundtrip_real import NPZ, build_real_volume, run_roundtrip

pytestmark = pytest.mark.skipif(
    not NPZ.exists(),
    reason=f"real reconstruction artifact absent (gitignored): {NPZ}",
)


def test_real_openst_volume_roundtrips_losslessly(tmp_path) -> None:
    # Subsample for a light test; the CLI runs the full real volume for evidence.
    vol = build_real_volume(max_holdouts=4, cells_per_slice=600, seed=0)
    assert vol.n_obs == 4 * 600
    assert vol.obsm["spatial_3d"].shape == (vol.n_obs, 3)
    # Multiple distinct ordinal depths -> a genuine multi-z volume, not a flat slice.
    assert np.unique(vol.obs["z_3d"]).size >= 4

    evidence = run_roundtrip(vol, tmp_path)

    assert evidence["roundtrip_lossless"] is True
    assert evidence["max_abs_dX"] == 0.0
    assert evidence["max_abs_dXYZ"] == 0.0
    assert evidence["scanpy_pca_ok"] is True
    assert evidence["schema_holds_after_scanpy"] is True
