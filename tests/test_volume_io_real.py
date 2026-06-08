"""Real-data export round-trip test for 3D volumes (CLAIM_LEDGER row 3).

Complements ``tests/test_volume_io.py`` (synthetic + reconstructor) with the
missing **real-data** gate: volumes assembled from real leave-one-out
reconstructions survive the ``volume_io`` write/read contract losslessly and stay
Scanpy-compatible. Two distinct real datasets are covered (platform / tissue /
gene-panel diversity) — openST/HNSCC and MERFISH mouse-hypothalamus.

The reconstruction npz files are gitignored local artifacts (results/* is not
committed), so each case SKIPS when its artifact is absent — e.g. in CI. Run the
full-volume evidence pass with ``python -m scripts.e2e.export_roundtrip_real``.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.e2e.export_roundtrip_real import DATASETS, build_real_volume, run_roundtrip


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_real_volume_roundtrips_losslessly(dataset, tmp_path) -> None:
    spec = DATASETS[dataset]
    if not spec["npz"].exists():
        pytest.skip(f"real reconstruction artifact absent (gitignored): {spec['npz']}")

    # Subsample for a light test; the CLI runs the full real volume for evidence.
    vol = build_real_volume(
        npz_path=spec["npz"],
        max_holdouts=3,
        cells_per_slice=500,
        seed=0,
        z_resolver=spec["z_resolver"],
    )
    assert vol.obsm["spatial_3d"].shape == (vol.n_obs, 3)
    # Multiple distinct depths -> a genuine multi-z volume, not a flat slice.
    assert np.unique(vol.obs["z_3d"]).size >= 2

    evidence = run_roundtrip(
        vol, tmp_path, dataset=dataset, source=spec["source"], z_kind=spec["z_kind"]
    )

    assert evidence["roundtrip_lossless"] is True
    assert evidence["max_abs_dX"] == 0.0
    assert evidence["max_abs_dXYZ"] == 0.0
    assert evidence["scanpy_pca_ok"] is True
    assert evidence["schema_holds_after_scanpy"] is True
