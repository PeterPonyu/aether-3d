"""Tests for the UOT-ablation real-data path (#227).

``scripts/ci/run_uot_ablation.py`` historically ran synthetic-only. It now
accepts ``--real-data``/``--data-dir`` to run the (alpha_spatial, lambda_class)
cost-matrix ablation on real cached MERFISH slices, while keeping synthetic the
default. These tests spy/patch the real loader so no real data is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.ci.run_uot_ablation as ablation


def test_synthetic_is_default(tmp_path, monkeypatch):
    """Without --real-data the script uses the synthetic paired-slice path."""
    called = {"real": False}

    def _fake_real_loader(*a, **k):  # pragma: no cover - must NOT be hit
        called["real"] = True
        raise AssertionError("real loader must not run on the synthetic default")

    monkeypatch.setattr(ablation, "load_real_paired_slices", _fake_real_loader)
    out = tmp_path / "syn.json"
    rc = ablation.main(["--grid", "coarse", "--out", str(out), "--n-cells", "15"])
    assert rc == 0
    assert called["real"] is False
    payload = json.loads(out.read_text())
    assert payload["n_points"] == 9  # 3 alphas x 3 lambdas


def test_real_data_routes_to_loader(tmp_path, monkeypatch):
    """--real-data --data-dir routes through load_real_paired_slices (spied)."""
    seen = {}

    def _fake_real_loader(data_dir, slice_a, slice_b, max_cells, seed):
        seen["data_dir"] = str(data_dir)
        seen["pair"] = (slice_a, slice_b)
        seen["max_cells"] = max_cells
        rng = np.random.default_rng(seed)
        n = 12
        s0 = {
            "x": rng.uniform(0, 100, (n, 2)).astype(np.float32),
            "g": rng.normal(0, 1, (n, 6)).astype(np.float32),
            "c": np.eye(n, dtype=np.float32),
        }
        perm = rng.permutation(n)
        s1 = {"x": s0["x"][perm], "g": s0["g"][perm], "c": s0["c"][perm]}
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(n)
        return s0, s1, inverse

    monkeypatch.setattr(ablation, "load_real_paired_slices", _fake_real_loader)
    out = tmp_path / "real.json"
    rc = ablation.main([
        "--real-data",
        "--data-dir", str(tmp_path / "merfish"),
        "--grid", "coarse",
        "--out", str(out),
    ])
    assert rc == 0
    assert seen["data_dir"] == str(tmp_path / "merfish")
    payload = json.loads(out.read_text())
    assert payload["n_points"] == 9
    # data_source provenance recorded as real
    assert payload.get("data_source") == "real"


def test_real_data_requires_data_dir():
    """--real-data without --data-dir is an error."""
    with pytest.raises(SystemExit):
        ablation.main(["--real-data"])


def test_load_real_paired_slices_reads_cached_h5ad(tmp_path):
    """The real loader reads two cached slices, intersects panels, and returns
    the (s0, s1, perm) triple the ablation runner expects."""
    import anndata as ad

    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(8)]
    for idx in range(2):
        n = 30
        a = ad.AnnData(X=rng.poisson(2.0, (n, 8)).astype(np.float32))
        a.var_names = genes
        a.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
        a.obs["cell_class"] = ["A" if i % 2 else "B" for i in range(n)]
        a.write(tmp_path / f"merfish_{idx}.h5ad")

    s0, s1, perm = ablation.load_real_paired_slices(
        str(tmp_path), slice_a=0, slice_b=1, max_cells=20, seed=0
    )
    for s in (s0, s1):
        assert set(s) == {"x", "g", "c"}
        assert s["x"].shape[1] == 2
        assert s["x"].shape[0] <= 20
    assert len(perm) == s0["x"].shape[0]
