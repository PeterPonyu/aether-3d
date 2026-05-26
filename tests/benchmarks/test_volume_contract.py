"""Contract tests for the 3D-reconstruction benchmark base.

Verifies comparability, audit-safety (held-out slices removed before adapter),
and provenance — the same three properties LuminaST enforces in 2D.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np

from aether_3d.benchmarks import (
    VolumeAdapterInput,
    VolumeBaseAdapter,
    aggregate_volume_results,
    compute_volume_metrics,
    run_holdout,
    write_volume_results_json,
)
from aether_3d.benchmarks.adapters import (
    LinearInterpAdapter,
    NearestSliceAdapter,
    SpatialZAdapter,
)


def _assert_nan_equal(a, b):
    """Deep equality treating NaN-vs-NaN as equal (JSON round-trip safe)."""
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a) == set(b), f"key mismatch: {set(a)} vs {set(b)}"
        for k in a:
            _assert_nan_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nan_equal(x, y)
    elif isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return
        assert a == b, f"{a} != {b}"
    else:
        assert a == b, f"{a!r} != {b!r}"


def _make_synthetic_slice(z: float, n_cells: int = 30, n_genes: int = 12, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed + int(z))
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    adata.obsm["spatial"] = coords
    adata.obs["z"] = float(z)
    adata.obs["cell_type"] = ["A"] * n_cells
    return adata


def _make_synthetic_stack(z_values: list[float], seed: int = 0) -> list[ad.AnnData]:
    return [_make_synthetic_slice(z, seed=seed) for z in z_values]


# -- VolumeAdapterInput audit boundary --------------------------------------


def test_visible_slices_excludes_held_out():
    stack = _make_synthetic_stack([0.0, 1.0, 2.0, 3.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1, 3])

    visible = inp.visible_slices()
    assert len(visible) == 2
    visible_z = [float(s.obs["z"].iloc[0]) for s in visible]
    assert visible_z == [0.0, 2.0]


def test_truth_slices_returns_held_out_only():
    stack = _make_synthetic_stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    truth = inp.truth_slices()
    assert len(truth) == 1
    assert float(truth[0].obs["z"].iloc[0]) == 1.0


def test_audit_boundary_adapter_does_not_see_held_out_slices():
    captured: dict = {}

    class SnoopAdapter(VolumeBaseAdapter):
        name = "snoop"

        def _reconstruct(self, visible, inp):
            captured["seen_z"] = sorted(float(s.obs["z"].iloc[0]) for s in visible)
            # Reconstruct trivially (just stamp first visible slice at truth z's)
            from aether_3d.benchmarks.adapters import NearestSliceAdapter
            return NearestSliceAdapter()._reconstruct(visible, inp)

    stack = _make_synthetic_stack([0.0, 1.0, 2.0, 3.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1, 3])
    _ = SnoopAdapter().run(inp)

    assert captured["seen_z"] == [0.0, 2.0], \
        "AUDIT FAILURE: adapter saw held-out slice z-values"


# -- Baselines run --------------------------------------------------------


def test_nearest_slice_runs_and_produces_volume_with_3d_coords():
    stack = _make_synthetic_stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    result = NearestSliceAdapter().run(inp)

    assert result.status == "ok", result.status
    assert result.method == "nearest-slice"
    assert result.volume_h5ad is not None
    assert "spatial_3d" in result.volume_h5ad.obsm
    assert result.volume_h5ad.obsm["spatial_3d"].shape[1] == 3
    assert result.metrics_json["n_virtual_cells"] > 0
    assert result.metrics_json["n_truth_slices"] == 1


def test_linear_interp_runs_and_returns_metrics():
    stack = _make_synthetic_stack([0.0, 1.0, 2.0, 3.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[2])
    result = LinearInterpAdapter().run(inp)

    assert result.status == "ok", result.status
    assert "per_holdout_slice" in result.metrics_json
    per = result.metrics_json["per_holdout_slice"]
    assert len(per) == 1
    assert per[0]["n_virtual"] > 0
    # Chamfer is finite (not NaN)
    assert not np.isnan(per[0]["chamfer"])


# -- SpatialZ availability paths ------------------------------------------


def test_spatialz_unavailable_when_not_installed():
    for n in ("spatialz", "SpatialZ", "spatial_z"):
        sys.modules.pop(n, None)
    stack = _make_synthetic_stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])

    result = SpatialZAdapter().run(inp)
    assert result.status.startswith("unavailable:"), result.status
    assert "spatialz-not-installed" in result.status
    assert result.volume_h5ad is None


def test_spatialz_runs_with_mocked_module_and_audit_holds():
    seen: dict = {}
    fake = types.ModuleType("spatialz")

    class _FakeSpatialZ:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, slices):
            seen["fit_z"] = sorted(float(s.obs["z"].iloc[0]) for s in slices)

        def predict(self, virtual_z):
            # Trivial: emit one virtual cell per requested z
            n = len(virtual_z)
            v = ad.AnnData(X=np.zeros((n, 12), dtype=np.float32))
            v.var_names = [f"GENE_{i:03d}" for i in range(12)]
            v.obsm["spatial"] = np.zeros((n, 2), dtype=np.float32)
            v.obsm["spatial_3d"] = np.hstack(
                [np.zeros((n, 2), dtype=np.float32), np.asarray(virtual_z, dtype=np.float32).reshape(-1, 1)]
            )
            v.obs["z"] = list(virtual_z)
            return v

    fake.SpatialZ = _FakeSpatialZ  # type: ignore[attr-defined]
    sys.modules["spatialz"] = fake
    try:
        stack = _make_synthetic_stack([0.0, 1.0, 2.0])
        inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])

        result = SpatialZAdapter().run(inp)
        assert result.status == "ok", result.status
        # Audit: SpatialZ only saw the visible slices' z-values
        assert seen["fit_z"] == [0.0, 2.0]
    finally:
        sys.modules.pop("spatialz", None)


# -- Runner + JSON --------------------------------------------------------


def test_run_holdout_executes_all_adapters_and_aggregates(tmp_path: Path):
    stack = _make_synthetic_stack([0.0, 1.0, 2.0, 3.0])
    adapters = [NearestSliceAdapter(), LinearInterpAdapter(), SpatialZAdapter()]

    results = run_holdout(
        adapters, stack, held_out_indices=[2], seed=7,
        dataset_name="synth-stack",
    )
    aggregated = aggregate_volume_results({("synth-stack", "holdout-z2"): results})

    assert aggregated["schema_version"] == "1"
    key = "synth-stack/holdout-z2"
    assert key in aggregated["holdouts"]
    method_keys = set(aggregated["holdouts"][key])
    assert method_keys == {"nearest-slice", "linear-interp", "spatialz"}
    assert aggregated["holdouts"][key]["nearest-slice"]["status"] == "ok"
    assert aggregated["holdouts"][key]["spatialz"]["status"].startswith("unavailable:")

    out_path = write_volume_results_json(aggregated, tmp_path / "results.json")
    loaded = json.loads(out_path.read_text())
    _assert_nan_equal(loaded, aggregated)
    # Provenance + seed recorded
    prov = loaded["holdouts"][key]["nearest-slice"]["provenance"]
    assert prov["method"] == "nearest-slice"
    assert prov["seed"] == 7
    assert "numpy" in prov["dependency_notes"]


def test_compute_volume_metrics_handles_no_holdout():
    stack = _make_synthetic_stack([0.0, 1.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[])

    result = NearestSliceAdapter().run(inp)
    # With no held-out slices, the volume is empty but the call must not crash.
    assert result.status == "ok"
    assert result.metrics_json["n_truth_slices"] == 0


def test_failing_volume_adapter_returns_error_status():
    class BrokenAdapter(VolumeBaseAdapter):
        name = "broken"

        def _reconstruct(self, visible, inp):
            raise ValueError("intentional failure")

    stack = _make_synthetic_stack([0.0, 1.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    result = BrokenAdapter().run(inp)

    assert result.status.startswith("error:")
    assert "intentional failure" in result.status
