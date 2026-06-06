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

import pytest

from aether_3d.benchmarks import (
    VolumeAdapterInput,
    VolumeBaseAdapter,
    aggregate_volume_results,
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


def test_metric_failure_inside_adapter_boundary_returns_error_status():
    """Regression for issue #29: metric/schema errors raised AFTER _reconstruct
    must be caught by the adapter failure boundary and reported as
    status="error:..." instead of escaping ``VolumeBaseAdapter.run``.
    """

    class BadVolume(VolumeBaseAdapter):
        name = "bad-volume"

        def _reconstruct(self, visible, inp):
            # Returns a volume that lacks obsm['spatial'] / obsm['spatial_3d'],
            # which will trip compute_volume_metrics() (KeyError 'spatial').
            v = ad.AnnData(X=np.ones((1, 1), dtype=np.float32))
            v.obs["z"] = [0.0]
            return v

    stack = [ad.AnnData(X=np.ones((1, 1), dtype=np.float32))]
    stack[0].obs["z"] = [0.0]
    stack[0].obsm["spatial"] = np.zeros((1, 2), dtype=np.float32)
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[0])

    # Must not raise — failure boundary now wraps metric computation too.
    result = BadVolume().run(inp)

    assert result.status.startswith("error:"), result.status
    assert result.volume_h5ad is None
    assert result.metrics_json == {}


def test_chamfer_uses_nearest_neighbor_without_pairwise_materialization(monkeypatch):
    from aether_3d.benchmarks import contract

    def fail_pairwise(*args, **kwargs):
        raise AssertionError("_pairwise_sq should not be used for nearest-neighbor metrics")

    monkeypatch.setattr(contract, "_pairwise_sq", fail_pairwise)
    a = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    assert contract._chamfer_distance(a, b) == 1.0
    assert contract._coord_rmse(a, b) == 1.0


def test_volume_adapter_input_rejects_out_of_range_held_out_index():
    """Regression for issue #30: VolumeAdapterInput must validate held-out
    indices at construction time so misconfigured holdouts fail closed
    instead of silently passing visible_slices() and crashing in scoring.
    """
    s = ad.AnnData(X=np.ones((1, 1), dtype=np.float32))
    s.obs["z"] = [0.0]

    with pytest.raises(ValueError, match="out-of-range"):
        VolumeAdapterInput(slices=[s], held_out_indices=[99])

    with pytest.raises(ValueError, match="out-of-range"):
        VolumeAdapterInput(slices=[s], held_out_indices=[-1])


def test_volume_adapter_input_rejects_duplicate_held_out_index():
    s = ad.AnnData(X=np.ones((1, 1), dtype=np.float32))
    s.obs["z"] = [0.0]
    s.obsm["spatial"] = np.zeros((1, 2), dtype=np.float32)
    stack = [s, s.copy()]

    with pytest.raises(ValueError, match="duplicate"):
        VolumeAdapterInput(slices=stack, held_out_indices=[0, 0])


def test_volume_adapter_input_accepts_valid_held_out_indices():
    s = ad.AnnData(X=np.ones((1, 1), dtype=np.float32))
    s.obs["z"] = [0.0]
    s.obsm["spatial"] = np.zeros((1, 2), dtype=np.float32)

    # Should not raise
    inp = VolumeAdapterInput(slices=[s], held_out_indices=[0])
    assert inp.held_out_indices == [0]


# -- z-window fix (auto-derived from inter-slice spacing) ------------------


def test_z_window_derived_from_tight_spacing_collects_fewer_cells():
    """With Bregma-like tight spacing (~0.04 mm), the auto-derived z-window
    (0.5 * 0.04 = 0.02 mm) must collect far fewer virtual cells than the
    old hardcoded ±0.5 window would have.
    """
    from aether_3d.benchmarks.contract import compute_volume_metrics

    # Four slices at 0.00, 0.04, 0.08, 0.12 mm (Bregma-like spacing).
    z_vals = [0.00, 0.04, 0.08, 0.12]
    stack = _make_synthetic_stack(z_vals)
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[2])  # truth at 0.08 mm

    # Virtual volume: 200 cells uniformly spread over z ∈ [0, 0.12] mm.
    rng = np.random.default_rng(7)
    n_cells = 200
    v = ad.AnnData(X=rng.poisson(2.0, size=(n_cells, 12)).astype(np.float32))
    v.var_names = [f"GENE_{i:03d}" for i in range(12)]
    v.obsm["spatial"] = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    v.obs["z"] = np.linspace(0.00, 0.12, n_cells).tolist()
    v.obs["cell_type"] = ["A"] * n_cells

    result = compute_volume_metrics(volume=v, inp=inp)
    per = result["per_holdout_slice"]
    assert len(per) == 1, "expected exactly one per-slice entry for held-out z=0.08"

    n_virtual = per[0].get("n_virtual", 0)

    # Old ±0.5 window would have collected ALL 200 cells (span is only 0.12 mm).
    # New auto-derived window: 0.5 * 0.04 = 0.02 mm.
    # Cells within ±0.02 of 0.08 cover 0.04/0.12 ≈ 1/3 of the range → ~67 cells.
    assert n_virtual < n_cells, (
        f"auto-derived z-window should exclude cells far from z=0.08; "
        f"got {n_virtual}/{n_cells}"
    )
    # Must collect less than half — clearly narrower than the old ±0.5 blanket.
    assert n_virtual < n_cells // 2, (
        f"auto-derived window (±0.02 mm) should collect less than half of "
        f"{n_cells} cells spread over 0.12 mm; got {n_virtual}"
    )


def test_z_window_explicit_override_is_respected():
    """Explicit z_window= overrides the auto-derived value."""
    from aether_3d.benchmarks.contract import compute_volume_metrics

    # Stack uses n_genes=12 by default (_make_synthetic_slice default).
    stack = _make_synthetic_stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    n_genes = 12  # must match the truth slices

    # Build a volume with cells spread across z ∈ [0, 2] so the window matters.
    rng = np.random.default_rng(3)
    n = 60
    v = ad.AnnData(X=rng.poisson(2.0, size=(n, n_genes)).astype(np.float32))
    v.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    v.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    v.obs["z"] = np.linspace(0.0, 2.0, n).tolist()
    v.obs["cell_type"] = ["A"] * n

    # Wide explicit window: collects most/all cells near truth z=1.0.
    res_wide = compute_volume_metrics(volume=v, inp=inp, z_window=10.0)
    n_wide = res_wide["per_holdout_slice"][0]["n_virtual"]
    assert n_wide == n, f"z_window=10.0 should collect all {n} cells; got {n_wide}"

    # Narrow explicit window (0.1): collects only cells very close to z=1.0.
    res_narrow = compute_volume_metrics(volume=v, inp=inp, z_window=0.1)
    per_narrow = res_narrow["per_holdout_slice"]
    assert len(per_narrow) == 1
    n_narrow = per_narrow[0].get("n_virtual", 0)
    assert n_narrow < n_wide, (
        f"z_window=0.1 should collect fewer cells than z_window=10; "
        f"got narrow={n_narrow}, wide={n_wide}"
    )

    # Vanishingly small window yields 0 virtual cells (strict < comparison).
    res_zero = compute_volume_metrics(volume=v, inp=inp, z_window=1e-9)
    per_zero = res_zero["per_holdout_slice"]
    if per_zero:
        n_zero = per_zero[0].get("n_virtual", 0)
        assert n_zero == 0 or per_zero[0].get("error") == "no_virtual_cells_at_z"


def test_compute_volume_metrics_reports_mean_domain_ami():
    """mean_domain_ami must be present in the aggregate keys after fix #277."""
    stack = _make_synthetic_stack([0.0, 1.0, 2.0])
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    result = NearestSliceAdapter().run(inp)
    assert result.status == "ok"
    assert "mean_domain_ami" in result.metrics_json, (
        f"mean_domain_ami missing; keys: {list(result.metrics_json)}"
    )
    per = result.metrics_json["per_holdout_slice"]
    assert len(per) == 1
    assert "domain_ami" in per[0], (
        f"domain_ami missing from per-slice entry; keys: {list(per[0])}"
    )
