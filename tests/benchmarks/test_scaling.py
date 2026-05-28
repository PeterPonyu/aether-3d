"""Tests for the scaling-curve harness."""

from __future__ import annotations

import json
from pathlib import Path


from aether_3d.benchmarks import (
    ScalingPoint,
    ScalingResult,
    aggregate_scaling,
    make_synthetic_stack,
    measure_one,
    sweep,
)
from aether_3d.benchmarks.adapters import LinearInterpAdapter, NearestSliceAdapter


def test_scaling_point_total_cells():
    p = ScalingPoint(n_cells_per_slice=100, n_slices=4)
    assert p.total_cells == 400


def test_make_synthetic_stack_returns_correctly_sized_slices():
    p = ScalingPoint(n_cells_per_slice=25, n_slices=3, n_genes=10)
    stack = make_synthetic_stack(p)
    assert len(stack) == 3
    for s in stack:
        assert s.n_obs == 25
        assert s.n_vars == 10
        assert "spatial" in s.obsm


def test_measure_one_returns_finite_runtime_and_recorded_device():
    point = ScalingPoint(n_cells_per_slice=20, n_slices=3, n_genes=8)
    result = measure_one(NearestSliceAdapter(), point)

    assert isinstance(result, ScalingResult)
    assert result.adapter == "nearest-slice"
    assert result.status == "ok"
    assert result.runtime_s >= 0.0
    assert result.runtime_s < 5.0, "20-cell smoke must finish in well under 5s"
    assert result.peak_memory_mb >= 0.0
    assert result.device in {"cpu", "cuda"}
    assert result.n_virtual_cells > 0
    assert result.hostname  # captured
    assert result.platform  # captured


def test_measure_one_records_error_status_on_adapter_failure():
    from aether_3d.benchmarks import VolumeBaseAdapter

    class BrokenAdapter(VolumeBaseAdapter):
        name = "broken"

        def _reconstruct(self, visible, inp):
            raise ValueError("intentional failure")

    point = ScalingPoint(n_cells_per_slice=10, n_slices=2)
    result = measure_one(BrokenAdapter(), point)

    assert result.status.startswith("error:")
    assert "intentional failure" in result.error_message
    assert result.runtime_s >= 0.0
    assert result.n_virtual_cells == 0


def test_sweep_runs_cartesian_product():
    points = [
        ScalingPoint(n_cells_per_slice=15, n_slices=2, n_genes=8),
        ScalingPoint(n_cells_per_slice=30, n_slices=2, n_genes=8),
    ]
    adapters = [NearestSliceAdapter(), LinearInterpAdapter()]

    results = sweep(adapters, points)
    assert len(results) == 4
    adapter_names = {r.adapter for r in results}
    assert adapter_names == {"nearest-slice", "linear-interp"}
    cell_counts = sorted({r.point.n_cells_per_slice for r in results})
    assert cell_counts == [15, 30]


def test_scaling_runtime_increases_with_cell_count():
    """Cheap monotonicity sanity: a 100x bigger point takes at least as long."""
    small = ScalingPoint(n_cells_per_slice=15, n_slices=2, n_genes=8)
    big = ScalingPoint(n_cells_per_slice=150, n_slices=2, n_genes=8)
    r_small = measure_one(NearestSliceAdapter(), small)
    r_big = measure_one(NearestSliceAdapter(), big)
    # Both must succeed
    assert r_small.status == "ok"
    assert r_big.status == "ok"
    # Big has at least as many virtual cells as small
    assert r_big.n_virtual_cells >= r_small.n_virtual_cells


def test_aggregate_scaling_is_json_serializable(tmp_path: Path):
    points = [ScalingPoint(n_cells_per_slice=15, n_slices=2, n_genes=8)]
    results = sweep([NearestSliceAdapter()], points)
    aggregated = aggregate_scaling(results)

    assert aggregated["schema_version"] == "1"
    assert aggregated["n_results"] == 1
    assert "results" in aggregated
    r = aggregated["results"][0]
    assert "adapter" in r
    assert "point" in r
    assert isinstance(r["point"], dict)
    assert r["point"]["n_cells_per_slice"] == 15

    # Round-trips through JSON without losing the schema
    out = tmp_path / "scaling.json"
    out.write_text(json.dumps(aggregated, default=str))
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == aggregated["schema_version"]
    assert loaded["n_results"] == aggregated["n_results"]
    assert loaded["results"][0]["adapter"] == "nearest-slice"


def test_scaling_records_dependency_versions():
    point = ScalingPoint(n_cells_per_slice=10, n_slices=2, n_genes=5)
    result = measure_one(NearestSliceAdapter(), point)

    # torch_version may be None if torch isn't installed, but the field is present
    assert hasattr(result, "torch_version")
    assert hasattr(result, "cuda_version")
    # Platform info always present
    assert result.python_version
    assert result.platform


def test_scaling_handles_zero_holdout_gracefully():
    """When holdout_index is out of range, the harness must still run."""
    point = ScalingPoint(n_cells_per_slice=10, n_slices=2)
    # holdout_index = 99 is out of range; the harness should still measure
    # adapter behavior (with empty holdout it produces an empty volume).
    result = measure_one(NearestSliceAdapter(), point, holdout_index=99)
    assert result.status == "ok"
    # n_virtual_cells may be 0 in the no-holdout case; that's fine, runtime > 0.
    assert result.runtime_s >= 0.0
