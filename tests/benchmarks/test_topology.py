"""Tests for topology + vector-field metrics (Betti / divergence / anisotropy)."""

from __future__ import annotations

import numpy as np
import pytest

from aether_3d.benchmarks.topology import (
    betti_zero,
    betti_zero_stability,
    divergence_summary,
    flow_divergence_map,
    topology_metrics,
    velocity_anisotropy,
)


# -- Betti-0 ------------------------------------------------------------


def test_betti_zero_single_dense_blob_is_one_component():
    """A tight blob of points should be one connected component."""
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 0.1, size=(40, 2)).astype(np.float32)
    assert betti_zero(coords, k=4) == 1


def test_betti_zero_two_far_apart_blobs_is_two_components():
    """Two well-separated blobs should report 2 components."""
    rng = np.random.default_rng(0)
    blob1 = rng.normal(0, 0.1, size=(20, 2)).astype(np.float32)
    blob2 = rng.normal(100, 0.1, size=(20, 2)).astype(np.float32)
    coords = np.vstack([blob1, blob2])
    assert betti_zero(coords, k=4) == 2


def test_betti_zero_empty_returns_zero():
    assert betti_zero(np.zeros((0, 2), dtype=np.float32)) == 0


def test_betti_zero_single_point_is_one_component():
    assert betti_zero(np.array([[1.0, 2.0]], dtype=np.float32)) == 1


def test_betti_zero_stability_identical_clouds_is_one():
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1, size=(30, 2)).astype(np.float32)
    s = betti_zero_stability(coords, coords.copy(), k=4)
    assert s == pytest.approx(1.0)


def test_betti_zero_stability_diverges_when_clouds_differ_in_topology():
    """One cloud has 1 component, the other has 2. Stability should drop."""
    rng = np.random.default_rng(0)
    one_blob = rng.normal(0, 0.1, size=(40, 2)).astype(np.float32)
    two_blobs = np.vstack([
        rng.normal(0, 0.1, size=(20, 2)),
        rng.normal(100, 0.1, size=(20, 2)),
    ]).astype(np.float32)
    s = betti_zero_stability(one_blob, two_blobs, k=4)
    assert s == pytest.approx(0.5)


# -- Flow divergence ----------------------------------------------------


def test_flow_divergence_uniform_field_is_near_zero():
    """A spatially-uniform velocity field has zero divergence everywhere."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(200, 2)).astype(np.float32)
    velocities = np.tile(np.array([1.5, -0.3], dtype=np.float32), (200, 1))
    div = flow_divergence_map(coords, velocities, grid_size=8)
    summary = divergence_summary(div)
    assert summary["mean_abs_divergence"] < 1e-3, summary


def test_flow_divergence_radial_field_has_positive_mean_divergence():
    """A radially-outward field has divergence > 0 (source pattern)."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(-10, 10, size=(400, 2)).astype(np.float32)
    velocities = coords.copy()  # v(x) = x → ∇·v = 2 in 2D
    div = flow_divergence_map(coords, velocities, grid_size=12)
    summary = divergence_summary(div)
    assert summary["rms_divergence"] > 0.5, summary
    # Mean (signed) divergence of a radial source should be positive
    finite = div[np.isfinite(div)]
    assert finite.mean() > 0


def test_flow_divergence_handles_empty():
    empty = np.zeros((0, 2), dtype=np.float32)
    div = flow_divergence_map(empty, empty, grid_size=4)
    summary = divergence_summary(div)
    assert np.isnan(summary["mean_abs_divergence"])
    assert np.isnan(summary["rms_divergence"])


def test_flow_divergence_shape_mismatch_raises():
    coords = np.zeros((10, 2), dtype=np.float32)
    vel = np.zeros((9, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        flow_divergence_map(coords, vel, grid_size=4)


# -- Velocity anisotropy -----------------------------------------------


def test_velocity_anisotropy_isotropic_field_is_near_one():
    rng = np.random.default_rng(0)
    velocities = rng.normal(0, 1, size=(500, 2)).astype(np.float32)
    a = velocity_anisotropy(velocities)
    assert 0.7 < a < 1.5, f"isotropic field should give ratio near 1, got {a}"


def test_velocity_anisotropy_strongly_directional_field_is_large():
    """If all velocity vectors point along x, anisotropy → very large."""
    rng = np.random.default_rng(0)
    velocities = np.column_stack([
        rng.normal(5.0, 2.0, size=500),  # large variance along x
        rng.normal(0.0, 0.05, size=500),  # tiny variance along y
    ]).astype(np.float32)
    a = velocity_anisotropy(velocities)
    assert a > 100.0, f"directional field should give large ratio, got {a}"


def test_velocity_anisotropy_too_few_samples_is_nan():
    assert np.isnan(velocity_anisotropy(np.zeros((1, 2), dtype=np.float32)))
    assert np.isnan(velocity_anisotropy(np.zeros((0, 2), dtype=np.float32)))


# -- Roll-up ---------------------------------------------------------


def test_topology_metrics_returns_all_keys_with_velocities():
    rng = np.random.default_rng(0)
    coords_t = rng.uniform(0, 10, size=(50, 2)).astype(np.float32)
    coords_r = rng.uniform(0, 10, size=(50, 2)).astype(np.float32)
    velocities = rng.normal(0, 1, size=(50, 2)).astype(np.float32)
    m = topology_metrics(coords_t, coords_r, velocities_recon=velocities, grid_size=8)
    for k in (
        "betti0_stability",
        "mean_abs_divergence",
        "max_abs_divergence",
        "rms_divergence",
        "velocity_anisotropy",
    ):
        assert k in m, f"missing key {k}"


def test_topology_metrics_without_velocities_nan_fills():
    rng = np.random.default_rng(0)
    coords_t = rng.uniform(0, 10, size=(30, 2)).astype(np.float32)
    coords_r = rng.uniform(0, 10, size=(30, 2)).astype(np.float32)
    m = topology_metrics(coords_t, coords_r, velocities_recon=None)
    assert not np.isnan(m["betti0_stability"])
    assert np.isnan(m["mean_abs_divergence"])
    assert np.isnan(m["velocity_anisotropy"])
