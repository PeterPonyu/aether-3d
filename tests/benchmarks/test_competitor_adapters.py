"""Tests for the 3d-OT, ASIGN, and InterpolAI competitor adapters.

Each adapter: unavailable-when-not-installed + mocked-success-with-audit.
"""

from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import scanpy as sc

from aether_3d.benchmarks import VolumeAdapterInput
from aether_3d.benchmarks.adapters import (
    ASIGNAdapter,
    InterpolAIAdapter,
    ThreeDOTAdapter,
)


def _make_synthetic_slice(z: float, n_cells: int = 25, n_genes: int = 12, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed + int(z))
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    a.obsm["spatial"] = coords
    a.obs["z"] = float(z)
    return a


def _stack(zs: list[float], seed: int = 0) -> list[ad.AnnData]:
    return [_make_synthetic_slice(z, seed=seed) for z in zs]


def _clear(*names: str) -> None:
    for n in names:
        sys.modules.pop(n, None)


# -- 3d-OT -----------------------------------------------------------------


def test_three_d_ot_unavailable_when_not_installed():
    _clear("three_d_ot", "ot3d", "threed_ot", "ThreeDOT")
    inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
    res = ThreeDOTAdapter().run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "three-d-ot-not-installed" in res.status


def test_three_d_ot_runs_with_mocked_module_and_audit_holds():
    seen: dict = {}
    fake = types.ModuleType("three_d_ot")

    class _Fake:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, slices):
            seen["z"] = sorted(float(s.obs["z"].iloc[0]) for s in slices)

        def reconstruct(self, virtual_z):
            n = len(virtual_z)
            v = ad.AnnData(X=np.zeros((n, 12), dtype=np.float32))
            v.var_names = [f"GENE_{i:03d}" for i in range(12)]
            v.obsm["spatial"] = np.zeros((n, 2), dtype=np.float32)
            v.obsm["spatial_3d"] = np.hstack(
                [np.zeros((n, 2), dtype=np.float32), np.asarray(virtual_z, dtype=np.float32).reshape(-1, 1)]
            )
            v.obs["z"] = list(virtual_z)
            return v

    fake.ThreeDOT = _Fake  # type: ignore[attr-defined]
    sys.modules["three_d_ot"] = fake
    try:
        inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
        res = ThreeDOTAdapter().run(inp)
        assert res.status == "ok", res.status
        assert seen["z"] == [0.0, 2.0]
    finally:
        _clear("three_d_ot")


# -- ASIGN ------------------------------------------------------------------


def test_asign_unavailable_when_module_missing():
    _clear("asign", "ASIGN")
    inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
    res = ASIGNAdapter(wsi_stack=[np.zeros((10, 10))]).run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "asign-not-installed" in res.status


def test_asign_unavailable_when_wsi_stack_missing():
    fake = types.ModuleType("asign")

    class _Fake:
        def __init__(self, device: str = "cpu"):
            pass

        def fit(self, visible_slices, wsi_stack):
            pass

        def impute(self, virtual_z):
            return None

    fake.ASIGN = _Fake  # type: ignore[attr-defined]
    sys.modules["asign"] = fake
    try:
        inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
        res = ASIGNAdapter(wsi_stack=None).run(inp)
        assert res.status.startswith("unavailable:"), res.status
        assert "asign-requires-wsi-stack" in res.status
    finally:
        _clear("asign")


def test_asign_runs_with_mocked_module_and_audit_holds():
    seen: dict = {}
    fake = types.ModuleType("asign")

    class _Fake:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, visible_slices, wsi_stack):
            seen["visible_z"] = sorted(float(s.obs["z"].iloc[0]) for s in visible_slices)
            seen["wsi_len"] = len(wsi_stack)

        def impute(self, virtual_z):
            n = len(virtual_z)
            v = ad.AnnData(X=np.zeros((n, 12), dtype=np.float32))
            v.var_names = [f"GENE_{i:03d}" for i in range(12)]
            v.obsm["spatial"] = np.zeros((n, 2), dtype=np.float32)
            v.obsm["spatial_3d"] = np.zeros((n, 3), dtype=np.float32)
            v.obs["z"] = list(virtual_z)
            return v

    fake.ASIGN = _Fake  # type: ignore[attr-defined]
    sys.modules["asign"] = fake
    try:
        inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
        res = ASIGNAdapter(wsi_stack=[np.zeros((10, 10)), np.zeros((10, 10))]).run(inp)
        assert res.status == "ok", res.status
        assert seen["visible_z"] == [0.0, 2.0]
        assert seen["wsi_len"] == 2
    finally:
        _clear("asign")


# -- InterpolAI -------------------------------------------------------------


def test_interpolai_unavailable_when_not_installed():
    _clear("interpolai", "InterpolAI", "interpol_ai")
    inp = VolumeAdapterInput(slices=_stack([0.0, 1.0, 2.0]), held_out_indices=[1])
    res = InterpolAIAdapter().run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "interpolai-not-installed" in res.status


def test_interpolai_runs_with_mocked_module_and_returns_volume():
    seen: dict = {}
    fake = types.ModuleType("interpolai")

    def fake_interpolate(frames, n_intermediate):
        seen["frames_shape"] = frames.shape
        seen["n_intermediate"] = n_intermediate
        # Trivial: return the frames unchanged
        return frames

    fake.interpolate = fake_interpolate
    sys.modules["interpolai"] = fake
    try:
        inp = VolumeAdapterInput(
            slices=_stack([0.0, 1.0, 2.0]),
            held_out_indices=[1],
        )
        res = InterpolAIAdapter(grid_size=16).run(inp)
        assert res.status == "ok", res.status
        assert res.volume_h5ad is not None
        # Frames shape: (2 visible slices, n_genes=12, 16, 16)
        assert seen["frames_shape"] == (2, 12, 16, 16)
        assert seen["n_intermediate"] >= 1
    finally:
        _clear("interpolai")
