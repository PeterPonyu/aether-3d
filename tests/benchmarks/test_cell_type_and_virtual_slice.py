"""Unit tests for the Round 11 W004/W005 additions to aether_3d.benchmarks.metrics."""

from __future__ import annotations

import math

import anndata as ad
import numpy as np
import pytest

from aether_3d.benchmarks.metrics import (
    celltype_distribution_cosine,
    virtual_slice_at_depth,
)


class TestCellTypeDistributionCosine:
    def test_identical_distributions_score_one(self) -> None:
        truth = ["A", "A", "B", "B", "C"]
        assert celltype_distribution_cosine(truth, truth) == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_labels_score_zero(self) -> None:
        truth = ["A", "A", "A"]
        recon = ["X", "Y", "Z"]
        assert celltype_distribution_cosine(truth, recon) == pytest.approx(0.0, abs=1e-9)

    def test_proportional_distributions_score_one(self) -> None:
        # cosine is scale-invariant: doubling counts shouldn't move the score.
        truth = ["A", "B", "B", "C", "C", "C"]
        recon = truth + truth  # 2x counts of each label
        assert celltype_distribution_cosine(truth, recon) == pytest.approx(1.0, abs=1e-9)

    def test_empty_input_returns_nan(self) -> None:
        assert math.isnan(celltype_distribution_cosine([], []))


class TestVirtualSliceAtDepth:
    @staticmethod
    def _make_volume(z_values: list[float]) -> ad.AnnData:
        rng = np.random.default_rng(0)
        n = len(z_values)
        adata = ad.AnnData(X=rng.normal(size=(n, 5)).astype(np.float32))
        adata.obs["physical_z_um"] = np.asarray(z_values, dtype=np.float64)
        adata.obsm["spatial"] = rng.normal(size=(n, 2)).astype(np.float32)
        return adata

    def test_slice_returns_cells_within_eps(self) -> None:
        volume = self._make_volume([0.0, 5.0, 10.0, 15.0, 20.0])
        sliced = virtual_slice_at_depth(volume, z_target=10.0, eps=2.5)
        assert sliced.n_obs == 1
        assert sliced.uns["virtual_slice"]["z_target"] == pytest.approx(10.0)
        assert sliced.uns["virtual_slice"]["eps"] == pytest.approx(2.5)
        assert sliced.uns["virtual_slice"]["n_cells_in_slab"] == 1

    def test_wider_slab_captures_more_cells(self) -> None:
        volume = self._make_volume([0.0, 5.0, 10.0, 15.0, 20.0])
        sliced = virtual_slice_at_depth(volume, z_target=10.0, eps=5.5)
        # |z - 10| <= 5.5 catches 5, 10, 15 → 3 cells
        assert sliced.n_obs == 3
        assert sliced.uns["virtual_slice"]["n_cells_in_slab"] == 3

    def test_missing_obs_column_raises(self) -> None:
        rng = np.random.default_rng(0)
        adata = ad.AnnData(X=rng.normal(size=(3, 2)).astype(np.float32))
        with pytest.raises(KeyError):
            virtual_slice_at_depth(adata, z_target=0.0, eps=1.0)

    def test_zero_or_negative_eps_raises(self) -> None:
        volume = self._make_volume([0.0, 1.0])
        with pytest.raises(ValueError):
            virtual_slice_at_depth(volume, z_target=0.0, eps=0.0)
        with pytest.raises(ValueError):
            virtual_slice_at_depth(volume, z_target=0.0, eps=-1.0)

    def test_custom_physical_z_key(self) -> None:
        rng = np.random.default_rng(0)
        adata = ad.AnnData(X=rng.normal(size=(3, 2)).astype(np.float32))
        adata.obs["depth"] = np.asarray([0.0, 5.0, 10.0])
        sliced = virtual_slice_at_depth(adata, z_target=5.0, eps=1.0,
                                        physical_z_key="depth")
        assert sliced.n_obs == 1
        assert sliced.uns["virtual_slice"]["physical_z_key"] == "depth"
