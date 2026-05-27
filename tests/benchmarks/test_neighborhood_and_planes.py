"""Tests for Round 12 W004 + W005 additions in aether-3d."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from aether_3d.benchmarks.metrics import virtual_plane
from aether_3d.benchmarks.neighborhood import (
    per_region_proportion_spearman,
    radius_neighborhood_enrichment,
    z_axis_density_around,
)


class TestRadiusNeighborhoodEnrichment:
    def test_clustered_target_enriches_for_itself(self) -> None:
        # 30 "T" cells tightly clustered, plus 30 "B" cells far away.
        rng = np.random.default_rng(0)
        t_cells = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
        b_cells = rng.normal(loc=[10, 10], scale=0.1, size=(30, 2))
        coords = np.vstack([t_cells, b_cells])
        labels = ["T"] * 30 + ["B"] * 30
        out = radius_neighborhood_enrichment(coords, labels, "T", radius=0.5)
        assert out["n_targets"] == 30
        # T's near a T are over-represented; B's are absent within 0.5 radius.
        assert out["per_celltype_enrichment"]["T"] > 1.5
        assert out["per_celltype_obs_proportion"]["B"] == pytest.approx(0.0, abs=1e-9)

    def test_no_target_returns_empty(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = radius_neighborhood_enrichment(coords, ["A", "B"], "Z", radius=1.0)
        assert out["n_targets"] == 0
        assert out["per_celltype_enrichment"] == {}

    def test_zero_radius_raises(self) -> None:
        coords = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            radius_neighborhood_enrichment(coords, ["A"], "A", radius=0.0)


class TestZAxisDensityAround:
    def test_density_peak_at_query_depth(self) -> None:
        # T-cells all at z=10; B-cells spread across z in [0, 20].
        rng = np.random.default_rng(0)
        t = np.column_stack([rng.uniform(0, 1, size=20),
                             rng.uniform(0, 1, size=20),
                             np.full(20, 10.0)])
        b = np.column_stack([rng.uniform(0, 1, size=200),
                             rng.uniform(0, 1, size=200),
                             rng.uniform(0, 20, size=200)])
        coords = np.vstack([t, b])
        labels = ["T"] * 20 + ["B"] * 200
        out = z_axis_density_around(coords, labels, query_label="T",
                                    z_bin_width=2.0, xy_radius=2.0)
        assert "B" in out["per_celltype_density"]
        # The xy-radius is wide enough that B-cells get counted; with z bins
        # of 2 µm and bin centers spanning [0,20], B density should be > 0 in
        # several bins.
        b_density = out["per_celltype_density"]["B"]
        assert (b_density > 0).sum() >= 2

    def test_bad_input_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            z_axis_density_around(np.zeros((3, 2)), ["A"] * 3, "A")


class TestPerRegionProportionSpearman:
    def test_perfect_match_per_region(self) -> None:
        coords = np.zeros((10, 2))  # unused
        labels = ["A", "A", "B", "B", "C", "A", "A", "C", "C", "B"]
        regions = ["r1"] * 5 + ["r2"] * 5
        truth = {
            "r1": {"A": 0.4, "B": 0.4, "C": 0.2},
            "r2": {"A": 0.4, "B": 0.2, "C": 0.4},
        }
        out = per_region_proportion_spearman(coords, labels, regions, truth)
        assert out["r1"] == pytest.approx(1.0, abs=1e-9)
        # r2 matches in rank order too.
        assert out["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_none_truth_returns_nans(self) -> None:
        coords = np.zeros((4, 2))
        labels = ["A"] * 2 + ["B"] * 2
        regions = ["r1"] * 4
        out = per_region_proportion_spearman(coords, labels, regions, None)
        assert np.isnan(out["r1"])


class TestVirtualPlane:
    @staticmethod
    def _make_volume(n: int = 60) -> ad.AnnData:
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 10, size=(n, 2))
        z = rng.uniform(0, 20, size=n)
        adata = ad.AnnData(X=rng.normal(size=(n, 4)).astype(np.float32))
        adata.obsm["spatial"] = xy.astype(np.float32)
        adata.obs["physical_z_um"] = z
        return adata

    def test_axis_z_slabs_match(self) -> None:
        volume = self._make_volume()
        sliced = virtual_plane(volume, axis="z", target=10.0, eps=2.0)
        z = volume.obs["physical_z_um"].values
        assert sliced.n_obs == int((np.abs(z - 10.0) <= 2.0).sum())
        assert sliced.uns["virtual_plane"]["axis"] == "z"

    def test_axis_x_slabs_match(self) -> None:
        volume = self._make_volume()
        sliced = virtual_plane(volume, axis="x", target=5.0, eps=1.0)
        x = volume.obsm["spatial"][:, 0]
        assert sliced.n_obs == int((np.abs(x - 5.0) <= 1.0).sum())
        assert sliced.uns["virtual_plane"]["axis"] == "x"

    def test_axis_y_slabs_match(self) -> None:
        volume = self._make_volume()
        sliced = virtual_plane(volume, axis="y", target=5.0, eps=1.0)
        y = volume.obsm["spatial"][:, 1]
        assert sliced.n_obs == int((np.abs(y - 5.0) <= 1.0).sum())
        assert sliced.uns["virtual_plane"]["axis"] == "y"

    def test_bad_axis_raises(self) -> None:
        volume = self._make_volume()
        with pytest.raises(ValueError):
            virtual_plane(volume, axis="w", target=0.0, eps=1.0)

    def test_negative_eps_raises(self) -> None:
        volume = self._make_volume()
        with pytest.raises(ValueError):
            virtual_plane(volume, axis="z", target=0.0, eps=0.0)
