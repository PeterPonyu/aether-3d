"""Tests for aether_3d.benchmarks.graph_3d (Round 13 W002)."""

from __future__ import annotations

import numpy as np
import pytest

from aether_3d.benchmarks.graph_3d import (
    branch_point_count,
    degree_histogram,
    edge_length_distribution,
    hub_centrality,
)


class TestDegreeHistogram:
    def test_isolated_nodes_count(self) -> None:
        # 5 nodes, 1 edge (0-1) → degrees [1, 1, 0, 0, 0].
        out = degree_histogram(n_nodes=5, edges=[(0, 1)])
        assert out["n_isolated"] == 3
        assert out["mean_degree"] == pytest.approx(2 / 5)

    def test_self_loops_dropped(self) -> None:
        out = degree_histogram(n_nodes=3, edges=[(0, 0), (0, 1)])
        # The (0,0) self-loop is dropped, so 0 and 1 each have degree 1.
        assert out["n_isolated"] == 1  # node 2
        assert out["mean_degree"] == pytest.approx(2 / 3)

    def test_duplicate_edges_dedupe(self) -> None:
        out = degree_histogram(n_nodes=3, edges=[(0, 1), (1, 0), (0, 1)])
        # All three list entries collapse to one undirected (0,1) edge.
        assert out["counts"].sum() == 3
        # Two nodes at degree 1, one isolated.
        assert out["n_isolated"] == 1

    def test_invalid_node_raises(self) -> None:
        with pytest.raises(ValueError):
            degree_histogram(n_nodes=2, edges=[(0, 5)])


class TestEdgeLengthDistribution:
    def test_unit_grid_lengths(self) -> None:
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        edges = [(0, 1), (0, 2), (0, 3)]
        out = edge_length_distribution(coords, edges, n_bins=5)
        np.testing.assert_allclose(out["lengths"], np.ones(3))
        assert out["mean"] == pytest.approx(1.0)

    def test_empty_edges_returns_nan_stats(self) -> None:
        coords = np.zeros((4, 3))
        out = edge_length_distribution(coords, [], n_bins=5)
        assert out["lengths"].size == 0
        assert np.isnan(out["mean"])

    def test_bad_coords_raises(self) -> None:
        with pytest.raises(ValueError):
            edge_length_distribution(np.zeros((3, 2)), [(0, 1)])


class TestBranchPointCount:
    def test_star_graph_one_branch_point(self) -> None:
        # Star: center=0 connected to 1, 2, 3 → branch at 0, endpoints 1,2,3.
        out = branch_point_count(n_nodes=4, edges=[(0, 1), (0, 2), (0, 3)])
        assert out["n_branch_points"] == 1
        assert out["n_endpoints"] == 3

    def test_path_graph_no_branch_points(self) -> None:
        # Path: 0-1-2-3 → no branches, two endpoints (0 and 3).
        out = branch_point_count(n_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
        assert out["n_branch_points"] == 0
        assert out["n_endpoints"] == 2

    def test_branch_degree_validation(self) -> None:
        with pytest.raises(ValueError):
            branch_point_count(n_nodes=3, edges=[(0, 1)], branch_degree=1)


class TestHubCentrality:
    def test_star_center_is_top_hub(self) -> None:
        out = hub_centrality(n_nodes=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)], top_k=1)
        assert out["top_hub_ids"][0] == 0
        # Center has degree 4 of (5-1)=4 → centrality 1.0.
        assert out["degree_centrality"][0] == pytest.approx(1.0)
        # All others are degree-1 → 1/4.
        for i in range(1, 5):
            assert out["degree_centrality"][i] == pytest.approx(0.25)

    def test_top_k_capped_at_n_nodes(self) -> None:
        out = hub_centrality(n_nodes=2, edges=[(0, 1)], top_k=5)
        assert out["top_hub_ids"].shape[0] == 2

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError):
            hub_centrality(n_nodes=3, edges=[(0, 1)], top_k=0)
