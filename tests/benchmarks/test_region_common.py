"""Tests for aether_3d.benchmarks.region_common (Round 13 W004)."""

from __future__ import annotations

import numpy as np
import pytest

from aether_3d.benchmarks.region_common import maximal_common_region


class TestMaximalCommonRegion:
    def test_two_overlapping_sections_intersect(self) -> None:
        s1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=float)
        s2 = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [3, 3]], dtype=float)
        out = maximal_common_region([s1, s2])
        xmin, xmax, ymin, ymax = out["bbox"]
        # s1 spans [0,2]x[0,2]; s2 spans [0.5,3]x[0.5,3]
        # intersection is [0.5, 2] x [0.5, 2].
        assert xmin == pytest.approx(0.5)
        assert xmax == pytest.approx(2.0)
        assert ymin == pytest.approx(0.5)
        assert ymax == pytest.approx(2.0)
        # s1's [2,2] point sits on the boundary → kept; [0,0] → dropped.
        assert out["masks"][0][4]
        assert not out["masks"][0][0]

    def test_shrink_inset(self) -> None:
        s1 = np.array([[0, 0], [10, 10]], dtype=float)
        s2 = np.array([[0, 0], [10, 10]], dtype=float)
        out = maximal_common_region([s1, s2], shrink=1.0)
        xmin, xmax, ymin, ymax = out["bbox"]
        assert xmin == pytest.approx(1.0)
        assert xmax == pytest.approx(9.0)
        assert ymin == pytest.approx(1.0)
        assert ymax == pytest.approx(9.0)

    def test_too_large_shrink_raises(self) -> None:
        s1 = np.array([[0, 0], [1, 1]], dtype=float)
        with pytest.raises(ValueError):
            maximal_common_region([s1, s1], shrink=10.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            maximal_common_region([])

    def test_non_2d_coords_raises(self) -> None:
        s = np.zeros((3, 3))
        with pytest.raises(ValueError):
            maximal_common_region([s])

    def test_empty_section_raises(self) -> None:
        s = np.zeros((0, 2))
        s2 = np.array([[0, 0], [1, 1]], dtype=float)
        with pytest.raises(ValueError):
            maximal_common_region([s, s2])

    def test_n_kept_per_section_matches_masks(self) -> None:
        s1 = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
        s2 = np.array([[3, 3], [4, 4], [11, 11]], dtype=float)
        out = maximal_common_region([s1, s2])
        for i, m in enumerate(out["masks"]):
            assert int(m.sum()) == int(out["n_kept_per_section"][i])
