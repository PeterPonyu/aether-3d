"""Regression tests for the Round 13 neighborhood-heatmap composer."""

from __future__ import annotations

import importlib.util
from types import ModuleType
from pathlib import Path

import numpy as np
import pytest


def _load_compose_neighborhood_heatmap() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "manuscript"
        / "compose_neighborhood_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("compose_neighborhood_heatmap", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_neighborhood_heatmap_writes_nonempty_png(tmp_path: Path) -> None:
    module = _load_compose_neighborhood_heatmap()
    matrix = np.array([[1.0, 1.4, 0.7], [0.8, 1.0, 1.9]], dtype=np.float64)

    out = module.render_neighborhood_heatmap(
        matrix,
        ["T", "B"],
        ["T", "B", "M"],
        tmp_path / "heatmap.png",
        radius_um=20.0,
    )

    assert out == tmp_path / "heatmap.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_neighborhood_heatmap_rejects_shape_mismatch(tmp_path: Path) -> None:
    module = _load_compose_neighborhood_heatmap()

    with pytest.raises(ValueError, match="shape"):
        module.render_neighborhood_heatmap(
            np.ones((2, 2)),
            ["T"],
            ["T", "B"],
            tmp_path / "bad.png",
        )


def test_render_neighborhood_heatmap_handles_all_neutral_matrix(tmp_path: Path) -> None:
    module = _load_compose_neighborhood_heatmap()

    out = module.render_neighborhood_heatmap(
        np.ones((2, 3), dtype=np.float64),
        ["T", "B"],
        ["T", "B", "M"],
        tmp_path / "neutral.png",
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_render_neighborhood_heatmap_handles_nan_only_matrix(tmp_path: Path) -> None:
    module = _load_compose_neighborhood_heatmap()

    out = module.render_neighborhood_heatmap(
        np.full((1, 1), np.nan, dtype=np.float64),
        ["T"],
        ["T"],
        tmp_path / "nan.png",
    )

    assert out.exists()
    assert out.stat().st_size > 0
