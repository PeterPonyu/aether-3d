"""Smoke tests for the 2.5D-vs-continuous contrast comparison figure.

Guards the renderer + metrics parser in
``scripts/visualize/fig_contrast_comparison.py``: a small synthetic contrast
bundle must parse into the expected per-method table and render a valid,
non-empty PNG with the expected subplot structure. The module is imported
defensively so the suite still passes on a checkout where the script does not
yet exist (fail-to-exist on origin/main → pass).
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

fig_mod = pytest.importorskip("scripts.visualize.fig_contrast_comparison")


# PNG signature + IHDR (width, height) per the PNG spec.
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == _PNG_SIG, "output is not a valid PNG"
    # IHDR chunk: 8-byte sig, 4-byte length, 4-byte 'IHDR', then width/height.
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _synthetic_loo_bundle() -> dict[str, float]:
    """A leave-one-out-style flat metrics bundle (continuous beats baselines)."""
    bundle: dict[str, float] = {}
    # continuous is best: high Moran's I, low chamfer/rmse/wasserstein.
    spec = {
        "continuous": {
            "morans_i_agreement_top100": 0.91,
            "chamfer_distance": 0.12,
            "coord_rmse": 0.18,
            "sliced_wasserstein_2d": 0.07,
            "betti0_stability": 0.95,
        },
        "nearest_slice": {
            "morans_i_agreement_top100": 0.62,
            "chamfer_distance": 0.41,
            "coord_rmse": 0.55,
            "sliced_wasserstein_2d": 0.33,
            "betti0_stability": 0.70,
        },
        "linear_interp": {
            "morans_i_agreement_top100": 0.71,
            "chamfer_distance": 0.30,
            "coord_rmse": 0.40,
            "sliced_wasserstein_2d": 0.22,
            "betti0_stability": 0.78,
        },
        "stacking_2.5d": {
            "morans_i_agreement_top100": 0.66,
            "chamfer_distance": 0.36,
            "coord_rmse": 0.47,
            "sliced_wasserstein_2d": 0.28,
            "betti0_stability": 0.74,
        },
    }
    for method, metrics in spec.items():
        for metric, val in metrics.items():
            bundle[f"loo_contrast_{method}_{metric}_mean"] = val
    return bundle


def test_parser_extracts_loo_contrast() -> None:
    table = fig_mod.load_contrast_from_metrics(_synthetic_loo_bundle())
    assert set(table) == {"continuous", "nearest_slice", "linear_interp", "stacking_2.5d"}
    assert table["continuous"]["morans_i_agreement_top100"] == pytest.approx(0.91)
    assert table["stacking_2.5d"]["chamfer_distance"] == pytest.approx(0.36)
    # Every present method must carry all five contrast metrics.
    for method_vals in table.values():
        assert len(method_vals) == 5


def test_parser_falls_back_to_per_slice() -> None:
    """With no LOO keys, the parser averages per-slice contrast keys."""
    bundle = {
        "contrast_slice0_continuous_morans_i_agreement_top100": 0.80,
        "contrast_slice1_continuous_morans_i_agreement_top100": 0.90,
        "contrast_slice0_nearest_slice_coord_rmse": 0.50,
        "contrast_slice1_nearest_slice_coord_rmse": 0.60,
        # Non-contrast keys must be ignored.
        "slice0_gene_profile_pearson": 0.99,
        "loo_gene_pearson_mean": 0.88,
    }
    table = fig_mod.load_contrast_from_metrics(bundle)
    assert table["continuous"]["morans_i_agreement_top100"] == pytest.approx(0.85)
    assert table["nearest_slice"]["coord_rmse"] == pytest.approx(0.55)


def test_render_produces_valid_png(tmp_path: Path) -> None:
    table = fig_mod.load_contrast_from_metrics(_synthetic_loo_bundle())
    out = tmp_path / "contrast.png"
    returned = fig_mod.render_contrast_comparison(table, out)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 2000, "PNG suspiciously small / empty"
    width, height = _png_dimensions(out)
    assert width > 200 and height > 200, "PNG has degenerate dimensions"


def test_render_empty_contrast_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fig_mod.render_contrast_comparison({}, tmp_path / "x.png")


def test_main_cli_end_to_end(tmp_path: Path) -> None:
    import json

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_synthetic_loo_bundle()))
    out = tmp_path / "fig.png"
    rc = fig_mod.main(["--metrics", str(metrics_path), "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 2000
