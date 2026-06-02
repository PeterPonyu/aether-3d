#!/usr/bin/env python3
"""2.5D-baselines-vs-continuous contrast comparison panel (DeepSpatial Fig.2 style).

This renders the head-to-head fidelity contrast that ``evaluate_25d_contrast``
(``scripts/e2e/validate_holdout_slice.py``) emits into the holdout metrics
bundle: the continuous flow reconstruction scored against the naive 2.5D
baselines (nearest-slice copy, linear interpolation, and the clean-room
``stacking-2.5d`` reimplementation) on the *same* real holdout slice.

The figure is a grid of grouped bar charts — one subplot per available contrast
metric, bars per method — with the continuous reconstruction highlighted and an
explicit ``↑``/``↓`` arrow stating the optimisation direction of each metric.
It is a pure renderer: it consumes the existing metrics JSON and does **not**
recompute anything.

CLI::

    python scripts/visualize/fig_contrast_comparison.py \\
        --metrics results/holdout_validation_metrics.json \\
        --out results/figures/aether_contrast_comparison.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = PROJECT_ROOT / "results" / "holdout_validation_metrics.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "aether_contrast_comparison.png"

# Method key (as written into the metrics JSON, with adapter '-' → '_') → display label.
# 'continuous' is the Aether3D flow reconstruction; the rest are 2.5D baselines.
METHOD_LABELS: Tuple[Tuple[str, str], ...] = (
    ("continuous", "Continuous\n(Aether3D)"),
    ("nearest_slice", "NearestSlice"),
    ("linear_interp", "LinearInterp"),
    ("stacking_2.5d", "Stacking2.5D"),
)
CONTINUOUS_KEY = "continuous"

# Contrast metric key → (display label, higher_is_better). Mirrors CONTRAST_KEYS
# in scripts/e2e/validate_holdout_slice.py.
METRIC_SPECS: Tuple[Tuple[str, str, bool], ...] = (
    ("morans_i_agreement_top100", "Moran's I agreement", True),
    ("chamfer_distance", "Chamfer distance", False),
    ("coord_rmse", "Coord RMSE", False),
    ("sliced_wasserstein_2d", "Sliced Wasserstein", False),
    ("betti0_stability", "Betti-0 stability", True),
)

# Publication palette: continuous reconstruction gets the accent colour, the
# 2.5D baselines a muted grey-blue ramp so the eye reads them as the floor.
CONTINUOUS_COLOR = "#C94C4C"
BASELINE_COLORS = ("#5B7FA6", "#7E9CC0", "#A7BBD4")
GRID_COLOR = "#D8DCE0"
INK = "#1F2328"


def load_contrast_from_metrics(
    metrics: Mapping[str, object],
) -> Dict[str, Dict[str, float]]:
    """Extract the per-method contrast table from a flat holdout metrics dict.

    Prefers the leave-one-out aggregate keys
    (``loo_contrast_{method}_{metric}_mean``); if none are present (single
    holdout run), falls back to averaging the per-slice keys
    (``contrast_slice{h}_{method}_{metric}``). Returns
    ``{method_key: {metric_key: value}}`` containing only finite values.
    """
    method_keys = [mk for mk, _ in METHOD_LABELS]
    metric_keys = [mk for mk, _, _ in METRIC_SPECS]
    table: Dict[str, Dict[str, float]] = {mk: {} for mk in method_keys}

    def _finite(value: object) -> Optional[float]:
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
        return None

    # 1. Preferred: leave-one-out aggregate.
    found_loo = False
    for method in method_keys:
        for metric in metric_keys:
            val = _finite(metrics.get(f"loo_contrast_{method}_{metric}_mean"))
            if val is not None:
                table[method][metric] = val
                found_loo = True
    if found_loo:
        return {m: v for m, v in table.items() if v}

    # 2. Fallback: average the per-slice contrast keys. Both method and metric
    # tokens may themselves contain underscores (e.g. ``nearest_slice``,
    # ``coord_rmse``), so a generic split is ambiguous — instead match each
    # known (method, metric) pair against an exact ``contrast_slice{N}_...`` key.
    for method in method_keys:
        for metric in metric_keys:
            pattern = re.compile(rf"^contrast_slice\d+_{re.escape(method)}_{re.escape(metric)}$")
            vals = [
                v for key, raw in metrics.items()
                if pattern.match(key) and (v := _finite(raw)) is not None
            ]
            if vals:
                table[method][metric] = float(np.mean(vals))

    return {m: v for m, v in table.items() if v}


def render_contrast_comparison(
    contrast: Mapping[str, Mapping[str, float]],
    out_path: Path,
    title: str = "2.5D baselines vs continuous reconstruction",
    subtitle: Optional[str] = None,
) -> Path:
    """Render the grouped-bar contrast panel and save it to ``out_path``.

    ``contrast`` maps method key → {metric key → value}, as produced by
    :func:`load_contrast_from_metrics`. One subplot is drawn per contrast
    metric that has at least one method value; methods missing a metric are
    simply omitted from that subplot.
    """
    methods = [(mk, lbl) for mk, lbl in METHOD_LABELS if contrast.get(mk)]
    if not methods:
        raise ValueError("No contrast methods with data to plot.")

    # Keep only metrics that at least one present method reports.
    metric_specs = [
        (key, lbl, hib)
        for key, lbl, hib in METRIC_SPECS
        if any(key in contrast.get(mk, {}) for mk, _ in methods)
    ]
    if not metric_specs:
        raise ValueError("No contrast metrics with data to plot.")

    color_of: Dict[str, str] = {}
    baseline_i = 0
    for mk, _ in methods:
        if mk == CONTINUOUS_KEY:
            color_of[mk] = CONTINUOUS_COLOR
        else:
            color_of[mk] = BASELINE_COLORS[baseline_i % len(BASELINE_COLORS)]
            baseline_i += 1

    n = len(metric_specs)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 3.4 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()

    x_positions: npt.NDArray[np.float64] = np.arange(len(methods), dtype=np.float64)
    labels = [lbl for _, lbl in methods]

    for ax_idx, (metric_key, metric_label, higher_is_better) in enumerate(metric_specs):
        ax = flat_axes[ax_idx]
        heights: npt.NDArray[np.float64] = np.array(
            [float(contrast.get(mk, {}).get(metric_key, np.nan)) for mk, _ in methods],
            dtype=np.float64,
        )
        colors = [color_of[mk] for mk, _ in methods]
        bars = ax.bar(
            x_positions, np.nan_to_num(heights, nan=0.0), color=colors,
            edgecolor=INK, linewidth=0.6, width=0.7, zorder=3,
        )

        # Mark the best performer with a star.
        finite_mask = np.isfinite(heights)
        if finite_mask.any():
            finite_vals = np.where(finite_mask, heights, np.nan)
            best_idx = int(
                np.nanargmax(finite_vals) if higher_is_better
                else np.nanargmin(finite_vals)
            )
        else:
            best_idx = -1

        for bar_idx, (bar, val) in enumerate(zip(bars, heights)):
            if not np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 0.0, "n/a",
                    ha="center", va="bottom", fontsize=7, color="#888",
                )
                continue
            star = " ★" if bar_idx == best_idx else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3g}{star}", ha="center", va="bottom", fontsize=7.5,
                color=INK,
            )

        arrow = "↑" if higher_is_better else "↓"
        better = "higher better" if higher_is_better else "lower better"
        ax.set_title(f"{metric_label}  ({arrow} {better})", fontsize=9.5, color=INK)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Headroom for the value labels.
        finite_heights = heights[finite_mask]
        if finite_heights.size:
            top = float(np.max(finite_heights))
            bottom = min(0.0, float(np.min(finite_heights)))
            pad = 0.18 * (top - bottom if top > bottom else abs(top) or 1.0)
            ax.set_ylim(bottom, top + pad)

    # Hide any unused axes in the grid.
    for spare in flat_axes[n:]:
        spare.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK)
    footnote = (
        subtitle
        or "Continuous = Aether3D flow reconstruction; 2.5D baselines are naive "
        "lower bounds (nearest-slice / linear-interp / stacking) scored on the "
        "same real holdout. ★ marks the best method per metric."
    )
    fig.text(0.5, 0.005, footnote, ha="center", va="bottom", fontsize=7.5,
             color="#65717E", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=DEFAULT_METRICS,
        help="Path to the holdout metrics JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output PNG path (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.metrics.exists():
        parser.error(f"metrics JSON not found: {args.metrics}")
    payload = json.loads(args.metrics.read_text())
    if not isinstance(payload, dict):
        parser.error("metrics JSON must be a flat object of metric → value")

    contrast = load_contrast_from_metrics(payload)
    if not contrast:
        parser.error(
            "no contrast_* keys found in metrics JSON — run "
            "scripts/e2e/validate_holdout_slice.py to emit the 2.5D contrast first"
        )
    out = render_contrast_comparison(contrast, args.out)
    print(f"Wrote contrast comparison panel → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
