#!/usr/bin/env python3
"""Compose multi-panel figures for the Aether3D manuscript.

Reads the synthetic benchmark JSONs under `aether-3d/results/benchmark/`
and renders each manuscript figure as a single PDF in `figures/`.

Build chain: `make figures` in this directory.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BENCH_DIR = PROJECT_ROOT / "results" / "benchmark"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLACEHOLDER_BANNER = "synthetic placeholder — real data pending"


def _load_or_warn(path: Path) -> dict | None:
    if not path.exists():
        print(f"[compose] WARN: missing {path.relative_to(PROJECT_ROOT)}; skipping",
              file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _annotate_placeholder(ax) -> None:
    ax.text(
        0.99, 0.01, PLACEHOLDER_BANNER,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, style="italic", color="grey",
    )


def _bold_panel_label(ax, letter: str) -> None:
    """Bold lowercase panel label at top-left.

    Mirrors `docs/REFERENCE_FIGURE_STYLE.md`.
    """
    ax.text(
        -0.10, 1.05, letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold",
    )


def _stamp_provenance(fig, source_path: Path, data_card_id: str | None) -> None:
    """Foot-of-figure provenance stamp."""
    try:
        rel = source_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = source_path
    card = f"  ·  data_card_id={data_card_id}" if data_card_id else ""
    fig.text(
        0.99, 0.005,
        f"source: {rel}{card}",
        ha="right", va="bottom",
        fontsize=5.5, style="italic", color="dimgrey",
        family="monospace",
    )


# -- Figure 1: virtual-slice holdout 7-metric quartet --------------------


def compose_holdout(holdout: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    holdouts = holdout["holdouts"]
    key = next(iter(holdouts))
    methods = holdouts[key]

    metric_keys = [
        "mean_chamfer", "mean_coord_rmse", "mean_sliced_wasserstein_2d",
        "mean_morans_i_agreement", "mean_domain_ari", "mean_domain_nmi",
        "mean_betti0_stability",
    ]
    metric_labels = [
        "Chamfer ↓", "Coord RMSE ↓", "Sliced 2D Wasserstein ↓",
        "Moran's I agreement ↑", "Domain ARI ↑", "Domain NMI ↑",
        "Betti-0 stability ↑",
    ]
    method_names = sorted(methods.keys())

    n_rows = 2
    n_cols = math.ceil(len(metric_keys) / n_rows)
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.45)

    for mi, (mk, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = fig.add_subplot(gs[mi // n_cols, mi % n_cols])
        vals: list[tuple[str, float]] = []
        for m in method_names:
            v = methods[m].get("metrics", {}).get(mk, float("nan"))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            if not math.isnan(fv):
                vals.append((m, fv))
        if not vals:
            ax.text(0.5, 0.5, "no available results",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            xs = np.arange(len(vals))
            ax.bar(xs, [v for _, v in vals], color="teal", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels([m for m, _ in vals], rotation=30, ha="right",
                               fontsize=8)
            for x, (_, v) in zip(xs, vals):
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel(mlabel)
        _bold_panel_label(ax, chr(ord('a') + mi))
        ax.set_title(mlabel, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _annotate_placeholder(ax)

    fig.suptitle(
        "Virtual-slice holdout — 7-metric quartet (synthetic 5-slice stack, middle slice held out)",
        fontsize=11, y=0.99,
    )
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 2: UOT ablation heatmap -------------------------------------


def compose_uot_ablation(ablation: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    results = ablation["results"]
    alphas = sorted({r["point"]["alpha_spatial"] for r in results})
    lambdas = sorted({r["point"]["lambda_class"] for r in results})

    top1 = np.full((len(alphas), len(lambdas)), np.nan, dtype=np.float64)
    mass = np.full((len(alphas), len(lambdas)), np.nan, dtype=np.float64)
    for r in results:
        ai = alphas.index(r["point"]["alpha_spatial"])
        li = lambdas.index(r["point"]["lambda_class"])
        top1[ai, li] = r["top1_accuracy"]
        mass[ai, li] = r["mean_true_pair_mass"]

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    for pi, (mat, label) in enumerate([
        (top1, "top-1 coupling accuracy ↑"),
        (mass, "mean true-pair mass ↑"),
    ]):
        ax = fig.add_subplot(gs[0, pi])
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=1)
        _bold_panel_label(ax, chr(ord('a') + pi))
        ax.set_xticks(np.arange(len(lambdas)))
        ax.set_xticklabels([f"{l:g}" for l in lambdas])
        ax.set_yticks(np.arange(len(alphas)))
        ax.set_yticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel(r"$\lambda_{\mathrm{class}}$")
        ax.set_ylabel(r"$\alpha_{\mathrm{spatial}}$")
        ax.set_title(label)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _annotate_placeholder(ax)

    fig.suptitle(r"UOT-cost component ablation: $\alpha_{\mathrm{spatial}} \times \lambda_{\mathrm{class}}$",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 3: scaling curve --------------------------------------------


def compose_scaling(scaling: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    results = scaling["results"]

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    # (a) runtime vs cells per slice, grouped by adapter
    ax_a = fig.add_subplot(gs[0, 0])
    by_adapter: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        by_adapter.setdefault(r["adapter"], []).append(
            (r["point"]["n_cells_per_slice"], r["runtime_s"])
        )
    for ad, points in by_adapter.items():
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax_a.plot(xs, ys, marker="o", label=ad)
    ax_a.set_xlabel("cells per slice")
    ax_a.set_ylabel("runtime (s)")
    _bold_panel_label(ax_a, "a")
    ax_a.set_title("Runtime vs synthetic stack size", fontsize=10)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.legend(fontsize=8)
    ax_a.grid(linestyle=":", alpha=0.5)
    _annotate_placeholder(ax_a)

    # (b) per-measurement provenance text panel
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    if results:
        sample = results[0]
        prov_lines = [
            "Per-measurement provenance",
            "—" * 40,
            f"device:        {sample['device']}",
            f"torch:         {sample['torch_version']}",
            f"CUDA:          {sample['cuda_version']}",
            f"hostname:      {sample.get('hostname', '?')}",
            f"python:        {sample.get('python_version', '?')}",
            f"platform:      {sample.get('platform', '?')[:48]}",
            f"git SHA:       {(sample.get('git_sha') or '(unset)')[:12]}",
            "",
            f"n_results recorded: {scaling.get('n_results', len(results))}",
            "schema_version: " + str(scaling.get('schema_version', '1')),
        ]
        for i, line in enumerate(prov_lines):
            ax_b.text(0.0, 0.95 - i * 0.07, line,
                      transform=ax_b.transAxes, family="monospace",
                      fontsize=9, va="top")
        _bold_panel_label(ax_b, "b")
        ax_b.set_title("Reproducibility provenance", fontsize=10)
        _annotate_placeholder(ax_b)

    fig.suptitle("Hardware-honest scaling on synthetic stacks",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-json", type=Path,
        default=BENCH_DIR / "synthetic_holdout.json",
        help="Benchmark JSON for the virtual-slice holdout figure.",
    )
    parser.add_argument(
        "--ablation-json", type=Path,
        default=BENCH_DIR / "uot_ablation.json",
        help="Benchmark JSON for the UOT-cost ablation figure.",
    )
    parser.add_argument(
        "--scaling-json", type=Path,
        default=BENCH_DIR / "scaling_curve.json",
        help="Benchmark JSON for the scaling curve figure.",
    )
    parser.add_argument(
        "--data-card-id", type=str, default=None,
        help="Optional data card id stamped on every figure's provenance footer.",
    )
    args = parser.parse_args()

    print(f"[compose] benchmark dir: {BENCH_DIR}")
    print(f"[compose] figures out:   {FIG_DIR}")
    if args.data_card_id:
        print(f"[compose] data_card_id:  {args.data_card_id}")

    holdout = _load_or_warn(args.holdout_json)
    if holdout:
        compose_holdout(holdout, FIG_DIR / "fig_holdout_quartet.pdf", args.holdout_json, args.data_card_id)

    ablation = _load_or_warn(args.ablation_json)
    if ablation:
        compose_uot_ablation(ablation, FIG_DIR / "fig_uot_ablation.pdf", args.ablation_json, args.data_card_id)

    scaling = _load_or_warn(args.scaling_json)
    if scaling:
        compose_scaling(scaling, FIG_DIR / "fig_scaling_curve.pdf", args.scaling_json, args.data_card_id)

    print("[compose] all figures composed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
