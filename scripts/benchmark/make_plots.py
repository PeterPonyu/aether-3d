#!/usr/bin/env python3
"""
Render Aether3D benchmark figures + Markdown report from a sweep JSON.

Reads results/benchmark/aether_sweep_latest.json and writes:

  docs/benchmark/figures/metric_*.png
  docs/benchmark/figures/loss_curves.png
  docs/benchmark/figures/runtime.png
  docs/benchmark/figures/peak_gpu_mem.png
  docs/benchmark/figures/volume_xy_<config>.png    (top-down view per config)
  docs/benchmark/figures/volume_xz_<config>.png    (side view per config)
  docs/benchmark/summary.csv
  docs/benchmark/BENCHMARK_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "results" / "benchmark" / "aether_sweep_latest.json"
DEFAULT_DOCS = PROJECT_ROOT / "docs" / "benchmark"


METRIC_ORDER = [
    ("gene_profile_pearson", "Gene-profile Pearson (↑)"),
    ("cell_level_mean_pearson", "Cell-level mean Pearson (↑)"),
    ("gene_profile_mse", "Gene-profile MSE (↓)"),
    ("cell_level_mean_mse", "Cell-level mean MSE (↓)"),
]


def _bar(ax, names: List[str], values: List[float], ylabel: str, title: str) -> None:
    bars = ax.bar(names, values, color="#4C72B0", edgecolor="#1f1f1f")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for b, v in zip(bars, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)


def render_metric_bars(records: List[Dict[str, Any]], fig_dir: Path) -> List[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [r["config"]["name"] for r in records]
    paths: List[Path] = []
    for key, label in METRIC_ORDER:
        vals = [r["metrics"].get(key, float("nan")) for r in records]
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        _bar(ax, names, [float(v) if v is not None else float("nan") for v in vals], label, label)
        fig.tight_layout()
        p = fig_dir / f"metric_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def render_runtime_and_mem(records: List[Dict[str, Any]], fig_dir: Path) -> Dict[str, Path]:
    names = [r["config"]["name"] for r in records]
    totals = [r["wall_seconds"]["total"] for r in records]
    train = [r["wall_seconds"]["flow_train"] for r in records]
    recon = [r["wall_seconds"]["reconstruct"] for r in records]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(names))
    ax.bar(x, train, label="Flow train", color="#4C72B0")
    ax.bar(x, recon, bottom=train, label="Reconstruct", color="#C44E52")
    ax.set_xticks(x, names)
    ax.set_ylabel("Wall seconds")
    ax.set_title("Wall-clock breakdown per config")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, t in enumerate(totals):
        ax.annotate(f"{t:.1f}s", (i, t), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    runtime_path = fig_dir / "runtime.png"
    fig.savefig(runtime_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    peak = [r["peak_gpu_mem_mb"] if r["peak_gpu_mem_mb"] is not None else 0.0 for r in records]
    _bar(ax, names, peak, "Peak GPU MB", "Peak GPU memory per config")
    fig.tight_layout()
    mem_path = fig_dir / "peak_gpu_mem.png"
    fig.savefig(mem_path, dpi=150)
    plt.close(fig)

    return {"runtime": runtime_path, "mem": mem_path}


def render_loss_curves(records: List[Dict[str, Any]], fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for r in records:
        curve_path = PROJECT_ROOT / r["loss_curve_path"]
        if not curve_path.exists():
            continue
        curve = json.loads(curve_path.read_text())
        ax.plot(curve.get("flow", []), label=f"{r['config']['name']} (flow)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow loss")
    ax.set_title("Flow training loss per config")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "loss_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def render_volume_views(records: List[Dict[str, Any]], fig_dir: Path) -> Dict[str, List[Path]]:
    import scanpy as sc

    xy_paths: List[Path] = []
    xz_paths: List[Path] = []
    for r in records:
        vol_path = PROJECT_ROOT / r["volume_path"]
        if not vol_path.exists():
            continue
        v = sc.read_h5ad(vol_path)
        xy = v.obsm.get("spatial_3d") if "spatial_3d" in v.obsm else None
        if xy is None:
            continue
        z = np.asarray(v.obs["z_3d"]) if "z_3d" in v.obs else np.zeros(v.n_obs)
        depth = np.asarray(v.obs["virtual_depth"]) if "virtual_depth" in v.obs else np.zeros(v.n_obs)
        coords = np.asarray(xy)

        # Top-down (XY) coloured by depth
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=depth, s=2, cmap="viridis")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(f"{r['config']['name']} — XY (color = virtual depth)")
        fig.colorbar(sc_plot, ax=ax, label="virtual_depth")
        fig.tight_layout()
        p = fig_dir / f"volume_xy_{r['config']['name']}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        xy_paths.append(p)

        # Side view (X vs Z) coloured by Y
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        sc_plot = ax.scatter(coords[:, 0], z, c=coords[:, 1], s=2, cmap="plasma")
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_title(f"{r['config']['name']} — XZ (color = Y)")
        fig.colorbar(sc_plot, ax=ax, label="Y")
        fig.tight_layout()
        p = fig_dir / f"volume_xz_{r['config']['name']}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        xz_paths.append(p)
    return {"xy": xy_paths, "xz": xz_paths}


def write_summary_csv(records: List[Dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "name", "hidden_size", "depth", "num_heads", "max_epochs",
        "n_params", "wall_total_s", "wall_train_s", "wall_recon_s",
        "peak_gpu_mb",
        "gene_pearson", "cell_pearson", "gene_mse", "cell_mse",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in records:
            c = r["config"]; m = r["metrics"]; t = r["wall_seconds"]
            w.writerow([
                c["name"], c["hidden_size"], c["depth"], c["num_heads"], c["max_epochs"],
                r["n_params"],
                f"{t['total']:.3f}", f"{t['flow_train']:.3f}", f"{t['reconstruct']:.3f}",
                r["peak_gpu_mem_mb"],
                m.get("gene_profile_pearson"), m.get("cell_level_mean_pearson"),
                m.get("gene_profile_mse"), m.get("cell_level_mean_mse"),
            ])


def _md_metrics_table(records: List[Dict[str, Any]]) -> str:
    lines = [
        "| Config | hidden | depth | heads | epochs | Gene Pearson | Cell Pearson | Gene MSE | Cell MSE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["config"]; m = r["metrics"]
        def fmt(v): return "—" if v is None else f"{float(v):.4f}"
        lines.append(
            f"| `{c['name']}` | {c['hidden_size']} | {c['depth']} | {c['num_heads']} | {c['max_epochs']} | "
            f"{fmt(m.get('gene_profile_pearson'))} | {fmt(m.get('cell_level_mean_pearson'))} | "
            f"{fmt(m.get('gene_profile_mse'))} | {fmt(m.get('cell_level_mean_mse'))} |"
        )
    return "\n".join(lines)


def _md_resource_table(records: List[Dict[str, Any]]) -> str:
    lines = [
        "| Config | Params | Wall total (s) | Flow train (s) | Reconstruct (s) | Peak GPU (MB) |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["config"]; t = r["wall_seconds"]
        peak = "—" if r["peak_gpu_mem_mb"] is None else f"{r['peak_gpu_mem_mb']:.1f}"
        lines.append(
            f"| `{c['name']}` | {r['n_params']:,} | {t['total']:.2f} | {t['flow_train']:.2f} | {t['reconstruct']:.2f} | {peak} |"
        )
    return "\n".join(lines)


def write_report(payload: Dict[str, Any], fig_paths: Dict[str, Any], docs_dir: Path) -> Path:
    records = payload["records"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = lambda p: f"./figures/{p.name}"  # noqa: E731

    md = []
    md.append("# Aether3D Synthetic Benchmark Report")
    md.append("")
    md.append(f"- Generated: `{payload['timestamp']}`")
    md.append(f"- Device: `{payload['device']}`")
    d = payload["data_settings"]
    md.append(f"- Synthetic data: 3 slices × {d['cells_per_slice']} cells × {d['n_genes']} genes, "
              f"{d['n_classes']} cell classes, slice spacing {d['slice_spacing']}, seed {d['seed']}")
    md.append("")
    md.append("Slice 0 and Slice 2 are used for training; Slice 1 (Z=10) is held out and reconstructed "
              "via virtual-depth interpolation. Metrics compare reconstructed vs held-out.")
    md.append("")

    md.append("## Quality metrics per refined version")
    md.append("")
    md.append(_md_metrics_table(records))
    md.append("")
    for p in fig_paths.get("metrics", []) or []:
        md.append(f"![{p.stem}]({fig_rel(p)})")
    md.append("")

    md.append("## Resource usage per refined version")
    md.append("")
    md.append(_md_resource_table(records))
    md.append("")
    if fig_paths.get("runtime"):
        md.append(f"![runtime breakdown]({fig_rel(fig_paths['runtime'])})")
        md.append("")
    if fig_paths.get("mem"):
        md.append(f"![peak GPU memory]({fig_rel(fig_paths['mem'])})")
        md.append("")

    if fig_paths.get("loss_curves"):
        md.append("## Training loss curves")
        md.append("")
        md.append(f"![loss curves]({fig_rel(fig_paths['loss_curves'])})")
        md.append("")

    xy = fig_paths.get("volume_xy") or []
    xz = fig_paths.get("volume_xz") or []
    if xy or xz:
        md.append("## Reconstructed volume views")
        md.append("")
        for p in xy:
            md.append(f"![{p.stem}]({fig_rel(p)})")
        for p in xz:
            md.append(f"![{p.stem}]({fig_rel(p)})")
        md.append("")

    md.append("---")
    md.append("")
    md.append("Re-run with:")
    md.append("")
    md.append("```bash")
    md.append("conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py")
    md.append("conda run --no-capture-output -n dl python scripts/benchmark/make_plots.py")
    md.append("```")
    md.append("")

    out = docs_dir / "BENCHMARK_REPORT.md"
    out.write_text("\n".join(md))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    args = parser.parse_args()

    payload = json.loads(args.sweep_json.read_text())
    records = payload["records"]

    fig_dir = args.docs_dir / "figures"
    metric_paths = render_metric_bars(records, fig_dir)
    rt_mem = render_runtime_and_mem(records, fig_dir)
    lc_path = render_loss_curves(records, fig_dir)
    vol_paths = render_volume_views(records, fig_dir)

    write_summary_csv(records, args.docs_dir / "summary.csv")

    out = write_report(
        payload,
        {
            "metrics": metric_paths,
            "runtime": rt_mem["runtime"],
            "mem": rt_mem["mem"],
            "loss_curves": lc_path,
            "volume_xy": vol_paths["xy"],
            "volume_xz": vol_paths["xz"],
        },
        args.docs_dir,
    )
    print(f"[plots] Wrote {out}")
    print(f"[plots] Summary CSV: {args.docs_dir / 'summary.csv'}")
    print(f"[plots] Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
