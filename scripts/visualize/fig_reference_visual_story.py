#!/usr/bin/env python3
"""Clean-room Aether3D manuscript visualization storyboard.

This figure is informed by reading the baseline reference paper, but it does
not copy source artwork, captions, logos, or panel wording.  It translates the
observed visual logic into repository-specific Aether3D panels: serial-slice to
continuous-volume schematic, virtual slicing, z-depth composition, marker
consistency, neighborhood structure, and 3D point-cloud summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "aether_reference_visual_story.png"

COLORS = ["#356AA0", "#20A39E", "#E7903C", "#C94C4C", "#7D5FB2", "#5BA85B"]
GRAY = "#6E7781"


def _panel_label(ax, label: str) -> None:
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(
        -0.06,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def _load_holdout_summary() -> tuple[dict[str, float], str]:
    path = PROJECT_ROOT / "results" / "holdout_validation_metrics.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            vals = {}
            for k in (
                "coordinate_rmse",
                "chamfer",
                "marker_continuity",
                "proportion_corr",
                "domain_ari",
            ):
                if k in payload and isinstance(payload[k], (int, float)):
                    vals[k] = float(payload[k])
            if vals:
                return vals, f"local metrics: {path.relative_to(PROJECT_ROOT)}"
        except Exception:
            pass
    return {
        "coordinate_rmse": 0.18,
        "chamfer": 0.23,
        "marker_continuity": 0.82,
        "proportion_corr": 0.88,
        "domain_ari": 0.71,
    }, "planning/demo values — replace with benchmark JSON before paper claims"


def _make_volume(seed: int = 23) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-1.3, -0.5, -0.7], [0.0, 0.8, 0.0], [1.2, -0.25, 0.75], [-0.45, -1.1, 0.55]]
    )
    pts, cls = [], []
    for i, c in enumerate(centers):
        cov = np.diag([0.13 + 0.04 * i, 0.10 + 0.03 * i, 0.18])
        block = rng.multivariate_normal(c, cov, size=230)
        pts.append(block)
        cls.extend([i] * len(block))
    pts = np.vstack(pts)
    cls = np.asarray(cls)
    marker = np.exp(
        -((pts[:, 0] - 1.0) ** 2 + (pts[:, 1] + 0.2) ** 2 + (pts[:, 2] - 0.7) ** 2)
        / 0.55
    )
    return pts, cls, marker


def _draw_workflow(ax) -> None:
    ax.set_axis_off()
    _panel_label(ax, "A")
    ax.set_title("Aether3D reconstruction workflow", loc="left", fontsize=11)
    boxes = [
        (0.03, 0.60, 0.20, 0.24, "serial 2D slices\nXY + genes + labels", COLORS[0]),
        (0.31, 0.60, 0.20, 0.24, "UOT / flow model\nlearn z-continuity", COLORS[1]),
        (0.59, 0.60, 0.20, 0.24, "continuous volume\nvirtual cells", COLORS[2]),
        (0.59, 0.22, 0.20, 0.22, "virtual slicing\ncoronal/sagittal", COLORS[4]),
        (0.82, 0.22, 0.15, 0.22, "export +\nclaim gates", COLORS[3]),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                edgecolor=color,
                facecolor=color + "22",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    for a, b in [
        ((0.23, 0.72), (0.31, 0.72)),
        ((0.51, 0.72), (0.59, 0.72)),
        ((0.69, 0.60), (0.69, 0.44)),
        ((0.79, 0.33), (0.82, 0.33)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=12, color=GRAY, linewidth=1.2
            )
        )
    # small slice stack icon
    for i, z in enumerate([0.08, 0.12, 0.16]):
        ax.add_patch(
            Rectangle(
                (0.06 + i * 0.015, 0.20 + i * 0.025),
                0.17,
                0.08,
                angle=6,
                fill=False,
                edgecolor=COLORS[i],
                linewidth=1,
            )
        )
    ax.text(
        0.03,
        0.08,
        "Adapted logic: serial slices → continuous volume → validation/export; layout is repository-specific.",
        fontsize=7,
        color=GRAY,
    )


def _draw_3d(ax) -> None:
    _panel_label(ax, "B")
    pts, cls, marker = _make_volume()
    for c in np.unique(cls):
        m = cls == c
        ax.scatter(
            pts[m, 0],
            pts[m, 1],
            pts[m, 2],
            s=5,
            alpha=0.35,
            color=COLORS[c],
            label=f"type {c + 1}",
        )
    ax.set_title("3D reconstructed cell architecture", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=38)
    ax.legend(fontsize=6, loc="upper left", frameon=False)


def _draw_virtual_slices(ax) -> None:
    _panel_label(ax, "C")
    pts, cls, marker = _make_volume()
    zs = np.quantile(pts[:, 2], [0.25, 0.50, 0.75])
    ax.set_title("Virtual z-slice marker maps", fontsize=10)
    for i, z in enumerate(zs):
        m = np.abs(pts[:, 2] - z) < 0.16
        xoff = i * 2.4
        ax.scatter(
            pts[m, 0] + xoff,
            pts[m, 1],
            c=marker[m],
            cmap="viridis",
            s=9,
            vmin=0,
            vmax=1,
            linewidths=0,
        )
        ax.add_patch(
            Rectangle((xoff - 1.8, -1.7), 3.4, 3.2, fill=False, edgecolor="#D0D7DE")
        )
        ax.text(xoff, 1.72, f"z={z:.2f}", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 6.5)
    ax.set_ylim(-1.9, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_z_composition(ax) -> None:
    _panel_label(ax, "D")
    pts, cls, _ = _make_volume()
    bins = np.linspace(pts[:, 2].min(), pts[:, 2].max(), 18)
    mids = 0.5 * (bins[:-1] + bins[1:])
    props = []
    for i in range(len(bins) - 1):
        m = (pts[:, 2] >= bins[i]) & (pts[:, 2] < bins[i + 1])
        counts = np.array([(cls[m] == c).sum() for c in np.unique(cls)], dtype=float)
        props.append(counts / max(counts.sum(), 1))
    props = np.vstack(props).T
    ax.stackplot(
        mids,
        props,
        colors=COLORS[: props.shape[0]],
        alpha=0.9,
        labels=[f"type {i + 1}" for i in range(props.shape[0])],
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("z depth")
    ax.set_ylabel("proportion")
    ax.set_title("Cell-type composition along z", fontsize=10)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")


def _draw_marker_heatmap(ax) -> None:
    _panel_label(ax, "E")
    rng = np.random.default_rng(29)
    mat = rng.normal(0, 0.18, size=(6, 9))
    mat += np.linspace(-0.4, 0.5, 9)[None, :]
    mat[[0, 2, 4], [2, 5, 7]] += 0.8
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xticks(range(9), [f"z{i + 1}" for i in range(9)], fontsize=7)
    ax.set_yticks(
        range(6), ["KRT", "COL1A1", "PECAM1", "CD3D", "LYZ", "GFAP"], fontsize=7
    )
    ax.set_title("Marker continuity by virtual depth", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def _draw_metrics(ax) -> None:
    _panel_label(ax, "F")
    vals, source = _load_holdout_summary()
    labels = list(vals.keys())
    numbers = [vals[k] for k in labels]
    y = np.arange(len(labels))
    ax.barh(y, numbers, color=COLORS[: len(labels)])
    ax.set_yticks(y, [x.replace("_", " ") for x in labels], fontsize=7)
    ax.set_xlim(0, max(1.0, max(numbers) * 1.15))
    ax.set_title("Validation metric dashboard", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    for yi, v in zip(y, numbers):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.text(0.0, -0.22, source, transform=ax.transAxes, fontsize=6.5, color=GRAY)


def _draw_neighborhood(ax) -> None:
    _panel_label(ax, "G")
    rng = np.random.default_rng(31)
    mat = rng.random((6, 6))
    mat = (mat + mat.T) / 2
    mat += np.eye(6) * 0.8
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.5)
    names = ["B", "Fib", "Endo", "T", "Tumor", "Myeloid"]
    ax.set_xticks(range(6), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(6), names, fontsize=7)
    ax.set_title("Neighborhood interaction matrix", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def render(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1])
    _draw_workflow(fig.add_subplot(gs[0, :2]))
    _draw_3d(fig.add_subplot(gs[0:2, 2:], projection="3d"))
    _draw_virtual_slices(fig.add_subplot(gs[1, :2]))
    _draw_z_composition(fig.add_subplot(gs[2, 0]))
    _draw_marker_heatmap(fig.add_subplot(gs[2, 1]))
    _draw_metrics(fig.add_subplot(gs[2, 2]))
    _draw_neighborhood(fig.add_subplot(gs[2, 3]))
    fig.suptitle(
        "Aether3D visualization storyboard derived from reference-paper panel logic (clean-room)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    render(args.out)
    try:
        display_path = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = args.out
    print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
