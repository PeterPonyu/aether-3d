#!/usr/bin/env python3
"""Composed Aether3D main-claim figure.

This is a figure-level logic chain: serial 2D slices become a claim-gated
continuous 3D reconstruction, then virtual slices, z-depth biology, metrics,
and neighborhood structure gate any fidelity/scale claims. Clean-room inspired
by the supplied DEEPSPATIAL paper's figure grammar without copying artwork.
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
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "aether_composed_main_claim.png"
COL = ["#2F6FA7", "#1AA39A", "#E7903C", "#C94C4C", "#7D5FB2", "#5BA85B"]
GRAY = "#65717E"
INK = "#1F2328"


def _load_metrics():
    path = PROJECT_ROOT / "results" / "holdout_validation_metrics.json"
    vals = {
        "coord RMSE↓": 0.18,
        "Chamfer↓": 0.23,
        "marker cont.↑": 0.82,
        "prop corr↑": 0.88,
        "domain ARI↑": 0.71,
    }
    source = "demo/planning: replace with virtual-slice holdout metrics"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            mapping = {
                "coordinate_rmse": "coord RMSE↓",
                "chamfer": "Chamfer↓",
                "marker_continuity": "marker cont.↑",
                "proportion_corr": "prop corr↑",
                "domain_ari": "domain ARI↑",
            }
            found = {
                label: float(payload[k])
                for k, label in mapping.items()
                if k in payload and isinstance(payload[k], (int, float))
            }
            if found:
                vals.update(found)
                source = f"local-small: {path.relative_to(PROJECT_ROOT)}"
        except Exception:
            pass
    return vals, source


def _volume(seed=17):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-1.4, -0.55, -0.75], [0, 0.75, 0], [1.2, -0.25, 0.72], [-0.45, -1.05, 0.55]]
    )
    pts = []
    cls = []
    for i, c in enumerate(centers):
        block = rng.multivariate_normal(
            c, np.diag([0.14 + 0.03 * i, 0.11 + 0.02 * i, 0.18]), size=230
        )
        pts.append(block)
        cls += [i] * len(block)
    pts = np.vstack(pts)
    cls = np.asarray(cls)
    marker = np.exp(
        -((pts[:, 0] - 1.0) ** 2 + (pts[:, 1] + 0.2) ** 2 + (pts[:, 2] - 0.7) ** 2)
        / 0.55
    )
    return pts, cls, marker


def _label(ax, s):
    fn = getattr(ax, "text2D", ax.text)
    fn(-0.06, 1.06, s, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            fc=color + "22",
            ec=color,
            lw=1.25,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK
    )


def _arrow(ax, a, b, text=None):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.15)
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.035,
            text,
            ha="center",
            fontsize=7,
            color=GRAY,
        )


def _badge(ax, text, xy=(0.98, 0.02)):
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D0D7DE", lw=0.7),
    )


def spine(ax):
    ax.set_axis_off()
    steps = [
        ("1 Serial slices", COL[0]),
        ("2 Learn z-flow", COL[1]),
        ("3 Reconstruct 3D", COL[2]),
        ("4 Slice validate", COL[4]),
        ("5 Biology across z", COL[5]),
        ("6 Claim gate", COL[3]),
    ]
    x0 = 0.02
    w = 0.145
    gap = 0.018
    for i, (txt, c) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, 0.27, w, 0.46, txt, c)
        if i < len(steps) - 1:
            _arrow(
                ax,
                (x + w, 0.50),
                (x + w + gap, 0.50),
                "therefore" if i in (1, 3) else None,
            )
    ax.text(
        0.02,
        0.86,
        "Figure-level logic chain: continuity claims require geometry, virtual-slice, biological, and metric gates",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.08,
        "Evidence tier: demo/planning unless holdout metrics and reconstructed volumes are present. No copied source-paper layouts or scale claims.",
        fontsize=8,
        color=GRAY,
    )


def panel_input(ax):
    ax.set_axis_off()
    _label(ax, "A")
    ax.set_title("Serial 2D input and z-gap problem", loc="left", fontsize=11)
    for i, (x, y, c) in enumerate(
        [
            (0.12, 0.22, COL[0]),
            (0.17, 0.34, COL[1]),
            (0.22, 0.46, COL[2]),
            (0.27, 0.58, COL[3]),
        ]
    ):
        ax.add_patch(Rectangle((x, y), 0.35, 0.12, angle=7, fill=False, ec=c, lw=1.3))
        ax.text(x + 0.38, y + 0.05, f"z{i}", fontsize=7, color=c)
    _box(ax, 0.58, 0.54, 0.32, 0.20, "observed XY\n+ genes/proteins", COL[0])
    _box(ax, 0.58, 0.22, 0.32, 0.20, "missing depths\nneed virtual tissue", COL[3])
    _arrow(ax, (0.42, 0.49), (0.58, 0.64))
    _arrow(ax, (0.42, 0.31), (0.58, 0.32))
    ax.text(
        0.03,
        0.05,
        "Claim question: can discrete slices support continuous 3D inference?",
        fontsize=7.5,
        color=GRAY,
    )


def panel_workflow(ax):
    ax.set_axis_off()
    _label(ax, "B")
    ax.set_title("Aether3D reconstruction workflow", loc="left", fontsize=11)
    boxes = [
        (0.04, "align +\nvalidate", COL[0]),
        (0.28, "UOT / flow\ntrain", COL[1]),
        (0.52, "continuous\nvolume", COL[2]),
        (0.76, "export +\nclaim gates", COL[3]),
    ]
    for x, t, c in boxes:
        _box(ax, x, 0.46, 0.18, 0.25, t, c)
    for x in [0.22, 0.46, 0.70]:
        _arrow(ax, (x, 0.585), (x + 0.06, 0.585))
    _box(
        ax,
        0.30,
        0.13,
        0.38,
        0.20,
        "virtual slicing\ncoronal · sagittal · horizontal",
        COL[4],
    )
    _arrow(ax, (0.61, 0.46), (0.52, 0.33), "validate")
    _badge(ax, "method + gate")


def panel_3d(ax):
    _label(ax, "C")
    pts, cls, _ = _volume()
    for c in np.unique(cls):
        m = cls == c
        ax.scatter(
            pts[m, 0],
            pts[m, 1],
            pts[m, 2],
            s=5,
            alpha=0.35,
            color=COL[c],
            label=f"type {c + 1}",
        )
    ax.view_init(elev=22, azim=38)
    ax.set_title("3D cellular architecture", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=6, frameon=False, loc="upper left")


def panel_slices(ax):
    _label(ax, "D")
    pts, cls, marker = _volume()
    zs = np.quantile(pts[:, 2], [0.25, 0.5, 0.75])
    for i, z in enumerate(zs):
        m = np.abs(pts[:, 2] - z) < 0.16
        off = i * 2.35
        ax.scatter(
            pts[m, 0] + off,
            pts[m, 1],
            c=marker[m],
            cmap="viridis",
            s=9,
            vmin=0,
            vmax=1,
            linewidths=0,
        )
        ax.add_patch(Rectangle((off - 1.75, -1.7), 3.35, 3.2, fill=False, ec="#D0D7DE"))
        ax.text(off, 1.72, f"virtual z={z:.2f}", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 6.35)
    ax.set_ylim(-1.9, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Virtual slice validation", fontsize=11)
    _badge(ax, "replace with held-out slices")


def panel_zbiology(ax):
    _label(ax, "E")
    pts, cls, _ = _volume()
    bins = np.linspace(pts[:, 2].min(), pts[:, 2].max(), 18)
    mids = 0.5 * (bins[:-1] + bins[1:])
    props = []
    for i in range(len(bins) - 1):
        m = (pts[:, 2] >= bins[i]) & (pts[:, 2] < bins[i + 1])
        counts = np.array([(cls[m] == c).sum() for c in np.unique(cls)], float)
        props.append(counts / max(counts.sum(), 1))
    props = np.vstack(props).T
    ax.stackplot(
        mids,
        props,
        colors=COL[: props.shape[0]],
        alpha=0.9,
        labels=[f"type {i + 1}" for i in range(props.shape[0])],
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("z depth")
    ax.set_ylabel("proportion")
    ax.set_title("Cell-type continuity along z", fontsize=11)
    ax.legend(fontsize=6, ncol=2, frameon=False)


def panel_metrics(ax):
    _label(ax, "F")
    vals, source = _load_metrics()
    labels = list(vals)
    nums = [vals[k] for k in labels]
    y = np.arange(len(labels))
    ax.barh(y, nums, color=COL[: len(labels)])
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, max(1, max(nums) * 1.15))
    ax.grid(axis="x", ls=":", alpha=0.35)
    ax.set_title("Fidelity metric gate", fontsize=11)
    for yi, v in zip(y, nums):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=7)
    _badge(ax, source)


def panel_neighborhood(ax):
    _label(ax, "G")
    rng = np.random.default_rng(12)
    mat = rng.random((6, 6))
    mat = (mat + mat.T) / 2 + np.eye(6) * 0.8
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.5)
    names = ["B", "Fib", "Endo", "T", "Tumor", "Myeloid"]
    ax.set_xticks(range(6), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(6), names, fontsize=7)
    ax.set_title("Neighborhood consistency gate", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def render(out_path: Path = DEFAULT_OUT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, height_ratios=[0.48, 1, 1, 1])
    spine(fig.add_subplot(gs[0, :]))
    panel_input(fig.add_subplot(gs[1, 0:2]))
    panel_workflow(fig.add_subplot(gs[1, 2:4]))
    panel_3d(fig.add_subplot(gs[1:3, 4:6], projection="3d"))
    panel_slices(fig.add_subplot(gs[2, 0:4]))
    panel_zbiology(fig.add_subplot(gs[3, 0:2]))
    panel_metrics(fig.add_subplot(gs[3, 2:4]))
    panel_neighborhood(fig.add_subplot(gs[3, 4:6]))
    fig.suptitle(
        "Aether3D composed main-claim figure: from serial slices to claim-gated continuous 3D reconstruction",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    render(args.out)
    try:
        display_path = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = args.out
    print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
