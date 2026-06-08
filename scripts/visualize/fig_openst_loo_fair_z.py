"""openST/HNSCC GSE251926 leave-one-out result figure — fair-z view.

Renders the real 17-interior-holdout leave-one-out result for aether-3d on the
openST head-and-neck serial sections, comparing the **continuous** flow-matching
reconstruction against the 2.5D baselines (nearest-slice, linear-interp,
stacking-2.5d) on the identical real held-out slices.

Fair-z honesty (companion to #298's physical-µm restoration on MERFISH):
openST carries **no** physical inter-section spacing — ``obs['z_coord']`` is the
raw section ordinal ``[2,3,4,5,6,7,9,11,17,18,19,23,24,25,26,28,33,34,36]``,
which is NON-UNIFORM and NOT microns (card ``openst_hnscc_gse251926.yaml``;
#291 keeps real-µm sourcing open). So the per-holdout panels place each held-out
slice at its **true ordinal depth** (non-uniform x), never on a fabricated
uniform µm grid. The axis is labelled accordingly.

Verdict the figure carries: continuous **loses** to the 2.5D baselines on the
spatial-coherence metrics (Moran's-I, Betti-0) — i.e. CLAIM_LEDGER row-1 stays
``planned (contradicted)``. This is a model-repair signal, not a reporting tweak.

Reads only the committed metrics JSON; the ordinal z mapping is read from the
processed h5ad when present and otherwise falls back to the documented card
ordinals (kept in sync with the card constant below).

Run:
    python -m scripts.visualize.fig_openst_loo_fair_z
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
METRICS_JSON = REPO / "results/openst_hnscc_gse251926/aether-3d/metrics.json"
PROCESSED_H5AD = REPO.parent / "data/processed/openst_hnscc_gse251926/serial_sections.h5ad"
OUT_PNG = REPO / "results/figures/openst_loo_fair_z.png"

# Documented section ordinals from data/cards/openst_hnscc_gse251926.yaml
# (obs['n_section']); fallback when the 8.87 GB processed artifact is absent.
CARD_SECTION_ORDINALS: tuple[float, ...] = (
    2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36,
)

# method key -> (display label, colour). continuous is highlighted red.
METHODS: dict[str, tuple[str, str]] = {
    "continuous": ("Aether continuous", "#C44E52"),
    "linear_interp": ("linear-interp 2.5D", "#4C72B0"),
    "nearest_slice": ("nearest-slice 2.5D", "#55A868"),
    "stacking_2.5d": ("stacking 2.5D", "#937860"),
}
# (metric key, display, higher_is_better)
METRICS: list[tuple[str, str, bool]] = [
    ("morans_i_agreement_top100", "Moran's I agreement\n(top-100 genes)", True),
    ("betti0_stability", "Betti-0 stability", True),
    ("sliced_wasserstein_2d", "Sliced-Wasserstein\n(2D, ↓)", False),
    ("chamfer_distance", "Chamfer distance (↓)", False),
    ("coord_rmse", "Coord RMSE (↓)", False),
]


def _ordered_ordinals() -> tuple[np.ndarray, str]:
    """Sorted unique section ordinals; from the h5ad if present, else the card."""
    try:
        import h5py

        with h5py.File(PROCESSED_H5AD, "r") as f:
            node = f["obs/z_coord"]
            arr = node["codes"][:] if isinstance(node, h5py.Group) else node[:]
        return np.unique(np.asarray(arr, dtype=float)), "processed h5ad obs['z_coord']"
    except (OSError, KeyError, ImportError):
        return np.array(CARD_SECTION_ORDINALS, dtype=float), "data card (h5ad absent)"


def _per_slice(metrics: dict, method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (slice_index, value) arrays for one method/metric, sorted by index."""
    pat = re.compile(rf"^contrast_slice(\d+)_{re.escape(method)}_{re.escape(metric)}$")
    pairs = [(int(m.group(1)), v) for k, v in metrics.items() if (m := pat.match(k))]
    pairs.sort()
    idx = np.array([p for p, _ in pairs], dtype=int)
    val = np.array([v for _, v in pairs], dtype=float)
    return idx, val


def render(metrics: dict, out_path: Path = OUT_PNG) -> Path:
    ordered_z, z_source = _ordered_ordinals()

    # Holdout count from the data; interior holdout k (1-based) sits at ordered_z[k]
    # (endpoints ordered_z[0] / ordered_z[-1] are never held out).
    n_hold = len({
        int(m.group(1))
        for k in metrics
        if (m := re.match(r"^contrast_slice(\d+)_continuous_morans_i_agreement_top100$", k))
    })
    if len(ordered_z) < n_hold + 2:
        raise ValueError(
            f"need >= {n_hold + 2} ordered sections for {n_hold} interior holdouts; "
            f"got {len(ordered_z)} from {z_source}"
        )
    holdout_z = ordered_z[1 : n_hold + 1]  # physical-ordinal depth of each holdout

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, len(METRICS), height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.34)

    # --- Top: per-holdout coherence vs TRUE ordinal depth (fair-z line panels) ---
    for col, (mkey, mlabel, _hib) in enumerate(METRICS[:2]):
        ax = fig.add_subplot(gs[0, col * 2 : col * 2 + 2] if len(METRICS) >= 4 else gs[0, col])
        for method, (label, colour) in METHODS.items():
            idx, val = _per_slice(metrics, method, mkey)
            if idx.size == 0:
                continue
            zpos = ordered_z[idx]  # idx is 1-based holdout -> ordinal depth
            lw, ms, z = (2.4, 6, 3) if method == "continuous" else (1.4, 4, 2)
            ax.plot(zpos, val, "-o", color=colour, lw=lw, ms=ms, label=label, zorder=z)
        ax.set_title(f"{mlabel.splitlines()[0]} per held-out slice", fontsize=10)
        ax.set_xlabel("section ordinal z  (non-uniform spacing — NOT physical µm)", fontsize=8)
        ax.set_ylabel(mlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        # rug marking the true non-uniform ordinal positions
        for z in holdout_z:
            ax.axvline(z, color="#bbbbbb", lw=0.4, ymax=0.04, zorder=0)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # --- Bottom: mean ± std over all holdouts, one bar panel per metric ---
    method_keys = list(METHODS)
    for col, (mkey, mlabel, hib) in enumerate(METRICS):
        ax = fig.add_subplot(gs[1, col])
        means, stds, colours = [], [], []
        for method in method_keys:
            _, val = _per_slice(metrics, method, mkey)
            means.append(float(np.mean(val)) if val.size else np.nan)
            stds.append(float(np.std(val)) if val.size else 0.0)
            colours.append(METHODS[method][1])
        x = np.arange(len(method_keys))
        ax.bar(x, means, yerr=stds, color=colours, capsize=2, width=0.7,
               error_kw={"elinewidth": 0.8})
        # highlight winner direction
        best = (np.nanargmax(means) if hib else np.nanargmin(means))
        ax.bar(x[best], means[best], color="none", edgecolor="black", lw=1.4, width=0.7)
        ax.set_title(mlabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[m][0].replace(" ", "\n", 1) for m in method_keys],
                           fontsize=6, rotation=0)
        ax.tick_params(axis="y", labelsize=7)
        arrow = "higher better" if hib else "lower better"
        ax.text(0.5, 0.97, arrow, transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="#555")

    cont_moran = np.mean(_per_slice(metrics, "continuous", "morans_i_agreement_top100")[1])
    near_moran = np.mean(_per_slice(metrics, "nearest_slice", "morans_i_agreement_top100")[1])
    fig.suptitle(
        "openST/HNSCC GSE251926 — real 17-interior-slice leave-one-out\n"
        f"Continuous flow LOSES to 2.5D baselines on spatial coherence "
        f"(Moran's I {cont_moran:.3f} vs nearest-slice {near_moran:.3f})  →  "
        "CLAIM_LEDGER row-1 = planned (contradicted)",
        fontsize=11, y=0.99,
    )
    fig.text(0.5, 0.005,
             f"z = section ordinal ({z_source}); non-uniform, NOT physical µm (#291). "
             "Baselines are honest lower bounds, not reproductions of any published 2.5D method.",
             ha="center", fontsize=7, color="#555")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if not METRICS_JSON.exists():
        raise SystemExit(f"missing metrics JSON: {METRICS_JSON}")
    metrics = json.loads(METRICS_JSON.read_text())["metrics"]
    out = render(metrics)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
