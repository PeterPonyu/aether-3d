"""Distance-band cellular neighborhood heatmap composer skeleton.

Cellular-neighbourhood distance-band heatmap composer for Aether3D.
The composer accepts an enrichment matrix
produced by ``aether_3d.benchmarks.neighborhood.radius_neighborhood_enrichment``
across multiple query labels and renders a (target × neighbor) matrix
with a diverging colormap centered at 1.0 (no enrichment).

The synthetic-mode demo here uses a deterministically generated
enrichment matrix so the rendered PNG is part of CI artifacts; the
real-data demo (Day 7 IMC volume) plugs into the same entry point.

Run as a script: ``python compose_neighborhood_heatmap.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def render_neighborhood_heatmap(
    enrichment_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    title: str = "Distance-band neighborhood enrichment (synthetic placeholder)",
    radius_um: float | None = None,
) -> Path:
    """Render an (R × C) enrichment heatmap with TwoSlopeNorm centered at 1.

    Args:
        enrichment_matrix: (R, C) array; rows = query cell types,
            cols = neighbor cell types. Each entry is observed-over-null
            enrichment (> 1 = over-represented).
        row_labels: length-R labels.
        col_labels: length-C labels.
        out_path: PNG output path.
        title: figure title.
        radius_um: optional radius for the legend stamp.

    Returns:
        ``out_path`` after writing.
    """
    matrix = np.asarray(enrichment_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"enrichment_matrix must be 2-D, got {matrix.shape}")
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise ValueError("enrichment_matrix shape != (len(row_labels), len(col_labels))")

    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmin = float(min(finite.min(), 1.0 / max(finite.max(), 1e-9)))
        vmax = float(max(finite.max(), 1.0 / max(finite.min(), 1e-9)))
        if not (vmin < 1.0 < vmax):
            delta = max(abs(vmax - 1.0), abs(1.0 - vmin), 0.1)
            vmin = 1.0 - delta
            vmax = 1.0 + delta
    else:
        vmin = 0.5
        vmax = 2.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(0.6 * len(col_labels) + 3,
                                    0.5 * len(row_labels) + 2),
                           constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Neighbor cell type")
    ax.set_ylabel("Query cell type")
    suffix = f" (r = {radius_um:g} µm)" if radius_um is not None else ""
    ax.set_title(title + suffix, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, label="obs / null enrichment")
    cbar.ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _synthetic_enrichment(seed: int = 0) -> tuple[np.ndarray, list[str], list[str]]:
    """Deterministic synthetic enrichment matrix for skeleton rendering."""
    rng = np.random.default_rng(seed)
    queries = ["T", "B", "Myeloid", "Endothelial", "Fibroblast"]
    neighbors = ["T", "B", "Myeloid", "Endothelial", "Fibroblast", "Stromal"]
    base = rng.uniform(0.6, 1.5, size=(len(queries), len(neighbors)))
    # Self-enrichment on the diagonal to mimic the known biology.
    for i, q in enumerate(queries):
        if q in neighbors:
            base[i, neighbors.index(q)] *= 2.5
    return base, queries, neighbors


def main() -> None:
    """Skeleton entrypoint: render a synthetic heatmap + manifest sidecar."""
    here = Path(__file__).parent
    sys.path.insert(0, str(here.parent / "src"))

    out_dir = here / "figures"
    matrix, rows, cols = _synthetic_enrichment(seed=0)
    png = render_neighborhood_heatmap(
        matrix, rows, cols,
        out_dir / "fig_neighborhood_heatmap_skeleton.png",
        radius_um=20.0,
    )
    manifest = {
        "title": "Distance-band neighborhood enrichment (synthetic placeholder)",
        "source": "compose_neighborhood_heatmap._synthetic_enrichment",
        "real_data_contract": (
            "Replace _synthetic_enrichment() with a loader that calls "
            "aether_3d.benchmarks.neighborhood.radius_neighborhood_enrichment "
            "over each query label and stacks per_celltype_enrichment "
            "values into a (target × neighbor) matrix; see "
            "docs/DATA_PREP_CALENDAR.md Day 7 for real IMC inputs."
        ),
        "rendered_path": str(png.name),
    }
    (out_dir / "fig_neighborhood_heatmap_skeleton.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
