#!/usr/bin/env python3
"""
Aether3D biology figure pack — what the model actually does.

Runs in three modes:

  synthetic  read results/benchmark/volumes/<config>.h5ad from the sweep
  real       auto-detect data/baselines/serial3d_ref/merfish_mouse_hypothalamus/
             merfish_*.h5ad, train a small AetherFlow + reconstruct a
             continuous volume
  all        both

Emits to docs/biology/<mode>/<dataset>/figures/:

  pointcloud_3d_class.{html,png}    interactive 3D scatter coloured by cell class
  pointcloud_3d_gene_<gene>.{html,png}  3D scatter coloured by marker-gene expression
  orthogonal_projections.png        XY / XZ / YZ scatter triptych
  virtual_slice_z<value>.png        2D cross-sections at three Z values
  z_class_composition.png           stacked-area cell-class proportion along Z
  input_vs_reconstruction.png       raw 2D inputs vs reconstructed XZ side view
  gene_trajectory_along_z.png       mean expression of top markers across Z bins
  tissue_mesh.{html,png}            Delaunay surface mesh of reconstructed volume

No new downloads. Run under: conda run --no-capture-output -n dl python ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.visualize._plot_utils import (
    CATEGORICAL_PALETTE,
    class_from_onehot,
    select_markers_by_group,
    stable_categorical_colors,
    to_dense,
)
from scripts.visualize.fig_density_similarity import render_density_similarity
from scripts.visualize.fig_morans_i_scatter import render_morans_i_scatter
from scripts.visualize.fig_multi_z_slice_grid import render_multi_z_slice_grid
from scripts.visualize.fig_marker_heatmap import render_marker_heatmap
from scripts.visualize.fig_neighborhood_matrix import render_neighborhood_matrix
from scripts.visualize.fig_z_density_anchored import render_z_density_anchored
from scripts.visualize.fig_per_section_proportion import render_per_section_proportion

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.modules.aether_flow_module import AetherFlowModule


BASELINE_ROOT = PROJECT_ROOT.parent / "data" / "baselines" / "serial3d_ref" / "merfish_mouse_hypothalamus"
SYNTHETIC_SWEEP = PROJECT_ROOT / "results" / "benchmark" / "volumes"


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARN] CUDA probe failed: {exc}; falling back to CPU.")
        return torch.device("cpu")


# ----------------------------------------------------------------------------
# Mode loaders
# ----------------------------------------------------------------------------

def load_synthetic_volume(config_name: str = "wide") -> AnnData:
    candidate = SYNTHETIC_SWEEP / f"{config_name}.h5ad"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No synthetic volume at {candidate}. "
            "Run scripts/benchmark/run_synthetic_sweep.py first, or pass --synthetic-config."
        )
    print(f"[bio] Loading synthetic volume from {candidate.relative_to(PROJECT_ROOT)}")
    return sc.read_h5ad(candidate)


def list_real_slices() -> List[Path]:
    if not BASELINE_ROOT.exists():
        return []
    return sorted(BASELINE_ROOT.glob("merfish_*.h5ad"))


def reconstruct_real_volume(
    slice_paths: List[Path], device: torch.device, max_cells_per_slice: int = 2000
) -> AnnData:
    print(f"[bio] Reconstructing real volume from {len(slice_paths)} MERFISH slices")
    adatas: List[AnnData] = []
    for i, p in enumerate(slice_paths):
        a = sc.read_h5ad(p)
        if a.n_obs > max_cells_per_slice:
            rng = np.random.default_rng(42 + i)
            idx = rng.choice(a.n_obs, max_cells_per_slice, replace=False)
            a = a[idx].copy()
        a.obs["z_coord"] = float(i * 10.0)
        adatas.append(a)
        print(f"[bio]   slice {i}: {a.n_obs} cells (z={i*10.0})")

    cfg = Aether3DConfig(
        hidden_size=64, depth=2, num_heads=4,
        batch_size=64, max_epochs=3, n_samples_base=1500,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adatas)

    # Train briefly
    model = recon.model.to(device)
    module = AetherFlowModule(cfg, model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(recon.dataset, batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = module.training_step(batch_dev, 0)
            loss.backward(); optimizer.step()
            module.on_train_batch_end()
            epoch_loss += float(loss.item())
        print(f"[bio]   epoch {epoch+1}/{cfg.max_epochs} | loss {epoch_loss / max(len(loader), 1):.4f}")

    recon.model = model.to(torch.device("cpu"))
    recon.ema_model = module.ema_model.to(torch.device("cpu"))
    volume = recon.reconstruct_continuous_volume(adatas, thickness=10.0, num_depths=4)

    # Carry over gene names + a per-cell virtual_depth (already on volume)
    if volume.var_names.size == 0 or volume.var_names[0].startswith("0"):
        volume.var_names = adatas[0].var_names[: volume.n_vars]
    return volume


# ----------------------------------------------------------------------------
# Figure functions
# ----------------------------------------------------------------------------

def _coords(adata: AnnData) -> np.ndarray:
    if "spatial_3d" in adata.obsm:
        return np.asarray(adata.obsm["spatial_3d"])
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])
        z = adata.obs["z_3d"].to_numpy() if "z_3d" in adata.obs else adata.obs["z_coord"].to_numpy()
        return np.hstack([spatial[:, :2], z.reshape(-1, 1)])
    raise KeyError("No spatial_3d or spatial coordinates in volume")


def fig_pointcloud_3d_class(adata: AnnData, html_path: Path, png_path: Path) -> None:
    coords = _coords(adata)
    classes = class_from_onehot(adata)
    if classes is None:
        return
    palette = stable_categorical_colors(classes)

    # Interactive HTML via plotly
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for c in np.unique(classes):
            m = (classes == c)
            fig.add_trace(go.Scatter3d(
                x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
                mode="markers",
                marker=dict(size=2, color=palette[str(c)]),
                name=str(c),
            ))
        fig.update_layout(
            title=f"Aether3D reconstructed volume — {adata.n_obs:,} cells, {len(np.unique(classes))} classes",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=800, height=700,
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path, include_plotlyjs="cdn")
    except Exception as exc:
        print(f"[warn] plotly HTML for class point cloud failed: {exc}")

    # Static PNG via matplotlib
    fig2 = plt.figure(figsize=(6, 5))
    ax = fig2.add_subplot(111, projection="3d")
    for c in np.unique(classes):
        m = (classes == c)
        ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], c=palette[str(c)], s=2, label=str(c))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D point cloud by cell class ({adata.n_obs:,} cells)")
    if len(np.unique(classes)) <= 8:
        ax.legend(fontsize=7, loc="upper left", markerscale=2)
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=150)
    plt.close(fig2)


def fig_pointcloud_3d_gene(adata: AnnData, gene: str, html_path: Path, png_path: Path) -> None:
    if gene not in adata.var_names:
        return
    coords = _coords(adata)
    gi = list(adata.var_names).index(gene)
    vals = to_dense(adata.X)[:, gi]
    vmax = float(np.percentile(vals, 99) + 1e-9)
    vmin = float(np.percentile(vals, 1))

    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=2, color=vals, colorscale="Viridis", cmin=vmin, cmax=vmax,
                        colorbar=dict(title=gene)),
            name=gene,
        )])
        fig.update_layout(
            title=f"3D expression of {gene}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=800, height=700,
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path, include_plotlyjs="cdn")
    except Exception as exc:
        print(f"[warn] plotly HTML for gene point cloud failed: {exc}")

    fig2 = plt.figure(figsize=(6, 5))
    ax = fig2.add_subplot(111, projection="3d")
    sc_h = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=vals, s=2, cmap="viridis", vmin=vmin, vmax=vmax)
    fig2.colorbar(sc_h, ax=ax, label=gene, fraction=0.04)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D expression of {gene}")
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=150)
    plt.close(fig2)


def fig_orthogonal_projections(adata: AnnData, out_path: Path) -> None:
    coords = _coords(adata)
    classes = class_from_onehot(adata)
    if classes is None:
        classes = np.array(["all"] * adata.n_obs)
    palette = stable_categorical_colors(classes)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (xi, yi, lbl) in zip(axes, [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        for c in np.unique(classes):
            m = (classes == c)
            ax.scatter(coords[m, xi], coords[m, yi], c=palette[str(c)], s=1.5, label=str(c))
        ax.set_xlabel(lbl[0]); ax.set_ylabel(lbl[1])
        ax.set_title(f"{lbl} projection")
        ax.set_aspect("equal", adjustable="datalim")
    if len(np.unique(classes)) <= 8:
        axes[-1].legend(fontsize=6, markerscale=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_virtual_slices(adata: AnnData, out_path: Path) -> None:
    coords = _coords(adata)
    z = coords[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    targets = [z_min + frac * (z_max - z_min) for frac in (0.25, 0.5, 0.75)]
    classes = class_from_onehot(adata)
    if classes is None:
        classes = np.array(["all"] * adata.n_obs)
    palette = stable_categorical_colors(classes)
    band = (z_max - z_min) * 0.05

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, t in zip(axes, targets):
        mask = (z >= t - band) & (z <= t + band)
        for c in np.unique(classes):
            m = mask & (classes == c)
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], c=palette[str(c)], s=4, label=str(c))
        ax.set_title(f"Virtual slice at Z={t:.2f}\n({mask.sum()} cells)")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
    if len(np.unique(classes)) <= 8:
        axes[-1].legend(fontsize=6, markerscale=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_z_class_composition(adata: AnnData, out_path: Path, n_bins: int = 25) -> None:
    coords = _coords(adata)
    classes = class_from_onehot(adata)
    if classes is None:
        return
    z = coords[:, 2]
    z_bins = np.linspace(z.min(), z.max(), n_bins + 1)
    centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    bin_idx = np.clip(np.digitize(z, z_bins) - 1, 0, n_bins - 1)
    unique = np.unique(classes)
    counts = np.zeros((len(unique), n_bins), dtype=float)
    for ci, c in enumerate(unique):
        mask = (classes == c)
        for b in np.unique(bin_idx[mask]):
            counts[ci, b] = mask[bin_idx == b].sum() if True else 0
        counts[ci] = np.bincount(bin_idx[mask], minlength=n_bins)
    totals = counts.sum(axis=0) + 1e-9
    fractions = counts / totals
    fractions = np.stack([gaussian_filter1d(row, sigma=1.0) for row in fractions])

    palette = stable_categorical_colors(classes)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.stackplot(centers, fractions, labels=[str(c) for c in unique],
                 colors=[palette[str(c)] for c in unique], alpha=0.9)
    ax.set_xlabel("Reconstructed Z")
    ax.set_ylabel("Cell class proportion")
    ax.set_title("Cell-class composition along Z")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_input_vs_reconstruction(
    input_slices: Optional[List[AnnData]], volume: AnnData, out_path: Path
) -> None:
    coords = _coords(volume)
    z = coords[:, 2]
    z_min, z_max = float(z.min()), float(z.max())

    cols = max(3, len(input_slices) if input_slices else 3)
    fig, axes = plt.subplots(2, cols, figsize=(3.0 * cols, 6), squeeze=False)

    # Row 0: input slices (if available) - XY scatter colored by cell class
    if input_slices:
        for ci, s in enumerate(input_slices[:cols]):
            classes = s.obs["cell_class"].astype(str).to_numpy() if "cell_class" in s.obs else None
            xy = np.asarray(s.obsm["spatial"])[:, :2]
            ax = axes[0, ci]
            if classes is not None:
                palette = stable_categorical_colors(classes)
                for c in np.unique(classes):
                    m = (classes == c)
                    ax.scatter(xy[m, 0], xy[m, 1], c=palette[str(c)], s=2)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#4C72B0")
            zv = s.obs.get("z_coord", 0)
            if hasattr(zv, "iloc"):
                zv = zv.iloc[0]
            ax.set_title(f"Input slice {ci} (z={float(zv):.1f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        for ci in range(len(input_slices), cols):
            axes[0, ci].axis("off")
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "No input slices provided", ha="center", va="center")
            ax.axis("off")

    # Row 1: reconstructed XZ scatter at three matched Z bands
    classes = class_from_onehot(volume)
    palette = stable_categorical_colors(classes) if classes is not None else None
    bands = np.linspace(z_min, z_max, cols + 1)
    for ci in range(cols):
        ax = axes[1, ci]
        m = (z >= bands[ci]) & (z < bands[ci + 1])
        if classes is not None:
            for c in np.unique(classes):
                cm = m & (classes == c)
                if cm.sum() == 0:
                    continue
                ax.scatter(coords[cm, 0], coords[cm, 1], c=palette[str(c)], s=2)
        else:
            ax.scatter(coords[m, 0], coords[m, 1], s=2, color="#C44E52")
        ax.set_title(f"Reconstructed band z=[{bands[ci]:.1f}, {bands[ci+1]:.1f}]\n{m.sum()} virtual cells", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Input 2D slices vs continuous Aether3D reconstruction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_gene_trajectory_along_z(adata: AnnData, out_path: Path, n_bins: int = 20, n_genes: int = 3) -> None:
    coords = _coords(adata)
    z = coords[:, 2]
    z_bins = np.linspace(z.min(), z.max(), n_bins + 1)
    centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    bin_idx = np.clip(np.digitize(z, z_bins) - 1, 0, n_bins - 1)
    classes = class_from_onehot(adata)
    if classes is not None and "cell_class" not in adata.obs:
        adata = adata.copy()
        adata.obs["cell_class"] = classes
    markers_by_group = select_markers_by_group(adata, "cell_class", n_per_group=1) if classes is not None else {}
    chosen = list({g for genes in markers_by_group.values() for g in genes})[:n_genes]
    if not chosen:
        # fallback: top-variance genes
        X = to_dense(adata.X)
        chosen = [adata.var_names[i] for i in np.argsort(X.var(axis=0))[-n_genes:][::-1]]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for ci, gene in enumerate(chosen):
        gi = list(adata.var_names).index(gene)
        vals = to_dense(adata.X)[:, gi]
        means = np.array([vals[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(n_bins)])
        stds  = np.array([vals[bin_idx == b].std()  if (bin_idx == b).any() else np.nan for b in range(n_bins)])
        color = CATEGORICAL_PALETTE[ci % len(CATEGORICAL_PALETTE)]
        ax.plot(centers, means, label=gene, color=color, linewidth=1.6)
        ax.fill_between(centers, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xlabel("Reconstructed Z")
    ax.set_ylabel("Mean gene expression (± 1 SD)")
    ax.set_title(f"Top markers along reconstructed Z (n={len(chosen)})")
    ax.legend(fontsize=7, loc="best")
    ax.grid(linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_tissue_mesh(adata: AnnData, html_path: Path, png_path: Path) -> None:
    coords = _coords(adata)
    if coords.shape[0] < 4:
        return
    # Subsample for tractable Delaunay
    n = min(coords.shape[0], 4000)
    if coords.shape[0] > n:
        rng = np.random.default_rng(42)
        idx = rng.choice(coords.shape[0], n, replace=False)
        coords_s = coords[idx]
    else:
        coords_s = coords
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords_s)
        i = hull.simplices[:, 0]; j = hull.simplices[:, 1]; k = hull.simplices[:, 2]
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Mesh3d(
                x=coords_s[:, 0], y=coords_s[:, 1], z=coords_s[:, 2],
                i=i, j=j, k=k,
                color="#4C72B0", opacity=0.4, name="tissue_hull",
            )])
            fig.update_layout(
                title=f"Tissue convex-hull mesh ({n} vertices, {len(i)} faces)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                width=800, height=700,
            )
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(html_path, include_plotlyjs="cdn")
        except Exception as exc:
            print(f"[warn] plotly mesh HTML failed: {exc}")

        # Matplotlib trisurf snapshot
        fig2 = plt.figure(figsize=(6, 5))
        ax = fig2.add_subplot(111, projection="3d")
        ax.plot_trisurf(coords_s[:, 0], coords_s[:, 1], coords_s[:, 2],
                        triangles=hull.simplices, alpha=0.4, color="#4C72B0", edgecolor="none")
        ax.scatter(coords_s[:, 0], coords_s[:, 1], coords_s[:, 2], s=1, color="#1f1f1f", alpha=0.4)
        ax.set_title("Tissue convex-hull mesh")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        fig2.tight_layout()
        fig2.savefig(png_path, dpi=150)
        plt.close(fig2)
    except Exception as exc:
        print(f"[warn] mesh generation failed: {exc}")


# ----------------------------------------------------------------------------
# Pipeline runners
# ----------------------------------------------------------------------------

def render_figures_for_volume(volume: AnnData, input_slices: Optional[List[AnnData]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, Any] = {}

    html_p = out_dir / "pointcloud_3d_class.html"
    png_p = out_dir / "pointcloud_3d_class.png"
    fig_pointcloud_3d_class(volume, html_p, png_p)
    if png_p.exists():
        figures["pointcloud_3d_class"] = {"png": png_p.name, "html": html_p.name if html_p.exists() else None}

    classes = class_from_onehot(volume)
    if classes is not None:
        markers = select_markers_by_group(volume.copy() if "cell_class" in volume.obs
                                          else AnnData(X=volume.X, obs={"cell_class": classes},
                                                       var=volume.var, obsm=dict(volume.obsm)),
                                          "cell_class", n_per_group=1)
        chosen = list({g for genes in markers.values() for g in genes})[:2]
        figures.setdefault("pointcloud_3d_gene", [])
        for gene in chosen:
            h = out_dir / f"pointcloud_3d_gene_{gene}.html"
            p = out_dir / f"pointcloud_3d_gene_{gene}.png"
            fig_pointcloud_3d_gene(volume, gene, h, p)
            if p.exists():
                figures["pointcloud_3d_gene"].append({"gene": gene, "png": p.name, "html": h.name if h.exists() else None})

    for fn, name in [
        (fig_orthogonal_projections, "orthogonal_projections.png"),
        (fig_virtual_slices, "virtual_slices.png"),
        (fig_z_class_composition, "z_class_composition.png"),
        (fig_gene_trajectory_along_z, "gene_trajectory_along_z.png"),
    ]:
        p = out_dir / name
        fn(volume, p)
        if p.exists():
            figures[name.replace(".png", "")] = name

    p = out_dir / "input_vs_reconstruction.png"
    fig_input_vs_reconstruction(input_slices, volume, p)
    if p.exists():
        figures["input_vs_reconstruction"] = p.name

    html_p = out_dir / "tissue_mesh.html"; png_p = out_dir / "tissue_mesh.png"
    fig_tissue_mesh(volume, html_p, png_p)
    if png_p.exists():
        figures["tissue_mesh"] = {"png": png_p.name, "html": html_p.name if html_p.exists() else None}

    # === Wave 1 ===
    p = out_dir / "density_similarity_bars.png"
    render_density_similarity(volume, input_slices or [], p)
    if p.exists():
        figures["density_similarity"] = p.name

    p = out_dir / "morans_i_scatter.png"
    render_morans_i_scatter(volume, input_slices or [], p)
    if p.exists():
        figures["morans_i_scatter"] = p.name

    p = out_dir / "multi_z_slice_grid.png"
    render_multi_z_slice_grid(volume, input_slices, p)
    if p.exists():
        figures["multi_z_slice_grid"] = p.name

    # === Wave 2 ===
    p = out_dir / "marker_heatmap.png"
    render_marker_heatmap(volume, input_slices or [], p)
    if p.exists():
        figures["marker_heatmap"] = p.name

    p = out_dir / "neighborhood_matrix.png"
    render_neighborhood_matrix(volume, p)
    if p.exists():
        figures["neighborhood_matrix"] = p.name

    z_template = str(out_dir / "z_density_anchored_{anchor}.png")
    rendered = render_z_density_anchored(volume, z_template)
    if rendered:
        figures["z_density_anchored"] = rendered

    p = out_dir / "per_section_proportion.png"
    render_per_section_proportion(volume, input_slices or [], p)
    if p.exists():
        figures["per_section_proportion"] = p.name

    return figures


def write_report(mode_results: Dict[str, Dict[str, Any]], docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# Aether3D Biology Figure Pack")
    md.append("")
    md.append("Aether3D reconstructs continuous 3D tissue volumes from sparse serial 2D sections by "
              "training a multi-modal velocity field on spatial coordinates, gene expression, and "
              "cell-class identity. The reconstructed volume is a regular AnnData object whose features include:")
    md.append("")
    md.append("- **3D coordinates** for every virtual cell (`obsm['spatial_3d']`, `obs['z_3d']`, "
              "`obs['virtual_depth']`) that can be queried at arbitrary depths between input slices;")
    md.append("- **predicted gene expression and cell-class probability** for each virtual cell, enabling "
              "marker analysis, cell-class stratification, and tissue-domain queries directly in 3D;")
    md.append("- a **continuous tissue volume** that supports virtual cross-sections at any Z, arbitrary "
              "orthogonal projections, and surface-mesh extraction.")
    md.append("")
    md.append("This report exercises those outputs on two data sources: a synthetic 3-slice "
              "trajectory (fully reproducible from the sweep artifacts), and an on-disk MERFISH mouse "
              "hypothalamus serial-section dataset that Aether3D reconstructs into a single dense volume. "
              "No online downloads are required.")
    md.append("")
    for mode, runs in mode_results.items():
        md.append(f"## {mode.capitalize()}")
        md.append("")
        for dataset, payload in runs.items():
            md.append(f"### {dataset}")
            md.append("")
            md.append(f"- source: `{payload['source']}`")
            md.append(f"- volume: {payload['n_obs']:,} virtual cells, {payload['n_vars']:,} genes, "
                      f"runtime: {payload['runtime_s']:.1f}s, device: `{payload['device']}`")
            md.append("")
            figs = payload["figures"]
            fig_dir = f"./{mode}/{dataset}/figures"
            def link(name: str) -> str:
                return f"{fig_dir}/{name}"
            if "pointcloud_3d_class" in figs:
                obj = figs["pointcloud_3d_class"]
                md.append("**3D point cloud — cell class**")
                md.append("")
                md.append(f"![{obj['png']}]({link(obj['png'])})")
                if obj.get("html"):
                    md.append(f"\n[interactive HTML]({link(obj['html'])})")
                md.append("")
            if figs.get("pointcloud_3d_gene"):
                md.append("**3D expression of top markers**")
                md.append("")
                for entry in figs["pointcloud_3d_gene"]:
                    md.append(f"![{entry['png']}]({link(entry['png'])})")
                    if entry.get("html"):
                        md.append(f"\n[interactive HTML — {entry['gene']}]({link(entry['html'])})")
                    md.append("")
            for key, label in [
                ("orthogonal_projections",   "**Orthogonal projections (XY / XZ / YZ)**"),
                ("virtual_slices",           "**Virtual cross-sections at three Z values**"),
                ("z_class_composition",      "**Cell-class composition along reconstructed Z**"),
                ("input_vs_reconstruction",  "**Input 2D slices vs continuous Aether3D reconstruction**"),
                ("density_similarity",       "**Per-cell-class 3D density similarity** (reconstructed vs input KDE cosine + cell counts)"),
                ("morans_i_scatter",         "**Per-gene Moran's I** scatter, reconstructed vs input stack"),
                ("multi_z_slice_grid",       "**6-row Z-strata scatter grid** (reconstructed vs nearest input slice)"),
                ("marker_heatmap",           "**Cell-type × marker heatmap** (input stack vs reconstruction)"),
                ("neighborhood_matrix",      "**3D cellular neighborhood enrichment** (z-scored co-localization, perm test)"),
                ("z_density_anchored",       "**Z-density anchored on dominant class** (each class within radius of the anchor)"),
                ("per_section_proportion",   "**Per-section cell-type stacked-bar grid** (inputs vs reconstructed Z-bands)"),
                ("gene_trajectory_along_z",  "**Top markers along the reconstructed Z axis**"),
            ]:
                if key in figs:
                    md.append(label)
                    md.append("")
                    md.append(f"![{key}]({link(figs[key])})")
                    md.append("")
            if "tissue_mesh" in figs:
                obj = figs["tissue_mesh"]
                md.append("**Tissue surface mesh**")
                md.append("")
                md.append(f"![{obj['png']}]({link(obj['png'])})")
                if obj.get("html"):
                    md.append(f"\n[interactive HTML mesh]({link(obj['html'])})")
                md.append("")
            # Glob-driven fallback: catch any figure file on disk we did not embed above
            try:
                fig_dir_p = (docs_dir / mode / dataset / "figures").resolve()
                if fig_dir_p.exists():
                    embedded = set()
                    for v in figs.values():
                        if isinstance(v, str):
                            embedded.add(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, str):
                                    embedded.add(item)
                                elif isinstance(item, dict):
                                    for vv in item.values():
                                        if isinstance(vv, str):
                                            embedded.add(vv)
                        elif isinstance(v, dict):
                            for vv in v.values():
                                if isinstance(vv, str):
                                    embedded.add(vv)
                    leftover = sorted(p.name for p in fig_dir_p.glob("*.png") if p.name not in embedded)
                    if leftover:
                        md.append("**Additional figures (auto-detected on disk)**")
                        md.append("")
                        for fn in leftover:
                            md.append(f"![{fn}]({link(fn)})")
                        md.append("")
            except Exception:
                pass
    md.append("---")
    md.append("")
    md.append("Reproduce with (dl env required for the RTX 5090):")
    md.append("")
    md.append("```bash")
    md.append("conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode all")
    md.append("```")
    md.append("")
    out = docs_dir / "BIOLOGY_REPORT.md"
    out.write_text("\n".join(md))
    return out


def run_synthetic(out_root: Path, config_name: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    volume = load_synthetic_volume(config_name)
    out_dir = out_root / "synthetic" / config_name / "figures"
    figures = render_figures_for_volume(volume, input_slices=None, out_dir=out_dir)
    return {
        "source": str((SYNTHETIC_SWEEP / f"{config_name}.h5ad").relative_to(PROJECT_ROOT)),
        "n_obs": int(volume.n_obs),
        "n_vars": int(volume.n_vars),
        "runtime_s": time.perf_counter() - t0,
        "device": "cpu (precomputed)",
        "figures": figures,
    }


def run_real(out_root: Path, device: torch.device, max_cells_per_slice: int) -> Dict[str, Any]:
    slices = list_real_slices()
    if not slices:
        raise FileNotFoundError(
            f"No MERFISH baseline slices under {BASELINE_ROOT}. Skip --mode real or place files there."
        )
    t0 = time.perf_counter()
    input_adatas = [sc.read_h5ad(p) for p in slices]
    # Stamp z_coord if missing
    for i, a in enumerate(input_adatas):
        if "z_coord" not in a.obs:
            a.obs["z_coord"] = float(i * 10.0)
    volume = reconstruct_real_volume(slices, device=device, max_cells_per_slice=max_cells_per_slice)

    out_volume_dir = PROJECT_ROOT / "results" / "biology" / "real" / "merfish_hypothalamus"
    out_volume_dir.mkdir(parents=True, exist_ok=True)
    volume.write(out_volume_dir / "volume.h5ad")

    out_dir = out_root / "real" / "merfish_hypothalamus" / "figures"
    figures = render_figures_for_volume(volume, input_slices=input_adatas, out_dir=out_dir)
    return {
        "source": str(slices[0].relative_to(PROJECT_ROOT.parent)) if slices[0].is_relative_to(PROJECT_ROOT.parent)
                  else str(slices[0]),
        "n_obs": int(volume.n_obs),
        "n_vars": int(volume.n_vars),
        "runtime_s": time.perf_counter() - t0,
        "device": str(device),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real", "all"], default="all")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "docs" / "biology")
    parser.add_argument("--synthetic-config", default="wide")
    parser.add_argument("--max-cells-per-slice", type=int, default=1500)
    args = parser.parse_args()

    device = get_device()
    print(f"[bio] Device: {device}")
    results: Dict[str, Dict[str, Any]] = {}

    if args.mode in ("synthetic", "all"):
        print("\n[bio] === SYNTHETIC ===")
        results.setdefault("synthetic", {})[args.synthetic_config] = run_synthetic(args.out_root, args.synthetic_config)

    if args.mode in ("real", "all"):
        print("\n[bio] === REAL ===")
        try:
            results.setdefault("real", {})["merfish_hypothalamus"] = run_real(
                args.out_root, device, args.max_cells_per_slice
            )
        except FileNotFoundError as exc:
            print(f"[bio] Skipping real mode: {exc}")

    report = write_report(results, args.out_root)
    print(f"\n[bio] Wrote {report}")
    print(f"[bio] Figures under {args.out_root}")

    (args.out_root / "biology_run.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
