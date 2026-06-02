#!/usr/bin/env python3
"""Leave-one-out holdout reconstruction on REAL MERFISH serial slices (Aether3D).

This script replaces the previous synthetic-slice path. It loads the five cached
MERFISH mouse-hypothalamus slices, intersects their gene panels, derives a REAL
physical ``z_coord`` per slice from the data's anterior-posterior Bregma metadata
(``obs['slice_id']`` mm in the cached baseline; falls back to a configurable
synthetic spacing with ``z_is_physical=False`` only when absent — issue #222),
and runs a leave-one-out holdout
reconstruction over the three INTERIOR slices (indices 1, 2, 3; the boundary
slices 0 and 4 are never held out because interpolation needs neighbours on both
sides).

For each held-out interior slice ``h`` the continuous volume is reconstructed from
the two bracketing neighbour slices ``(h-1, h+1)`` (the model is fit on that
neighbour pair), and the virtual slice at the midpoint (``virtual_depth = 0.5``)
is compared against the real held-out slice. The comparison is fully
SELF-SUPERVISED reconstruction (the "truth" is the held-out part of the same real
data) so all emitted metrics are intrinsic — no external/ground-truth label is
consumed.

Results are emitted via the vendored uniform results contract
(``aether_3d.results_contract.write_results``) to ``results/aether-3d/``:
``metrics.json`` + ``run_metadata.json`` + ``outputs/`` (per-holdout
reconstruction arrays + the reconstructed volumes). A back-compat alias is also
written to ``results/holdout_validation_metrics.json`` (a regression test pins it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import reduce
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

# Add src and project root to pythonpath
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.data.physical_z import resolve_slice_z
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.modules.aether_flow_module import AetherFlowModule
from aether_3d import results_contract
from aether_3d.benchmarks.metrics import (
    celltype_distribution_cosine,
    celltype_proportion_spearman,
    domain_ari_nmi,
    morans_i_agreement,
    sliced_wasserstein_2d,
)
from aether_3d.benchmarks.contract import (
    VolumeAdapterInput,
    _chamfer_distance,
    _coord_rmse,
)
from aether_3d.benchmarks.topology import (
    betti_zero_stability,
    flow_divergence_map,
    divergence_summary,
    velocity_anisotropy,
)
from aether_3d.benchmarks.spatial_coherence import chaos_score, pas_score
from aether_3d.benchmarks.adapters import (
    LinearInterpAdapter,
    NearestSliceAdapter,
    Stack25DAdapter,
)

# Default location of the five cached real MERFISH slices.
DEFAULT_DATA_DIR = (
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/baselines/serial3d_ref/merfish_mouse_hypothalamus"
)
# Interior slices that can be held out (boundary slices 0 and 4 cannot, since
# the interpolation needs a neighbour on both sides).
INTERIOR_SLICES = (1, 2, 3)
# Fallback spacing (issue #222): used ONLY when no physical inter-slice z
# metadata (obs['Bregma'] / obs['slice_id'] / obsm['spatial3d']) is available.
# Configurable via --fallback-spacing; never assume this is the real spacing.
SLICE_SPACING = 10.0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            # Test if CUDA works (catches RTX 5090 capabilities mismatch).
            test_tensor = torch.zeros(1, device="cuda")
            _ = torch.relu(test_tensor)
            return torch.device("cuda")
        except Exception as e:  # pragma: no cover - hardware-dependent
            print(f"[WARNING] CUDA is available but failed test execution: {e}")
            print("Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def load_real_merfish_slices(
    data_dir: str,
    n_slices: int = 5,
    max_cells: int | None = None,
    seed: int = 42,
    fallback_spacing: float = SLICE_SPACING,
):
    """Load the real MERFISH slices, intersect gene panels, inject ``z_coord``.

    Returns ``(slices, dataset_paths, n_dropped_genes, n_shared_genes,
    z_is_physical)`` where ``slices`` is a list of AnnData ordered by slice index
    (== physical z order), every slice subset to the SAME sorted shared gene
    panel, with ``.obs['z_coord']`` injected from REAL physical metadata when
    present (issue #222) and ``.obsm['spatial']`` / ``.obs['cell_class']``
    preserved. ``z_is_physical`` is True when the injected z came from a physical
    field (e.g. Bregma / slice_id mm), False when the synthetic
    ``idx * fallback_spacing`` ladder was used.

    The reconstructor builds a dense ``n0 x n1`` UOT coupling whose flattened
    size feeds ``torch.multinomial`` (hard cap 2**24 categories). With ~5.5k
    cells per slice ``n0*n1`` is ~30M > 2**24, so ``max_cells`` seeded-subsamples
    each real slice to a tractable size (default 4000 ⇒ 16M < 2**24).
    """
    base = Path(data_dir)
    dataset_paths = [str(base / f"merfish_{i}.h5ad") for i in range(n_slices)]
    raw = [ad.read_h5ad(p) for p in dataset_paths]

    if max_cells is not None:
        rng = np.random.default_rng(seed)
        capped = []
        for a in raw:
            if a.n_obs > max_cells:
                keep = np.sort(rng.choice(a.n_obs, max_cells, replace=False))
                a = a[keep].copy()
            capped.append(a)
        raw = capped

    # Gene-panel intersection across slices: shared, sorted panel so every slice
    # exposes an identical ``gene_dim`` (the model derives gene_dim from
    # sample["g0"].shape[0]).
    shared = reduce(np.intersect1d, [s.var_names.to_numpy() for s in raw])
    shared = np.sort(shared)
    panels = [set(s.var_names) for s in raw]
    union = set().union(*panels)
    n_dropped = len(union) - len(shared)

    slices = []
    for idx, a in enumerate(raw):
        sub = a[:, list(shared)].copy()
        # Densify X to plain float32 ndarray (downstream model + numpy paths
        # assume dense arrays; MERFISH counts are already dense float32).
        X = sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        sub.X = X.astype(np.float32)
        # Schema sanity: spatial coords + cell_class must be present.
        if "spatial" not in sub.obsm:
            raise KeyError(f"slice {idx} missing .obsm['spatial']")
        if "cell_class" not in sub.obs:
            raise KeyError(f"slice {idx} missing .obs['cell_class']")
        sub.obs["cell_class"] = pd.Categorical(sub.obs["cell_class"].astype(str))
        slices.append(sub)

    # Issue #222: derive each slice's physical z from real metadata
    # (obs['Bregma'] / obs['slice_id'] mm / obsm['spatial3d']); only fall back to
    # the synthetic idx*fallback_spacing ladder when no physical field exists.
    z_values, z_is_physical = resolve_slice_z(slices, fallback_spacing=fallback_spacing)
    for sub, z in zip(slices, z_values):
        sub.obs["z_coord"] = float(z)

    return slices, dataset_paths, int(n_dropped), int(len(shared)), z_is_physical


def reconstruct_holdout(neighbor_slices, held_slice, cfg, device, seed):
    """Train on ``neighbor_slices`` and reconstruct the virtual mid-slice.

    ``neighbor_slices`` are the two slices on either side of the held-out slice;
    they bracket the held-out physical z so the ``virtual_depth = 0.5`` plane
    lands on the held-out slice's z. Returns ``(virtual_slice, volume)``.
    """
    dataset = SerialSliceTrajectoryDataset(neighbor_slices, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    sample = dataset[0]
    model = MultiModalVelocityField(
        spatial_dim=2,
        gene_dim=sample["g0"].shape[0],
        num_classes=len(dataset.label_encoder.classes_),
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
    ).to(device)

    module = AetherFlowModule(cfg, model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    model.train()
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = module.training_step(batch_dev, 0)
            loss.backward()
            optimizer.step()
            module.on_train_batch_end()
            epoch_loss += loss.item()
        print(
            f"    epoch {epoch + 1:02d}/{cfg.max_epochs:02d} | "
            f"loss {epoch_loss / max(len(loader), 1):.4f}"
        )

    # Reconstruct between the two neighbour slices. Thickness is the physical
    # z-gap between the bracketing neighbours (issue #222: derived from the real
    # z_coord injected on each slice, not a hard-coded 2*SLICE_SPACING), so the
    # midpoint d=0.5 maps to the held-out slice z regardless of unit/spacing.
    z_lo = float(neighbor_slices[0].obs["z_coord"].iloc[0])
    z_hi = float(neighbor_slices[1].obs["z_coord"].iloc[0])
    thickness = abs(z_hi - z_lo)
    recon = AetherReconstructor(cfg)
    recon.setup_data(neighbor_slices)
    recon.model = model.to(torch.device("cpu"))  # reconstructor runs on CPU
    volume = recon.reconstruct_continuous_volume(
        neighbor_slices, thickness=thickness, num_depths=3
    )

    # Decode predicted cell-class labels from the velocity-field class head
    # (argmax over the softmax probabilities in obsm["cell_class_vel"]) so the
    # cell-type proportion metrics have a reconstruction-side labelling.
    if "cell_class_vel" in volume.obsm and recon.dataset is not None:
        classes = np.asarray(recon.dataset.label_encoder.classes_)
        pred_idx = np.argmax(np.asarray(volume.obsm["cell_class_vel"]), axis=1)
        volume.obs["cell_class"] = pd.Categorical(classes[pred_idx])

    virtual_slice = volume[np.isclose(volume.obs["virtual_depth"], 0.5)].copy()
    return virtual_slice, volume


def _kmeans_labels(X, n_clusters, seed):
    """Seeded KMeans domain labels over an expression matrix (NaN-safe)."""
    from sklearn.cluster import KMeans

    n = X.shape[0]
    k = int(min(n_clusters, n))
    if k < 2:
        return np.zeros(n, dtype=np.int64)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(np.asarray(X))
    return km.labels_.astype(np.int64)


def _as_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)


def evaluate_holdout(virtual_slice, held_slice, seed=42, n_domains=5):
    """Compute the self-supervised reconstruction metrics for one holdout.

    Returns a flat metric dict. The four headline keys
    (``gene_profile_pearson`` etc.) are preserved verbatim for back-compat; the
    baseline-style geometry + molecular + topology metrics are appended.
    All metrics here are SELF-SUPERVISED: the held-out real slice is the truth.
    Flow-divergence + velocity anisotropy are DESCRIPTIVE (z is synthetic).
    """
    # A. Gene-level mean expression profile correlation & MSE.
    pred_mean_profile = np.mean(virtual_slice.X, axis=0)
    true_mean_profile = np.mean(held_slice.X, axis=0)
    gene_pearson, _ = pearsonr(pred_mean_profile, true_mean_profile)
    gene_mse = float(np.mean((pred_mean_profile - true_mean_profile) ** 2))

    # B. Cell-level nearest-neighbour comparison: match each virtual cell to the
    # nearest spatial neighbour in the real held-out slice, compare profiles.
    pred_coords = np.asarray(virtual_slice.obsm["spatial"], dtype=np.float32)
    true_coords = np.asarray(held_slice.obsm["spatial"], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(true_coords)
    _, indices = nn.kneighbors(pred_coords)
    indices = indices.squeeze()

    pred_expr = _as_dense(virtual_slice.X)
    true_expr_all = _as_dense(held_slice.X)
    true_expr = true_expr_all[indices]
    cell_mses = np.mean((pred_expr - true_expr) ** 2, axis=1)
    cell_pearsons = []
    for i in range(len(pred_expr)):
        p_val, _ = pearsonr(pred_expr[i], true_expr[i])
        if not np.isnan(p_val):
            cell_pearsons.append(p_val)

    out = {
        "gene_profile_pearson": float(gene_pearson),
        "gene_profile_mse": gene_mse,
        "cell_level_mean_pearson": (
            float(np.mean(cell_pearsons)) if cell_pearsons else float("nan")
        ),
        "cell_level_mean_mse": float(np.mean(cell_mses)),
    }

    # C. baseline-style geometry + molecular metrics (self-supervised:
    # reconstruction vs the held-out real slice).
    out["morans_i_agreement_top100"] = morans_i_agreement(
        X_truth=true_expr_all,
        coords_truth=true_coords,
        X_recon=pred_expr,
        coords_recon=pred_coords,
        top_k_hvg=100,
    )
    out["chamfer_distance"] = _chamfer_distance(pred_coords, true_coords)
    out["coord_rmse"] = _coord_rmse(pred_coords, true_coords)
    out["sliced_wasserstein_2d"] = sliced_wasserstein_2d(
        pred_coords, true_coords, seed=seed
    )

    ari_nmi = domain_ari_nmi(
        X_truth=true_expr_all,
        X_recon=pred_expr,
        coords_truth=true_coords,
        coords_recon=pred_coords,
        n_clusters=n_domains,
        seed=seed,
    )
    out["domain_ari"] = ari_nmi.get("ari", float("nan"))
    out["domain_nmi"] = ari_nmi.get("nmi", float("nan"))

    # D. Cell-type proportion agreement (real cell_class labels on both sides).
    if "cell_class" in held_slice.obs and "cell_class" in virtual_slice.obs:
        t_lab = held_slice.obs["cell_class"].astype(str).tolist()
        r_lab = virtual_slice.obs["cell_class"].astype(str).tolist()
        out["celltype_proportion_spearman"] = celltype_proportion_spearman(t_lab, r_lab)
        out["celltype_distribution_cosine"] = celltype_distribution_cosine(t_lab, r_lab)
    else:
        out["celltype_proportion_spearman"] = float("nan")
        out["celltype_distribution_cosine"] = float("nan")

    # E. Betti-0 topology stability (connected-component preservation).
    out["betti0_stability"] = betti_zero_stability(true_coords, pred_coords)

    # F. CHAOS + PAS spatial-coherence of recon domains vs truth domains
    #    (field-standard Dong 2025; lower = more coherent). Domains are seeded
    #    KMeans clusters of expression on each side.
    truth_dom = _kmeans_labels(true_expr_all, n_domains, seed)
    recon_dom = _kmeans_labels(pred_expr, n_domains, seed)
    out["chaos_truth"] = chaos_score(true_coords, truth_dom)
    out["chaos_recon"] = chaos_score(pred_coords, recon_dom)
    out["pas_truth"] = pas_score(true_coords, truth_dom)
    out["pas_recon"] = pas_score(pred_coords, recon_dom)

    # G. Flow-divergence (mass conservation) + velocity anisotropy.
    #    DESCRIPTIVE only — z is synthetic (idx*spacing), so the flow field's
    #    cross-slice component is not physically grounded.
    if "velocity" in virtual_slice.obsm:
        vel = np.asarray(virtual_slice.obsm["velocity"], dtype=np.float32)
        if vel.shape == pred_coords.shape:
            div = flow_divergence_map(pred_coords, vel, grid_size=16)
            div_summary = divergence_summary(div)
            out["flow_mean_abs_divergence"] = div_summary["mean_abs_divergence"]
            out["flow_max_abs_divergence"] = div_summary["max_abs_divergence"]
            out["flow_rms_divergence"] = div_summary["rms_divergence"]
            out["velocity_anisotropy"] = velocity_anisotropy(vel)
        else:
            out["flow_mean_abs_divergence"] = float("nan")
            out["flow_max_abs_divergence"] = float("nan")
            out["flow_rms_divergence"] = float("nan")
            out["velocity_anisotropy"] = float("nan")
    else:
        out["flow_mean_abs_divergence"] = float("nan")
        out["flow_max_abs_divergence"] = float("nan")
        out["flow_rms_divergence"] = float("nan")
        out["velocity_anisotropy"] = float("nan")

    return out


def evaluate_25d_contrast(slices, held_idx, virtual_slice, held_slice, seed=42):
    """Run the 2.5D baselines (nearest-slice, linear-interp, and the clean-room
    stack-2.5d virtual-slice baseline #221) on the SAME real holdout and
    contrast against the continuous flow reconstruction.

    Returns ``{baseline_name: {metric: value}}`` plus a ``continuous`` entry, so
    the emitted table shows continuous-recon vs naive stacking on identical
    real data (the baseline's core 2.5D-vs-continuous claim).
    """
    true_coords = np.asarray(held_slice.obsm["spatial"], dtype=np.float32)
    true_expr = _as_dense(held_slice.X)

    def _score(pred_coords, pred_expr):
        return {
            "chamfer_distance": _chamfer_distance(pred_coords, true_coords),
            "coord_rmse": _coord_rmse(pred_coords, true_coords),
            "sliced_wasserstein_2d": sliced_wasserstein_2d(
                pred_coords, true_coords, seed=seed
            ),
            "morans_i_agreement_top100": morans_i_agreement(
                X_truth=true_expr,
                coords_truth=true_coords,
                X_recon=pred_expr,
                coords_recon=pred_coords,
                top_k_hvg=100,
            ),
            "betti0_stability": betti_zero_stability(true_coords, pred_coords),
        }

    contrast = {
        "continuous": _score(
            np.asarray(virtual_slice.obsm["spatial"], dtype=np.float32),
            _as_dense(virtual_slice.X),
        )
    }

    # Build the adapter input: the same slice list, with held_idx held out.
    inp = VolumeAdapterInput(
        slices=slices,
        held_out_indices=[held_idx],
        z_key="z_coord",
        spatial_key="spatial",
        label_key="cell_class",
        seed=seed,
    )
    for adapter in (NearestSliceAdapter(), LinearInterpAdapter(), Stack25DAdapter()):
        try:
            visible = inp.visible_slices()
            volume = adapter._reconstruct(visible, inp)
            z_target = inp.truth_z_values()[0]
            v_z = volume.obs["z_coord"].astype(float).values
            window = np.abs(v_z - z_target) < 0.5
            if not window.any():
                window = np.ones(volume.n_obs, dtype=bool)
            v_slice = volume[window]
            p_coords = np.asarray(v_slice.obsm["spatial"], dtype=np.float32)
            p_expr = _as_dense(v_slice.X)
            contrast[adapter.name] = _score(p_coords, p_expr)
        except Exception as exc:  # pragma: no cover - baseline robustness
            contrast[adapter.name] = {"status": f"error:{type(exc).__name__}: {exc}"}

    return contrast


def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")
    seed = args.seed
    max_cells = args.max_cells if args.max_cells and args.max_cells > 0 else None
    args.max_cells = max_cells
    t_start = time.time()

    # 1. Load the REAL 5-slice MERFISH dataset (NOT synthetic).
    print("\n[INFO] Loading REAL MERFISH serial slices...")
    slices, dataset_paths, n_dropped, n_shared, z_is_physical = load_real_merfish_slices(
        args.data_dir,
        n_slices=5,
        max_cells=args.max_cells,
        seed=seed,
        fallback_spacing=args.fallback_spacing,
    )
    z_source = "PHYSICAL (real metadata)" if z_is_physical else (
        f"SYNTHETIC (idx*{args.fallback_spacing} fallback)"
    )
    print(f"  z source: {z_source}; z_is_physical={z_is_physical}")
    for idx, s in enumerate(slices):
        print(
            f"  slice {idx}: {s.shape[0]} cells x {s.shape[1]} genes "
            f"| z={float(s.obs['z_coord'].iloc[0])}"
        )
    print(f"  shared gene panel: {n_shared} genes ({n_dropped} dropped)")

    cfg = Aether3DConfig(
        seed=seed,
        hidden_size=64,
        depth=3,
        num_heads=4,
        batch_size=128,
        max_epochs=args.max_epochs,
        n_samples_base=1500,
    )

    # 2. Leave-one-out holdout over the interior slices.
    per_holdout = {}
    per_holdout_contrast = {}
    outputs_arrays = {}
    held_list = (
        list(INTERIOR_SLICES) if not args.single_holdout else [args.single_holdout]
    )
    for h in held_list:
        print(f"\n=== Holdout interior slice {h} (fit on neighbours h-1, h+1) ===")
        neighbor_slices = [slices[h - 1], slices[h + 1]]
        held_slice = slices[h]
        virtual_slice, volume = reconstruct_holdout(
            neighbor_slices, held_slice, cfg, device, seed
        )
        print(
            f"  reconstructed virtual slice: {virtual_slice.shape[0]} cells "
            f"vs real held-out {held_slice.shape[0]} cells"
        )
        m = evaluate_holdout(virtual_slice, held_slice, seed=seed)
        m["holdout_slice_id"] = float(h)
        per_holdout[h] = m
        print(
            f"  gene_profile_pearson={m['gene_profile_pearson']:.4f} "
            f"cell_level_mean_pearson={m['cell_level_mean_pearson']:.4f} "
            f"morans_i_agreement={m['morans_i_agreement_top100']:.4f} "
            f"chamfer={m['chamfer_distance']:.4f} "
            f"domain_ari={m['domain_ari']:.4f}"
        )

        # 2.5D-vs-continuous contrast on the SAME real holdout.
        contrast = evaluate_25d_contrast(slices, h, virtual_slice, held_slice, seed=seed)
        per_holdout_contrast[h] = contrast
        for name, sc in contrast.items():
            if "chamfer_distance" in sc:
                print(
                    f"    [2.5D contrast] {name:14s} "
                    f"chamfer={sc['chamfer_distance']:.4f} "
                    f"morans_i={sc['morans_i_agreement_top100']:.4f}"
                )

        outputs_arrays[f"holdout_{h}_pred"] = np.asarray(
            virtual_slice.X, dtype=np.float32
        )
        outputs_arrays[f"holdout_{h}_pred_spatial"] = np.asarray(
            virtual_slice.obsm["spatial"], dtype=np.float32
        )
        outputs_arrays[f"holdout_{h}_true"] = np.asarray(
            held_slice.X, dtype=np.float32
        )
        outputs_arrays[f"holdout_{h}_volume"] = np.asarray(
            volume.X, dtype=np.float32
        )
        # Persist the per-cell velocities the reconstructor integrated so the
        # flow-divergence + anisotropy results are reproducible from outputs/.
        if "velocity" in virtual_slice.obsm:
            outputs_arrays[f"holdout_{h}_velocity"] = np.asarray(
                virtual_slice.obsm["velocity"], dtype=np.float32
            )
        if "velocity" in volume.obsm:
            outputs_arrays[f"holdout_{h}_volume_velocity"] = np.asarray(
                volume.obsm["velocity"], dtype=np.float32
            )

    runtime_s = time.time() - t_start

    # 3. Assemble metrics: per-slice + leave-one-out aggregate.
    # Full per-holdout metric inventory (headline + baseline-style suite).
    PER_SLICE_KEYS = (
        "gene_profile_pearson",
        "gene_profile_mse",
        "cell_level_mean_pearson",
        "cell_level_mean_mse",
        "morans_i_agreement_top100",
        "chamfer_distance",
        "coord_rmse",
        "sliced_wasserstein_2d",
        "domain_ari",
        "domain_nmi",
        "celltype_proportion_spearman",
        "celltype_distribution_cosine",
        "betti0_stability",
        "chaos_truth",
        "chaos_recon",
        "pas_truth",
        "pas_recon",
        "flow_mean_abs_divergence",
        "flow_max_abs_divergence",
        "flow_rms_divergence",
        "velocity_anisotropy",
    )
    metrics: dict[str, float | None] = {}
    for h, m in per_holdout.items():
        for key in PER_SLICE_KEYS:
            metrics[f"slice{h}_{key}"] = m[key]

    # 2.5D-vs-continuous contrast: emit continuous vs each baseline per holdout.
    CONTRAST_KEYS = (
        "chamfer_distance",
        "coord_rmse",
        "sliced_wasserstein_2d",
        "morans_i_agreement_top100",
        "betti0_stability",
    )
    for h, contrast in per_holdout_contrast.items():
        for method, sc in contrast.items():
            method_key = method.replace("-", "_")
            for key in CONTRAST_KEYS:
                if key in sc:
                    metrics[f"contrast_slice{h}_{method_key}_{key}"] = sc[key]

    if len(per_holdout) > 1:
        # Leave-one-out aggregate = mean across the held-out interior slices.
        for key in PER_SLICE_KEYS:
            vals = [
                per_holdout[h][key]
                for h in per_holdout
                if np.isfinite(per_holdout[h][key])
            ]
            metrics[f"loo_{key}_mean"] = float(np.mean(vals)) if vals else None
        # Headline LOO gene Pearson (named per the plan).
        metrics["loo_gene_pearson_mean"] = metrics["loo_gene_profile_pearson_mean"]

        # LOO-mean 2.5D contrast per baseline (mean over holdouts).
        contrast_methods = set()
        for contrast in per_holdout_contrast.values():
            contrast_methods.update(contrast.keys())
        for method in sorted(contrast_methods):
            method_key = method.replace("-", "_")
            for key in CONTRAST_KEYS:
                vals = [
                    per_holdout_contrast[h][method].get(key)
                    for h in per_holdout_contrast
                    if method in per_holdout_contrast[h]
                    and isinstance(per_holdout_contrast[h][method].get(key), float)
                    and np.isfinite(per_holdout_contrast[h][method].get(key))
                ]
                metrics[f"loo_contrast_{method_key}_{key}_mean"] = (
                    float(np.mean(vals)) if vals else None
                )
    else:
        # Single-holdout fallback path: no LOO aggregate.
        metrics["loo_gene_pearson_mean"] = None
        only = held_list[0]
        # Surface the single holdout's headline values un-prefixed too.
        metrics["gene_profile_pearson"] = per_holdout[only]["gene_profile_pearson"]
        metrics["cell_level_mean_pearson"] = per_holdout[only][
            "cell_level_mean_pearson"
        ]

    n_holdout = len(per_holdout)
    z_desc = (
        "physical (real Bregma/slice_id metadata)"
        if z_is_physical
        else f"SYNTHETIC fallback idx*{args.fallback_spacing} (no physical metadata found)"
    )
    # Flow-divergence / velocity-anisotropy are only "descriptive due to
    # synthetic z" when z is NOT physical; with real spacing they are physically
    # grounded (issue #222 acceptance criterion).
    flow_clause = (
        " DESCRIPTIVE (z synthetic = idx*fallback_spacing, no physical "
        "cross-slice grounding): flow_*_divergence and velocity_anisotropy."
        if not z_is_physical
        else " flow_*_divergence and velocity_anisotropy are PHYSICALLY GROUNDED "
        "(z derived from real inter-slice spacing; z_is_physical=true)."
    )
    notes = (
        f"Real 5-slice MERFISH leave-one-out holdout over interior slices "
        f"{sorted(per_holdout.keys())} ({n_holdout} reconstruction passes); "
        f"shared gene panel {n_shared} genes ({n_dropped} dropped on "
        f"intersection); z source: {z_desc} (z_is_physical={z_is_physical}); "
        f"per-slice cell cap for reconstruction pairing: "
        f"{max_cells if max_cells else 'none'} "
        f"(keeps n0*n1 < 2**24 for torch.multinomial); "
        f"MERFISH counts used as-is (no normalization). "
        f"Metrics are intrinsic self-supervised reconstruction "
        f"(predict a held-out real slice from its neighbours). "
        f"SELF-SUPERVISED (held-out real-slice truth): gene_profile_*, "
        f"cell_level_*, morans_i_agreement_top100, chamfer_distance, coord_rmse, "
        f"sliced_wasserstein_2d, domain_ari/nmi, celltype_proportion_spearman, "
        f"celltype_distribution_cosine, betti0_stability, chaos_*, pas_*, and the "
        f"2.5D-vs-continuous contrast_* keys (continuous flow vs nearest-slice / "
        f"linear-interp baselines on the identical real holdout)." + flow_clause
    )
    if n_holdout == 1:
        notes += " Single-holdout fallback; loo_gene_pearson_mean=null."

    # 4. Emit via the vendored uniform contract.
    n_obs_primary = int(slices[INTERIOR_SLICES[1]].shape[0])  # central slice (2)
    # Issue #222: z_is_physical gates every physical-spacing-dependent metric.
    z_caveat = (
        "flow-divergence (flow_*_divergence) and velocity_anisotropy are "
        "DESCRIPTIVE only: z is synthetic (idx*fallback_spacing), so the velocity "
        "field's cross-slice component is not physically grounded"
        if not z_is_physical
        else "z derived from real inter-slice spacing (z_is_physical=true): "
        "flow_*_divergence and velocity_anisotropy ARE physically grounded"
    )
    run_metadata = {
        "dataset_paths": dataset_paths,
        "n_obs": n_obs_primary,
        "n_vars": n_shared,
        "seed": seed,
        "runtime_s": runtime_s,
        "device": str(device),
        "deterministic": False,
        "num_threads": torch.get_num_threads(),
        "reproducibility_level": "seeded",
        "normalization": {"applied": False, "method": "none"},
        "interpretability": {
            "model_is_learned": True,
            # Issue #222: machine-readable flag gating physical-spacing-dependent
            # metrics/figures. True iff z_coord came from real metadata (Bregma/
            # slice_id mm / spatial3d); False when the synthetic fallback ladder
            # was used. Nested under interpretability because the uniform results
            # contract only passes through this recognized subtree verbatim.
            "z_is_physical": bool(z_is_physical),
            "z_fallback_spacing": float(args.fallback_spacing),
            "encoder": (
                "multi-modal flow-matching velocity field (DiT-style) trained "
                "per holdout via UOT-coupled serial-slice trajectories"
            ),
            "domain_assignment": "n/a (reconstruction task, no domain labels)",
            "caveats": [
                "leave-one-out reconstruction is self-supervised: the held-out "
                "real slice is the target; no external ground-truth label used",
                f"trained for {args.max_epochs} epochs per holdout for tractable "
                "wall time; metrics scale with epoch budget",
                z_caveat,
                "CHAOS/PAS domains are seeded KMeans clusters of expression "
                "(no external domain labels); chaos_*/pas_* characterise spatial "
                "coherence of recon vs truth domains, not a GT-domain agreement",
            ],
            "metric_provenance": {
                "self_supervised_holdout_truth": [
                    "gene_profile_pearson",
                    "gene_profile_mse",
                    "cell_level_mean_pearson",
                    "cell_level_mean_mse",
                    "morans_i_agreement_top100",
                    "chamfer_distance",
                    "coord_rmse",
                    "sliced_wasserstein_2d",
                    "domain_ari",
                    "domain_nmi",
                    "celltype_proportion_spearman",
                    "celltype_distribution_cosine",
                    "betti0_stability",
                    "chaos_truth",
                    "chaos_recon",
                    "pas_truth",
                    "pas_recon",
                    "contrast_*",
                ],
                # Flow/velocity metrics depend on the cross-slice z component;
                # they are descriptive iff z is synthetic, physical otherwise
                # (issue #222).
                (
                    "descriptive_z_synthetic"
                    if not z_is_physical
                    else "physical_z_grounded"
                ): [
                    "flow_mean_abs_divergence",
                    "flow_max_abs_divergence",
                    "flow_rms_divergence",
                    "velocity_anisotropy",
                ],
            },
        },
        "notes": notes,
    }

    written = results_contract.write_results(
        project="aether-3d",
        dataset_card_id=results_contract.dataset_card_id(dataset_paths),
        metrics=metrics,
        outputs={
            "reconstructed_volume.npz": "outputs/reconstructed_volume.npz",
            "holdout_pred.npz": "outputs/holdout_pred.npz",
        },
        run_metadata=run_metadata,
        results_dir=args.results_root,
    )

    # 5. Write native output artifacts into outputs/.
    outputs_dir = written["outputs_dir"]
    np.savez_compressed(outputs_dir / "reconstructed_volume.npz", **outputs_arrays)
    # holdout_pred.npz: just the per-holdout predicted virtual slices + truth.
    pred_only = {
        k: v
        for k, v in outputs_arrays.items()
        if k.endswith("_pred")
        or k.endswith("_true")
        or k.endswith("_pred_spatial")
        or k.endswith("_velocity")
    }
    np.savez_compressed(outputs_dir / "holdout_pred.npz", **pred_only)

    print("\nLeave-one-out holdout summary:")
    for h, m in sorted(per_holdout.items()):
        print(
            f"  slice {h}: gene_pearson={m['gene_profile_pearson']:.4f} "
            f"cell_pearson={m['cell_level_mean_pearson']:.4f}"
        )
    if metrics.get("loo_gene_pearson_mean") is not None:
        print(f"  LOO gene_pearson_mean={metrics['loo_gene_pearson_mean']:.4f}")
    print(f"\nContract written to: {written['results_dir']}")
    print(f"  metrics.json:      {written['metrics']}")
    print(f"  run_metadata.json: {written['run_metadata']}")
    print(f"  outputs/:          {outputs_dir}")

    # 6. Back-compat alias (a regression test pins this file's existence).
    alias_path = project_root / "results" / "holdout_validation_metrics.json"
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = json.loads(written["metrics"].read_text())
    with open(alias_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"  back-compat alias: {alias_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real 5-slice MERFISH leave-one-out holdout reconstruction."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Number of flow-matching training epochs per holdout pass",
    )
    parser.add_argument(
        "--fallback-spacing",
        dest="fallback_spacing",
        type=float,
        default=SLICE_SPACING,
        help=(
            "Inter-slice z spacing used ONLY when no physical metadata "
            "(obs['Bregma']/obs['slice_id']/obsm['spatial3d']) is present; sets "
            "z_is_physical=False (issue #222)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_cells",
        type=int,
        default=4000,
        help=(
            "Seeded per-slice cell cap for the reconstruction UOT pairing "
            "(n0*n1 must stay < 2**24 for torch.multinomial). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory holding merfish_{0..4}.h5ad",
    )
    parser.add_argument(
        "--results_root",
        default=str(project_root / "results"),
        help="Root results dir; the contract writes <root>/aether-3d/",
    )
    parser.add_argument(
        "--single_holdout",
        type=int,
        default=None,
        choices=list(INTERIOR_SLICES),
        help="Fallback: hold out only this interior slice (no LOO aggregate)",
    )
    main(parser.parse_args())
