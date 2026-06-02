#!/usr/bin/env python3
"""
Aether3D synthetic configuration sweep with held-out slice validation.

Generates synthetic serial slices, trains each refined model variant on
Slice 0 + Slice 2, interpolates the missing Slice 1 (Z=10), and reports
quality metrics + resource usage per config.

Outputs:

  results/benchmark/aether_sweep_<TS>.json    (per-config records)
  results/benchmark/aether_sweep_latest.json  (latest copy)
  results/benchmark/curves/<config>.json      (per-epoch flow loss)
  results/benchmark/volumes/<config>.h5ad     (reconstructed virtual volume)

Run in the dl env (RTX 5090 sm_120 requires CUDA 13 wheels):

  conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scanpy as sc
import torch
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.data_flow.generate_serial_slices import generate_synthetic_serial_slices

from aether_3d.benchmarks.metrics import gene_pearson_fidelity
from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.modules.aether_flow_module import AetherFlowModule


@dataclass
class SweepConfig:
    name: str
    hidden_size: int
    depth: int
    num_heads: int
    max_epochs: int
    batch_size: int = 64
    n_samples_base: int = 1500


DEFAULT_SWEEP: List[SweepConfig] = [
    SweepConfig("tiny",  hidden_size=32,  depth=2, num_heads=2, max_epochs=4),
    SweepConfig("small", hidden_size=64,  depth=2, num_heads=2, max_epochs=4),
    SweepConfig("wide",  hidden_size=128, depth=2, num_heads=4, max_epochs=4),
    SweepConfig("deep",  hidden_size=64,  depth=4, num_heads=4, max_epochs=4),
]


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARN] CUDA probe failed ({exc}); falling back to CPU.")
        return torch.device("cpu")


def count_params(modules: List[torch.nn.Module]) -> int:
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
    return total


def evaluate_volume(virtual_slice: sc.AnnData, truth: sc.AnnData) -> Dict[str, float]:
    # NOTE (issue #130): `gene_profile_pearson` is the correlation of the two
    # slices' BULK (per-gene mean) expression profiles — a bulk metric that is
    # invariant to where cells are placed, so it must NOT be read as a headline
    # quality score on its own. It is reported alongside the spatially-matched
    # per-cell / per-gene Pearson + RMSE (the metrics that actually reflect
    # cell-level reconstruction) via metrics.gene_pearson_fidelity.
    pred_mean = np.mean(virtual_slice.X, axis=0)
    true_mean = np.mean(truth.X, axis=0)
    gene_p, _ = pearsonr(pred_mean, true_mean)
    gene_mse = float(np.mean((pred_mean - true_mean) ** 2))

    pred_coords = virtual_slice.obsm["spatial"]
    true_coords = truth.obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(true_coords)
    _, idx = nn.kneighbors(pred_coords)
    idx = idx.squeeze()

    pred_expr = virtual_slice.X
    true_expr = truth.X[idx]

    cell_mse = float(np.mean(np.mean((pred_expr - true_expr) ** 2, axis=1)))
    cell_p = []
    for i in range(pred_expr.shape[0]):
        v, _ = pearsonr(pred_expr[i], true_expr[i])
        if not np.isnan(v):
            cell_p.append(v)
    cell_pearson = float(np.mean(cell_p)) if cell_p else 0.0

    fidelity = gene_pearson_fidelity(
        X_recon=np.asarray(pred_expr, dtype=np.float32),
        coords_recon=np.asarray(pred_coords, dtype=np.float32),
        X_truth=np.asarray(truth.X, dtype=np.float32),
        coords_truth=np.asarray(true_coords, dtype=np.float32),
    )

    return {
        # Bulk slice-mean profile correlation — insensitive to spatial layout.
        "gene_profile_pearson": float(gene_p),
        "bulk_slice_mean_pearson": fidelity["bulk_slice_mean_pearson"],
        "gene_profile_mse": gene_mse,
        # Spatially-matched, cell-level metrics (the meaningful ones).
        "cell_level_mean_pearson": cell_pearson,
        "per_cell_gene_pearson": fidelity["per_cell_gene_pearson"],
        "per_gene_pearson": fidelity["per_gene_pearson"],
        "cell_level_mean_mse": cell_mse,
        "per_cell_gene_rmse": fidelity["per_cell_gene_rmse"],
    }


def run_one(
    cfg: SweepConfig,
    slices: List[sc.AnnData],
    heldout: sc.AnnData,
    device: torch.device,
    out_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    aether_cfg = Aether3DConfig(
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        n_samples_base=cfg.n_samples_base,
    )

    dataset = SerialSliceTrajectoryDataset(slices, aether_cfg)
    loader = DataLoader(dataset, batch_size=aether_cfg.batch_size, shuffle=True)
    sample = dataset[0]

    model = MultiModalVelocityField(
        spatial_dim=2,
        gene_dim=sample["g0"].shape[0],
        num_classes=len(dataset.label_encoder.classes_),
        patch_size=aether_cfg.patch_size,
        hidden_size=aether_cfg.hidden_size,
        depth=aether_cfg.depth,
        num_heads=aether_cfg.num_heads,
    ).to(device)

    module = AetherFlowModule(aether_cfg, model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=aether_cfg.lr, weight_decay=aether_cfg.weight_decay)

    flow_curve: List[float] = []
    t0 = time.perf_counter()
    model.train()
    for epoch in range(aether_cfg.max_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = module.training_step(batch_dev, 0)
            loss.backward()
            optimizer.step()
            module.on_train_batch_end()
            epoch_loss += float(loss.item())
        flow_curve.append(epoch_loss / max(len(loader), 1))
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    recon = AetherReconstructor(aether_cfg)
    recon.setup_data(slices)
    recon.model = model.to(torch.device("cpu"))
    volume = recon.reconstruct_continuous_volume(slices, thickness=20.0, num_depths=3)
    t_recon = time.perf_counter() - t0

    virtual_slice = volume[np.isclose(volume.obs["virtual_depth"], 0.5)].copy()
    metrics = evaluate_volume(virtual_slice, heldout)

    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    n_params = count_params([model])

    vol_path = out_dir / "volumes" / f"{cfg.name}.h5ad"
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    volume.write(vol_path)

    curve_path = out_dir / "curves" / f"{cfg.name}.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({"flow": flow_curve}, indent=2))

    record = {
        "config": asdict(cfg),
        "device": str(device),
        "n_params": n_params,
        "wall_seconds": {
            "flow_train": t_train,
            "reconstruct": t_recon,
            "total": t_train + t_recon,
        },
        "peak_gpu_mem_mb": peak_mb,
        "metrics": metrics,
        "data": {
            "train_slice_shapes": [list(s.shape) for s in slices],
            "heldout_shape": list(heldout.shape),
            "virtual_slice_shape": list(virtual_slice.shape),
            "volume_cells": int(volume.n_obs),
        },
        "volume_path": str(vol_path.relative_to(PROJECT_ROOT)),
        "loss_curve_path": str(curve_path.relative_to(PROJECT_ROOT)),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "benchmark")
    parser.add_argument("--cells-per-slice", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"[bench] Device: {device}")
    print(f"[bench] Output dir: {args.out_dir}")

    all_slices, classes = generate_synthetic_serial_slices(
        n_slices=3,
        cells_per_slice=args.cells_per_slice,
        n_genes=args.n_genes,
        n_classes=args.n_classes,
        seed=args.seed,
        slice_spacing=10.0,
    )
    slice_0, heldout, slice_2 = all_slices
    train_slices = [slice_0, slice_2]
    print(f"[bench] Train slices: {slice_0.shape}, {slice_2.shape}")
    print(f"[bench] Held-out slice: {heldout.shape} at z={heldout.obs['z_coord'].iloc[0]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for cfg in DEFAULT_SWEEP:
        print(f"\n[bench] --- {cfg.name} ---")
        rec = run_one(cfg, train_slices, heldout, device, args.out_dir, args.seed)
        print(
            f"[bench]   total={rec['wall_seconds']['total']:.2f}s "
            f"peak_gpu_mb={rec['peak_gpu_mem_mb']} "
            f"params={rec['n_params']:,}"
        )
        for k, v in rec["metrics"].items():
            print(f"[bench]     {k}: {v:.4f}")
        records.append(rec)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = args.out_dir / f"aether_sweep_{ts}.json"
    latest = args.out_dir / "aether_sweep_latest.json"
    payload = {
        "timestamp": ts,
        "device": str(device),
        "data_settings": {
            "cells_per_slice": args.cells_per_slice,
            "n_genes": args.n_genes,
            "n_classes": args.n_classes,
            "seed": args.seed,
            "slice_spacing": 10.0,
        },
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    latest.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] Wrote {out_json}")
    print(f"[bench] Wrote {latest}")


if __name__ == "__main__":
    main()
