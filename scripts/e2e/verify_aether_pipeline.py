#!/usr/bin/env python3
"""
Deep verification script for Aether3D (mirrors LuminaST's verify script).

- Generates synthetic serial 2D slices with spatial + gene + cell class
- Builds UOT trajectory dataset via AetherReconstructor.setup_data
- Runs a bounded training pass (Trainer(fast_dev_run=1)) so the smoke
  actually exercises train -> reconstruct rather than reconstructing with
  a zero-init untrained model (issue #124)
- Calls reconstruct_continuous_volume and exits nonzero if required output checks fail
"""

import argparse
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import pandas as pd
from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
from aether_3d.data.physical_z import resolve_slice_z


def generate_synthetic_slices(n_slices=3, cells_per_slice=48, n_genes=16, n_classes=4):
    rng = np.random.default_rng(42)
    adatas = []
    cancer_names = [f"Type{i}" for i in range(n_classes)]

    for s in range(n_slices):
        expr = rng.normal(0, 1, (cells_per_slice, n_genes)).astype(np.float32)
        xy = rng.uniform(0, 100, (cells_per_slice, 2)).astype(np.float32)
        labels = rng.choice(cancer_names, cells_per_slice)

        ad = sc.AnnData(X=expr)
        ad.obsm["spatial"] = xy
        ad.obs["cell_class"] = pd.Categorical(labels)
        ad.obs["z_coord"] = float(s) * 10.0  # physical Z
        adatas.append(ad)

    return adatas, cancer_names


def main():
    parser = argparse.ArgumentParser(
        description="Run the Aether3D synthetic smoke pipeline."
    )
    parser.add_argument(
        "--use-real-baseline",
        action="store_true",
        help=(
            "Opt into local baseline h5ad files from data/baselines/ when present. The default is synthetic "
            "data so CI/smoke runs stay bounded and avoid very large UOT category counts."
        ),
    )
    args = parser.parse_args()

    print("=== Deep Aether3D Pipeline Verification ===")

    # Auto-detect real baseline data inside the standalone repo
    # (MERFISH hypothalamus slices from the baseline data directory).
    # parents[2] resolves to <repo>/ regardless of where this script is
    # imported from, so the standalone clone never silently couples to a
    # parent-monorepo layout (issue #16).
    DATA_ROOT = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "baselines"
        / "serial3d_ref"
        / "merfish_mouse_hypothalamus"
    )
    if args.use_real_baseline and DATA_ROOT.exists():
        h5ads = sorted(DATA_ROOT.glob("merfish_*.h5ad"))
        if h5ads:
            print(f"[INFO] Found baseline data at {DATA_ROOT}")
            print(
                f"       Loading {len(h5ads)} real serial slices for E2E verification.\n"
            )
            adatas = [sc.read_h5ad(p) for p in h5ads]
            # Issue #222: source REAL physical inter-slice z from the data
            # (obs['Bregma'] / obs['slice_id'] mm / obsm['spatial3d']); only fall
            # back to a synthetic idx*spacing ladder when absent.
            z_values, z_is_physical = resolve_slice_z(adatas, fallback_spacing=10.0)
            for adata, z in zip(adatas, z_values):
                adata.obs["z_coord"] = float(z)
            print(
                f"       z_is_physical={z_is_physical} "
                f"(z range {min(z_values):.4g}..{max(z_values):.4g})\n"
            )
        else:
            print(
                "Baseline folder exists but no .h5ad files found. Falling back to synthetic."
            )
            adatas, _ = generate_synthetic_slices()
    else:
        if DATA_ROOT.exists():
            print(
                "Real baseline data found locally, but synthetic smoke is the default. Use --use-real-baseline to opt in.\n"
            )
        else:
            print(
                "No real baseline data found locally. Using improved synthetic serial slices.\n"
            )
        adatas, _ = generate_synthetic_slices()

    cfg = Aether3DConfig(
        hidden_size=16,
        depth=1,
        num_heads=2,
        batch_size=16,
        max_epochs=1,
        n_samples_base=48,
        n_samples_volume=48,
    )

    # Build dataset + dim-inferred model through the reconstructor so that
    # spatial/gene/class dims always match the dataset (avoids the historical
    # "shape mismatches in demo model" — fixed via dim inference in
    # AetherReconstructor.setup_data; see issue #124).
    recon = AetherReconstructor(cfg)
    recon.setup_data(adatas)
    assert recon.dataset is not None
    print(f"Trajectory dataset size: {len(recon.dataset)} pairs")

    # Bounded training so the reconstruction below uses an actually-trained
    # model. fast_dev_run=1 keeps the CI runtime minimal (single batch,
    # CPU-only) while still exercising the full train -> reconstruct path
    # the "Deep" E2E smoke promises (issue #124).
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    recon.fit(trainer=trainer)
    print(
        f"Bounded training completed: global_step={trainer.global_step} (fast_dev_run=1)"
    )

    # Reconstruction (now using the trained model attached to `recon`)
    volume = recon.reconstruct_continuous_volume(adatas, thickness=10.0, num_depths=3)

    print("\nReconstructed 3D volume:")
    print(f"  Cells: {volume.n_obs}")
    has_spatial_3d = "spatial_3d" in volume.obsm
    has_z_3d = "z_3d" in volume.obs.columns
    finite_spatial_3d = has_spatial_3d and np.isfinite(volume.obsm["spatial_3d"]).all()
    expected_nonempty = volume.n_obs > 0 and volume.n_vars > 0

    print(f"  Has spatial_3d: {has_spatial_3d}")
    print(f"  Has z_3d: {has_z_3d}")
    print(f"  All finite in spatial_3d: {finite_spatial_3d}")
    print(f"  Non-empty volume matrix: {expected_nonempty}")

    success = bool(
        has_spatial_3d and has_z_3d and finite_spatial_3d and expected_nonempty
    )
    if success:
        print("\n✅ Aether3D bounded reconstruction smoke PASSED")
    else:
        print("\n❌ Aether3D bounded reconstruction smoke FAILED")
    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
