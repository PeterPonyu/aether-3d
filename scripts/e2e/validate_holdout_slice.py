#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

# Add src and project root to pythonpath
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.modules.aether_flow_module import AetherFlowModule
from aether_3d.core.aether_reconstructor import AetherReconstructor

def get_device():
    if torch.cuda.is_available():
        try:
            # Test if CUDA works (catches RTX 5090 capabilities mismatch)
            test_tensor = torch.zeros(1, device="cuda")
            _ = torch.relu(test_tensor)
            return torch.device("cuda")
        except Exception as e:
            print(f"[WARNING] CUDA is available but failed test execution: {e}")
            print("Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # 1. Generate synthetic serial slices (3 slices: Z=0, Z=10, Z=20)
    print("\n[INFO] Generating synthetic serial slices...")
    from scripts.data_flow.generate_serial_slices import generate_synthetic_serial_slices
    slices, classes = generate_synthetic_serial_slices(
        n_slices=3,
        cells_per_slice=400,
        n_genes=32,
        n_classes=4,
        seed=42,
        slice_spacing=10.0
    )
    
    slice_0 = slices[0]
    slice_1_heldout = slices[1]
    slice_2 = slices[2]
    
    print(f"  Slice 0 (Train): {slice_0.shape[0]} cells, Z = {slice_0.obs['z_coord'].iloc[0]}")
    print(f"  Slice 1 (Heldout): {slice_1_heldout.shape[0]} cells, Z = {slice_1_heldout.obs['z_coord'].iloc[0]}")
    print(f"  Slice 2 (Train): {slice_2.shape[0]} cells, Z = {slice_2.obs['z_coord'].iloc[0]}")

    # 2. Setup training configuration and dataset (on Slice 0 and Slice 2 only)
    cfg = Aether3DConfig(
        hidden_size=64,
        depth=3,
        num_heads=4,
        batch_size=128,
        max_epochs=args.max_epochs,
        n_samples_base=1500,
    )
    
    train_slices = [slice_0, slice_2]
    dataset = SerialSliceTrajectoryDataset(train_slices, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # 3. Initialize multi-modal velocity field and module
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 4. Train the Flow matching network on Slice 0 -> Slice 2 pairs
    print(f"\n--- Training Aether3D Flow Matching ({args.max_epochs} epochs) ---")
    model.train()
    for epoch in range(args.max_epochs):
        epoch_loss = 0.0
        for batch in loader:
            # Move batch to device
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = module.training_step(batch_dev, 0)
            loss.backward()
            optimizer.step()
            module.on_train_batch_end()
            
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1:02d}/{args.max_epochs:02d} | Loss: {epoch_loss / len(loader):.4f}")
        
    # 5. Reconstruct continuous volume (Z=0 to Z=20)
    print("\n--- Running Virtual Slice Interpolation (d = 0.5) ---")
    recon = AetherReconstructor(cfg)
    recon.setup_data(train_slices)
    recon.model = model.to(torch.device("cpu"))  # Reconstructor runs on CPU for numpy compatibility
    
    # We reconstruct with num_depths = 3 (d=0.0, d=0.5, d=1.0)
    # thickness is 20.0 (distance between Slice 0 and Slice 2)
    volume = recon.reconstruct_continuous_volume(train_slices, thickness=20.0, num_depths=3)
    
    # Filter the virtual cells at Z = 10.0 (virtual_depth = 0.5)
    virtual_slice = volume[np.isclose(volume.obs["virtual_depth"], 0.5)].copy()
    print(f"  Interpolated Virtual Slice: {virtual_slice.shape[0]} cells at Z = 10.0")

    # 6. Evaluate interpolation quality against heldout Slice 1
    print("\n--- Evaluating Hold-Out Validation Quality ---")
    
    # A. Gene-level comparison (Mean expression profile correlation & MSE)
    pred_mean_profile = np.mean(virtual_slice.X, axis=0)
    true_mean_profile = np.mean(slice_1_heldout.X, axis=0)
    
    gene_pearson, _ = pearsonr(pred_mean_profile, true_mean_profile)
    gene_mse = np.mean((pred_mean_profile - true_mean_profile) ** 2)
    
    # B. Cell-level nearest neighbor comparison
    # Match virtual cells to the nearest spatial neighbor in actual Slice 1
    pred_coords = virtual_slice.obsm["spatial"]
    true_coords = slice_1_heldout.obsm["spatial"]
    
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(true_coords)
    distances, indices = nn.kneighbors(pred_coords)
    indices = indices.squeeze()
    
    # Compare expression profiles of matched cells
    pred_expr = virtual_slice.X
    true_expr = slice_1_heldout.X[indices]
    
    cell_mses = np.mean((pred_expr - true_expr) ** 2, axis=1)
    cell_pearsons = []
    for i in range(len(pred_expr)):
        p_val, _ = pearsonr(pred_expr[i], true_expr[i])
        if not np.isnan(p_val):
            cell_pearsons.append(p_val)
            
    mean_cell_mse = float(np.mean(cell_mses))
    mean_cell_pearson = float(np.mean(cell_pearsons)) if cell_pearsons else 0.0

    results = {
        "dataset_shape_train": [slice_0.shape, slice_2.shape],
        "dataset_shape_heldout": slice_1_heldout.shape,
        "virtual_slice_shape": virtual_slice.shape,
        "gene_profile_pearson": float(gene_pearson),
        "gene_profile_mse": float(gene_mse),
        "cell_level_mean_pearson": mean_cell_pearson,
        "cell_level_mean_mse": mean_cell_mse,
    }

    print("\nHold-Out Validation Summary:")
    print(f"  Gene Profile Pearson Correlation: {gene_pearson:.4f}")
    print(f"  Gene Profile Mean Squared Error:  {gene_mse:.4f}")
    print(f"  Cell-level Mean Pearson Corr:     {mean_cell_pearson:.4f}")
    print(f"  Cell-level Mean Squared Error:    {mean_cell_mse:.4f}")

    # Write metrics to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nHoldout validation metrics written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of matching training epochs")
    parser.add_argument("--output", default="./results/holdout_validation_metrics.json")
    args = parser.parse_args()
    main(args)
