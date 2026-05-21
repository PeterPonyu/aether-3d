#!/usr/bin/env python3
"""
High-fidelity Synthetic Serial Slice Generator for Aether3D

Generates realistic 2D spatial slices that mimic what DeepSpatial used in its tutorials
(MERFISH-style hypothalamus, STARmap brain, IMC breast, etc.):

- Spatial coordinates with realistic cell-type domains
- Smooth transitions across physical Z
- Gene expression with cell-type specific programs
- Proper .obsm['spatial'] + .obs['z_coord'] + .obs['cell_class']

Intended for data_flow tests and e2e reconstruction verification when the user
does not yet have real serial spatial datasets locally.
"""

import numpy as np
import scanpy as sc
import pandas as pd


def generate_synthetic_serial_slices(
    n_slices: int = 5,
    cells_per_slice: int = 800,
    n_genes: int = 64,
    n_classes: int = 4,
    seed: int = 123,
    spatial_range: float = 60.0,
    slice_spacing: float = 10.0,
):
    """
    Returns (list_of_AnnData, class_names)

    Each AnnData has:
      - .obsm['spatial'] (XY)
      - .obs['z_coord'] (physical position)
      - .obs['cell_class']
      - realistic spatial domains per cell type (important for UOT to work well)
    """
    rng = np.random.default_rng(seed)
    class_names = [f"CT{i}" for i in range(n_classes)]
    adatas = []

    # Each cell type has a preferred spatial "center" that shifts slightly across slices
    type_centers = rng.uniform(10, spatial_range - 10, (n_classes, 2))

    for s in range(n_slices):
        # Physical Z for this slice
        z = s * slice_spacing

        # Generate cells with mild spatial clustering per type
        positions = []
        labels = []
        for c_idx, cname in enumerate(class_names):
            n_c = cells_per_slice // n_classes + rng.integers(-30, 30)
            center = type_centers[c_idx] + rng.normal(0, 3, 2) * (s / max(n_slices - 1, 1))
            pos = rng.normal(center, 8.0, (n_c, 2))
            pos = np.clip(pos, 0, spatial_range)
            positions.append(pos)
            labels += [cname] * n_c

        xy = np.vstack(positions).astype(np.float32)
        labels = np.array(labels)

        # Gene expression: each cell type upregulates a block of genes
        X = np.zeros((len(labels), n_genes), dtype=np.float32)
        for c_idx, cname in enumerate(class_names):
            mask = labels == cname
            n_c = mask.sum()
            base = rng.normal(0, 0.9, (n_c, n_genes))

            # Type-specific markers
            marker_start = c_idx * (n_genes // n_classes)
            marker_end = marker_start + n_genes // 6
            base[:, marker_start:marker_end] += rng.normal(2.8, 0.5, (n_c, marker_end - marker_start))

            lib = rng.uniform(500, 1800, n_c)
            counts = np.exp(base)
            counts = (counts.T * lib / counts.sum(1)).T
            X[mask] = rng.poisson(counts).astype(np.float32)

        ad = sc.AnnData(X=X)
        ad.obsm["spatial"] = xy
        ad.obs["cell_class"] = pd.Categorical(labels)
        ad.obs["z_coord"] = z
        ad.var_names = [f"G{i:03d}" for i in range(n_genes)]

        adatas.append(ad)

    return adatas, class_names


if __name__ == "__main__":
    slices, classes = generate_synthetic_serial_slices()
    print(f"Generated {len(slices)} high-fidelity synthetic serial slices (mimics DeepSpatial/MERFISH-style)")
    print(f"  Cells per slice ~{len(slices[0])}")
    print(f"  Classes: {classes}")
    print(f"  Z range: 0 to {(len(slices)-1)*10}")
    print("Ready for UOT + reconstruction E2E runs.")