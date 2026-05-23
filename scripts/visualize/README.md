# Aether3D Biology Figure Pack

What it does: produces the biology-focused outputs Aether3D is actually meant
to deliver — continuous 3D reconstruction from serial 2D slices, interactive
point clouds + tissue mesh, orthogonal projections, virtual cross-sections,
cell-class stratification along Z, and marker trajectories. No online downloads.

## Run

```bash
# both synthetic + real (real uses local MERFISH baseline)
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode all

# synthetic only (uses the existing benchmark sweep output)
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode synthetic

# real only
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode real
```

Synthetic mode requires `scripts/benchmark/run_synthetic_sweep.py` to have
produced `results/benchmark/volumes/<config>.h5ad`.
Real mode trains an `AetherFlowModule` for 3 epochs on the on-disk MERFISH
slices and then runs `AetherReconstructor.reconstruct_continuous_volume`.

## Figures emitted

Under `docs/biology/<mode>/<dataset>/figures/`:

- `pointcloud_3d_class.{html,png}` — interactive 3D scatter coloured by cell class
- `pointcloud_3d_gene_<gene>.{html,png}` — same point cloud coloured by gene expression
- `orthogonal_projections.png` — XY / XZ / YZ scatter triptych
- `virtual_slices.png` — 2D cross-sections at three reconstructed Z values
- `z_class_composition.png` — stacked-area cell-class proportion along Z
- `input_vs_reconstruction.png` — raw 2D inputs vs reconstructed XZ side view
- `gene_trajectory_along_z.png` — mean expression of top markers along Z
- `tissue_mesh.{html,png}` — Delaunay / convex-hull surface mesh

`docs/biology/BIOLOGY_REPORT.md` collects everything with embedded thumbnails
and links to the interactive HTML files.
