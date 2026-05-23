# Aether3D Biology Figure Pack

Aether3D reconstructs continuous 3D tissue volumes from sparse serial 2D sections by training a multi-modal velocity field on spatial coordinates, gene expression, and cell-class identity. The reconstructed volume is a regular AnnData object whose features include:

- **3D coordinates** for every virtual cell (`obsm['spatial_3d']`, `obs['z_3d']`, `obs['virtual_depth']`) that can be queried at arbitrary depths between input slices;
- **predicted gene expression and cell-class probability** for each virtual cell, enabling marker analysis, cell-class stratification, and tissue-domain queries directly in 3D;
- a **continuous tissue volume** that supports virtual cross-sections at any Z, arbitrary orthogonal projections, and surface-mesh extraction.

This report exercises those outputs on two data sources: a synthetic 3-slice trajectory (fully reproducible from the sweep artifacts), and an on-disk MERFISH mouse hypothalamus serial-section dataset that Aether3D reconstructs into a single dense volume. No online downloads are required.

## Synthetic

### wide

- source: `results/benchmark/volumes/wide.h5ad`
- volume: 972 virtual cells, 32 genes, runtime: 1.0s, device: `cpu (precomputed)`

**3D point cloud — cell class**

![pointcloud_3d_class.png](./synthetic/wide/figures/pointcloud_3d_class.png)

[interactive HTML](./synthetic/wide/figures/pointcloud_3d_class.html)

**3D expression of top markers**

![pointcloud_3d_gene_11.png](./synthetic/wide/figures/pointcloud_3d_gene_11.png)

[interactive HTML — 11](./synthetic/wide/figures/pointcloud_3d_gene_11.html)

![pointcloud_3d_gene_4.png](./synthetic/wide/figures/pointcloud_3d_gene_4.png)

[interactive HTML — 4](./synthetic/wide/figures/pointcloud_3d_gene_4.html)

**Orthogonal projections (XY / XZ / YZ)**

![orthogonal_projections](./synthetic/wide/figures/orthogonal_projections.png)

**Virtual cross-sections at three Z values**

![virtual_slices](./synthetic/wide/figures/virtual_slices.png)

**Cell-class composition along reconstructed Z**

![z_class_composition](./synthetic/wide/figures/z_class_composition.png)

**Input 2D slices vs continuous Aether3D reconstruction**

![input_vs_reconstruction](./synthetic/wide/figures/input_vs_reconstruction.png)

**6-row Z-strata scatter grid** (reconstructed vs nearest input slice)

![multi_z_slice_grid](./synthetic/wide/figures/multi_z_slice_grid.png)

**Top markers along the reconstructed Z axis**

![gene_trajectory_along_z](./synthetic/wide/figures/gene_trajectory_along_z.png)

**Tissue surface mesh**

![tissue_mesh.png](./synthetic/wide/figures/tissue_mesh.png)

[interactive HTML mesh](./synthetic/wide/figures/tissue_mesh.html)

**Additional figures (auto-detected on disk)**

![pointcloud_3d_gene_2.png](./synthetic/wide/figures/pointcloud_3d_gene_2.png)
![pointcloud_3d_gene_27.png](./synthetic/wide/figures/pointcloud_3d_gene_27.png)

## Real

### merfish_hypothalamus

- source: `data/baselines/deepspatial/merfish_mouse_hypothalamus/merfish_0.h5ad`
- volume: 19,200 virtual cells, 155 genes, runtime: 18.8s, device: `cuda`

**3D point cloud — cell class**

![pointcloud_3d_class.png](./real/merfish_hypothalamus/figures/pointcloud_3d_class.png)

[interactive HTML](./real/merfish_hypothalamus/figures/pointcloud_3d_class.html)

**3D expression of top markers**

![pointcloud_3d_gene_Gad1.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Gad1.png)

[interactive HTML — Gad1](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Gad1.html)

![pointcloud_3d_gene_Myh11.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Myh11.png)

[interactive HTML — Myh11](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Myh11.html)

**Orthogonal projections (XY / XZ / YZ)**

![orthogonal_projections](./real/merfish_hypothalamus/figures/orthogonal_projections.png)

**Virtual cross-sections at three Z values**

![virtual_slices](./real/merfish_hypothalamus/figures/virtual_slices.png)

**Cell-class composition along reconstructed Z**

![z_class_composition](./real/merfish_hypothalamus/figures/z_class_composition.png)

**Input 2D slices vs continuous Aether3D reconstruction**

![input_vs_reconstruction](./real/merfish_hypothalamus/figures/input_vs_reconstruction.png)

**Per-cell-class 3D density similarity** (reconstructed vs input KDE cosine + cell counts)

![density_similarity](./real/merfish_hypothalamus/figures/density_similarity_bars.png)

**Per-gene Moran's I** scatter, reconstructed vs input stack

![morans_i_scatter](./real/merfish_hypothalamus/figures/morans_i_scatter.png)

**6-row Z-strata scatter grid** (reconstructed vs nearest input slice)

![multi_z_slice_grid](./real/merfish_hypothalamus/figures/multi_z_slice_grid.png)

**Top markers along the reconstructed Z axis**

![gene_trajectory_along_z](./real/merfish_hypothalamus/figures/gene_trajectory_along_z.png)

**Tissue surface mesh**

![tissue_mesh.png](./real/merfish_hypothalamus/figures/tissue_mesh.png)

[interactive HTML mesh](./real/merfish_hypothalamus/figures/tissue_mesh.html)

**Additional figures (auto-detected on disk)**

![pointcloud_3d_gene_Irs4.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Irs4.png)
![pointcloud_3d_gene_Mlc1.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Mlc1.png)
![pointcloud_3d_gene_Pak3.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Pak3.png)
![pointcloud_3d_gene_Selplg.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Selplg.png)
![pointcloud_3d_gene_Syt2.png](./real/merfish_hypothalamus/figures/pointcloud_3d_gene_Syt2.png)

---

Reproduce with (dl env required for the RTX 5090):

```bash
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode all
```
