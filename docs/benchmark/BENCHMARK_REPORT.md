# Aether3D Synthetic Benchmark Report

- Generated: `20260523-060845`
- Device: `cuda`
- Synthetic data: 3 slices × 400 cells × 32 genes, 4 cell classes, slice spacing 10.0, seed 42

Slice 0 and Slice 2 are used for training; Slice 1 (Z=10) is held out and reconstructed via virtual-depth interpolation. Metrics compare reconstructed vs held-out.

## Quality metrics per refined version

| Config | hidden | depth | heads | epochs | Gene Pearson | Cell Pearson | Gene MSE | Cell MSE |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 32 | 2 | 2 | 4 | 0.8957 | 0.1930 | 84.2312 | 11023.0830 |
| `small` | 64 | 2 | 2 | 4 | 0.8783 | 0.1836 | 103.0945 | 10334.5566 |
| `wide` | 128 | 2 | 4 | 4 | 0.8773 | 0.1621 | 106.1991 | 10946.5137 |
| `deep` | 64 | 4 | 4 | 4 | 0.8903 | 0.1835 | 90.3428 | 10640.8389 |

![metric_gene_profile_pearson](./figures/metric_gene_profile_pearson.png)
![metric_cell_level_mean_pearson](./figures/metric_cell_level_mean_pearson.png)
![metric_gene_profile_mse](./figures/metric_gene_profile_mse.png)
![metric_cell_level_mean_mse](./figures/metric_cell_level_mean_mse.png)

## Resource usage per refined version

| Config | Params | Wall total (s) | Flow train (s) | Reconstruct (s) | Peak GPU (MB) |
|---|---|---|---|---|---|
| `tiny` | 62,830 | 3.27 | 2.72 | 0.55 | 22.0 |
| `small` | 213,710 | 3.93 | 2.38 | 1.55 | 26.0 |
| `wide` | 779,662 | 4.56 | 2.37 | 2.20 | 40.2 |
| `deep` | 363,086 | 5.08 | 2.67 | 2.41 | 33.8 |

![runtime breakdown](./figures/runtime.png)

![peak GPU memory](./figures/peak_gpu_mem.png)

## Training loss curves

![loss curves](./figures/loss_curves.png)

## Reconstructed volume views

![volume_xy_tiny](./figures/volume_xy_tiny.png)
![volume_xy_small](./figures/volume_xy_small.png)
![volume_xy_wide](./figures/volume_xy_wide.png)
![volume_xy_deep](./figures/volume_xy_deep.png)
![volume_xz_tiny](./figures/volume_xz_tiny.png)
![volume_xz_small](./figures/volume_xz_small.png)
![volume_xz_wide](./figures/volume_xz_wide.png)
![volume_xz_deep](./figures/volume_xz_deep.png)

---

Re-run with:

```bash
conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py
# TODO(ref-parity): make_plots.py removed; regenerate plots after real results
```
