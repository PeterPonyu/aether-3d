# Aether3D Synthetic Benchmark Harness

Local, no-download sweep that trains the Aether3D flow on 2 synthetic slices,
reconstructs the held-out middle slice, and records quality + resource usage
for every refined model variant.

## Run

```bash
conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py
# TODO(ref-parity): make_plots.py removed; regenerate plots after real results
```

Raw JSON + reconstructed `.h5ad`s land in `results/benchmark/` (git-ignored).
Tables, figures, and the report land in `docs/benchmark/` (git-tracked).

## Configs swept

| Name   | hidden | depth | heads | epochs |
|--------|--------|-------|-------|--------|
| tiny   | 32     | 2     | 2     | 4      |
| small  | 64     | 2     | 2     | 4      |
| wide   | 128    | 2     | 4     | 4      |
| deep   | 64     | 4     | 4     | 4      |

All configs share the same synthetic 3-slice setup (slice 0/2 for training,
slice 1 held out for evaluation, slice spacing 10.0, seed 42).

## What is captured per config

- `gene_profile_pearson`, `cell_level_mean_pearson` (higher is better)
- `gene_profile_mse`, `cell_level_mean_mse` (lower is better)
- Wall seconds for flow training and reconstruction
- Peak GPU MB (when CUDA active)
- Parameter count
- Flow training loss curve per epoch
- Full reconstructed volume `.h5ad` (`obsm['spatial_3d']`, `obs['z_3d']`, `obs['virtual_depth']`)

## Outputs at a glance

```
results/benchmark/                              (git-ignored)
  aether_sweep_<TS>.json
  aether_sweep_latest.json
  curves/<config>.json
  volumes/<config>.h5ad

docs/benchmark/                                 (git-tracked)
  BENCHMARK_REPORT.md
  summary.csv
  figures/metric_*.png
  figures/loss_curves.png
  figures/runtime.png
  figures/peak_gpu_mem.png
  figures/volume_xy_<config>.png
  figures/volume_xz_<config>.png
```
