# Aether3D baselines

This directory holds intentionally-minimal baseline implementations used only
for in-repo comparison against Aether3D's reconstructed 3D volume. Nothing
here is a faithful reproduction of any published method.

Currently:

- `naive_25d_baseline.py` — naive identity-preserving 2.5D stacking baseline:
  for each interpolated Z position, virtual cells are assigned by
  nearest-neighbor lookup into the adjacent input slice cells. No gene-wise
  smoothing; each virtual cell inherits a real input cell's expression vector
  + an interpolated XY coordinate. ~150 LOC. Framed as a "naive
  identity-preserving lower bound" — not a published-method reproduction.
