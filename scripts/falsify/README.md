# Falsifiability controls

Synthetic positive/negative controls that prove the held-out contrast can
actually **detect** a continuous-method advantage — run before any real-data
spend.

- `aether_3d.benchmarks.synthetic_field` generates a stack sampling one *known*
  continuous field at every z (the same cells trace trajectories across slices):
  - **LINEAR** (negative control): true midpoint = linear blend of neighbours →
    a 2.5-D `linear-interp` baseline is near-exact; a learned model can't win.
  - **CURVED** (positive control): a quadratic bend makes linear interpolation
    provably biased at the midpoint → room a learned flow can exploit.

- `run_field_controls.py` runs both regimes through the audited volume-adapter
  contract for `AetherAdapter` + the 2.5-D baselines and reports, per regime,
  whether a trained Aether beats linear interpolation.

```bash
# Untrained smoke (fast):
scripts/run.sh scripts/falsify/run_field_controls.py --epochs 0

# Real adjudication (train per holdout); curved-regime LOSS = model defect:
scripts/run.sh scripts/falsify/run_field_controls.py --epochs 50 --out results/falsify_controls.json
```

Interpretation: a LINEAR-regime tie/loss for Aether is expected and correct; a
CURVED-regime loss (where 2.5-D is provably wrong) is a **model defect found
cheaply**, before real compute. This characterises the method — it graduates no
`CLAIM_LEDGER.md` row.

The validity of the controls themselves (negative ≈ exact, positive ≫ biased)
is locked in by `tests/benchmarks/test_synthetic_field.py`.

## Caveat: negative-control margin vs cell density

`linear-interp` pairs cells across slices by **2D nearest-neighbour**, not by
true identity, so at higher cell density NN-mispairing injects a small error
into the LINEAR ("near-exact") negative control — that residual is a baseline
pairing artifact, not field nonlinearity. The controls are validated at
`--n-cells 40` (the default; LINEAR coord_rmse ≈ 0.4–0.8). The positive/negative
separation stays large well beyond that, but the negative control's absolute
"≈0" softens as density rises — raise `--n-cells` knowing this.
