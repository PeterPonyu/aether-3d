# Aether3D Scripts Organization

Clear separation of concerns for the three types of scripts/tests you asked for.

```
scripts/
├── README.md
├── ci/            # Code-level CI tests (fast, lightweight)
├── data_flow/     # Data pipeline & integration tests
└── e2e/           # Real full-run E2E experiments (the important ones for research)
```

---

## 1. `ci/` — Code-level CI Workflow Tests

Fast unit and small integration tests suitable for automated CI (GitHub Actions, pre-commit).

- < 60–90 seconds on CPU
- No real data, minimal model training
- Located primarily in `tests/` (pytest discovery)

Run:
```bash
python -m pytest tests/ -q
```

---

## 2. `data_flow/` — Data Flowing / Integration Tests

Tests that exercise data movement:
- Serial slice loading
- UOT coupling (with fallback)
- Trajectory dataset construction
- Multi-modal state preparation
- Synthetic serial slice generators

These help debug data bugs before spending GPU hours on E2E.

---

## 3. `e2e/` — Real Full-Run End-to-End Tests (Your Research Scripts)

These are the heavy, meaningful scripts that produce actual 3D volumes or enhanced data.

Current important script:

- `verify_aether_pipeline.py` — Deep verification that exercises the full stack (UOT dataset → multi-modal velocity field → `AetherFlowModule` → `AetherReconstructor.reconstruct_continuous_volume()`)

When you have real serial slice datasets (MERFISH, STARmap, IMC, Xenium serial sections, etc.), you will create scripts in this folder such as:

- `reconstruct_real_serial_slices.py`
- `run_aether_on_brain_atlas.py`

**Recommended invocation (using your correct DL environment):**

```bash
conda run -n dl python scripts/e2e/verify_aether_pipeline.py
```

For real data you will later add:

```bash
conda run -n dl python scripts/e2e/reconstruct_real_serial_slices.py \
    --slices /path/to/slice_*.h5ad \
    --output ./results/brain_3d_volume.h5ad
```

---

## Decision Guide

| Goal                                          | Folder to look in     |
|-----------------------------------------------|-----------------------|
| Fast CI / "does the code still work?"         | `tests/` + `scripts/ci/` |
| Debugging data loading / UOT / state prep     | `scripts/data_flow/`  |
| Actually generating 3D volumes for papers on real or large data | `scripts/e2e/` |

---

Keep `e2e/` sacred for scripts that output real, inspectable `.h5ad` files with 3D coordinates, imputed expression, and printed quality metrics.

**Last updated**: 2026-05-21 (reorganization for clear CI vs Data-Flow vs Full Research E2E distinction)
