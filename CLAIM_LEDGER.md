# Aether3D claim ledger

All substantive claims are **planned** until local code, tests, and validated
datasets produce reproducible benchmark artifacts in this repository. Entries
record only locally-reproduced evidence; no claim is inherited from the
DeepSpatial prior-art reference. A pull request that asserts a performance or
biology claim must update this ledger and reference its issue (`Closes #N`).

| Claim | Required evidence | Current evidence | Missing evidence | Status |
|---|---|---|---|---|
| Continuous 3D vector-field reconstruction recovers held-out interior slices better than 2.5D stacking. | Held-out-slice protocol (LOSO, mean±std) scoring Aether through the same audited adapter contract as the baselines, on validated serial data. | Real leave-one-out on TWO serial datasets through the audited contract — MERFISH-hypothalamus (3 interior holdouts) and openST/HNSCC GSE251926 (17 interior holdouts, 2026-06-06). On BOTH, continuous **loses** to the linear-interp (np.interp) / nearest-slice / stacking-2.5D baselines on Moran's-I, Betti-0 and sliced-Wasserstein (openST: wins only chamfer 400.8 vs 422.9; betti0 0.09 vs 0.34). | **Model repair** so continuous beats the 2.5D baselines (current evidence contradicts the claim); a physical-µm-spaced volume for ground-truth (not self-supervised) scoring (#291). | planned (contradicted) |
| Reconstructed volumes preserve cell-level expression fidelity. | Per-gene held-out-slice expression recovery (PCC/RMSE) against truth slices. | Geometry + Moran's-I agreement metrics only. | Per-gene expression-recovery metric run on validated data. | planned |
| Aether3D interoperates with the AnnData / Scanpy / SpatialData ecosystem. | Export-contract tests over round-tripped volumes. | Volume-contract tests on synthetic volumes. | Real-data export round-trip evidence. | planned |
| The method scales to large serial atlases. | Scaling table + peak-memory evidence before any >30M-cell-style claim. | Bounded synthetic scaling smoke only. | Large-atlas runtime/memory measurements. | planned |

_No performance or biology claim graduates to the manuscript until the
corresponding row reaches `validated` with linked, reproducible artifacts._
