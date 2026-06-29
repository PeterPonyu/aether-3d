# Aether3D claim ledger

All substantive claims require locally-reproduced benchmark artifacts in this repository.
Entries record only locally-reproduced evidence; no claim is inherited from the
DeepSpatial prior-art reference. A pull request that asserts a performance or biology claim
must update this ledger and reference its issue (`Closes #N`).

Status vocabulary: `supported` = evidence collected and consistent with the claim (real-data,
LOSO, multiple seeds); `planned` = gated, no qualifying evidence yet;
`falsified` = evidence contradicts the claim (kept as the scientific record, not deleted);
`validated` = requires explicit human sign-off (not yet reached for any row).

---

## Supported claims (v14, 2026-06-28)

| Claim | Evidence | Numbers | Status |
|---|---|---|---|
| Cross-gene low-rank beats poly1 on accuracy (MERFISH sparse-$z$ stacks) | Real LOSO on MERFISH-A (9 seeds × interior holdouts) and MERFISH-B (6 seeds × interior holdouts); no test-slice-touching; provenance JSON archived. | SVD-LR: 0.320/0.304; poly1: 0.277/0.263 Pearson (MERFISH-A/B) | **supported** |
| jointSLR > SVD-LR on accuracy (+0.05 Pearson), specificity-checked, significance-tested | Same LOSO; spatial specificity confirmed by planted-spatial control (+0.080 vs poly1) versus spatially-white control (−0.018 vs poly1); paired Wilcoxon across seeds × held-out slices: MERFISH-A +0.049 p=0.004, MERFISH-B +0.056 p=0.031; HNSCC ties poly1 p=1.0 (neutral, as reported). Evidence: `results/benchmark/reconstruction_3d/external/aether_v14jointslr_*.json`. | jointSLR: 0.368/0.361 vs SVD 0.320/0.304 (MERFISH-A/B) | **supported** |
| Accuracy↔fidelity tradeoff closed by CV-constrained blend (blend\_pareto\_joint) | blend\_pareto\_joint dominates SVD-blend at matched Moran-match on both MERFISH sets; CV auto-reverts to poly1 on HNSCC where low-rank signal is weak. | blend\_pareto\_joint: 0.316/.89 and 0.334/.85 (MERFISH-A/B); poly1 Moran-match: .92/.93 | **supported** |
| Intrinsic oracle-bounded ceiling; registration not a lever | Oracle r32 benchmarked (held-out slice's own best rank-32 approximation, not a method). Leak-free registration probe: MERFISH shift ≤1 bin, gain ≈0; HNSCC rigid alignment hurts (mean −0.15, z7 −0.40). | Oracle: 0.734/0.733 (MERFISH); 0.866 (HNSCC). Gap = biological cross-$z$ variation. | **supported** |
| jointSLR's sparse-$z$ advantage reproduces on a fully independent stack (3rd dataset; sign-level generalization across tissue AND modality) | LOSO on Kuett 2022 3D imaging mass cytometry of human breast cancer (15 serial sections, 25 protein markers — independent tissue and modality): jointSLR beats truncated SVD on every held-out (seed,slice) pair; paired Wilcoxon p<1e-4. Caveat: 25-marker panel is a more strongly low-rank regime (oracle rank-24 Pearson 0.997) than a transcriptomic stack, so the prior transfers in SIGN, not absolute accuracy regime. | +0.030 mean Pearson over SVD; 15/15 pairs positive; p<1e-4; oracle r24 = 0.997 | **supported** |
| Held-out Esr1 case study recovers the bilateral sexually-dimorphic POA estrogen-receptor domain | Entire $z{=}2$ MERFISH slice held out; reconstruction scored against the measured truth slice; recovers Esr1 pattern in the correct anatomical location. | held-out Esr1 reconstruction Pearson r=0.80 | **supported** |

---

## Planned claims (evidence not yet collected)

| Claim | Required evidence | Missing evidence | Status |
|---|---|---|---|
| Reconstructed volumes preserve cell-level expression fidelity across a broader panel of platforms (Visium serial ladders). | Per-gene held-out-slice expression recovery on additional transcriptomic platforms. | IMC (Kuett 2022) now done and supported (3rd-dataset row above); still missing: Visium serial-ladder LOSO runs. | planned |
| The method scales to large serial atlases (>30M cells). | Scaling table + peak-memory evidence before any large-atlas claim. | Large-atlas runtime/memory measurements. | planned |
| Aether3D interoperates with the AnnData/Scanpy/SpatialData ecosystem (graduation to validated). | Two real-data round-trips reproduced and lossless (ΔX=0, Δxyz=0). | META 2-real-round-trip bar met (ruling 2026-06-07); graduation to `validated` awaits explicit human sign-off. | planned (sign-off pending) |

---

## Falsification record (kept, not deleted)

| Claim | Contradicting evidence | Status |
|---|---|---|
| Continuous 3D vector-field reconstruction (flow-matching / INR, v12) recovers held-out interior slices better than poly1 / 2.5D baselines on spatial fidelity metrics (Moran's-I, Betti-0, sliced-Wasserstein). | Real leave-one-out on MERFISH-hypothalamus and Open-ST HNSCC (2026-06-06): continuous-field adapter **loses** to linear-interp and nearest-slice on Moran-I, Betti-0, and sliced-Wasserstein. Root cause: z-trajectory is nearly vacuous (mean≈poly1); the lever is the cross-gene estimator, not z-interpolation modelling. | **falsified** |
| Post-hoc spatial-factor smoothing (v12) improves over plain SVD low-rank. | Rejected during v13 development: not a principled objective; replaced by the single graph-Laplacian jointSLR objective. | **falsified** |

---

*No performance or biology claim graduates to `validated` or manuscript-final until the
corresponding row reaches `validated` with linked, reproducible artifacts and explicit
human sign-off.*
