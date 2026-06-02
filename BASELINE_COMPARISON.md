# Aether3D vs DeepSpatial Baseline — Refactor Mapping & Leakage Audit

**Project**: Aether3D (Learning Continuous 3D Tissue Vector Fields from Serial Spatial Omics Slices)
**Baseline**: the frozen `DeepSpatial-original/` audit tree (kept in the internal monorepo; not shipped with this standalone clone)
**Date of mapping**: 2026-05-21
**Status**: Initial skeleton (heavy refactor + rename in progress)

## Purpose
Living audit proving that Aether3D delivers the identical 3D reconstruction science (UOT slice coupling + multi-modal flow-matching velocity fields + ODE integration + density-preserving pruning) under a completely new brand and implementation, with zero textual leakage.

## High-Level Rename Table

| Original Concept / File                     | Aether3D Equivalent (new)                              | Rationale / Improvement |
|---------------------------------------------|--------------------------------------------------------|---------------------------|
| `DeepSpatial` (facade class)                | `AetherReconstructor`                                  | New brand, "reconstructor" emphasizes continuous volume |
| `DeepSpatialModule`                         | `AetherFlowModule`                                     | Consistent "Flow" naming |
| `DeepSpatialDataset`                        | `SerialSliceTrajectoryDataset`                         | Describes exactly what it does |
| `GiT` (multi-stream for x/g/c)              | `MultiModalVelocityField` (or `SpatialGeneClassNet`)   | No "GiT", explicit multi-modal |
| `compute_uot_coupling` + cost matrix        | `coupling.py` → `UnbalancedOTCoupler`                  | Clean API, optional pure-torch sinkhorn fallback |
| `reconstruct_full_volume` / chunked ODE     | `reconstruct_continuous_volume()` + `generate_between_slices()` | More descriptive |
| `vis_utils.py` (30k LOC monolithic)         | `viz/` package (plotly + optional napari/SpatialData)  | Modular, `[viz]` extra, no bloat in core |
| `n_samples_base`, magic UOT alphas          | All in `Aether3DConfig` (Pydantic)                     | Reproducible, documented |
| `transport/` + `models/commons.py` + `git.py` | `flow/` + `models/{...}` (clean re-type, deduped)     | Clean, deduplicated implementation |
| `core.py` (huge facade)                     | `core/reconstructor.py` + thin public API              | Testable pieces, better separation |
| Hard-coded label encoder etc.               | Config-driven `CellTypeRegistry`                       | Supports arbitrary taxonomies |

## Leakage Prevention

Leakage scanner:
```bash
rg --type py --type md --glob '!BASELINE_COMPARISON.md' \
   'DeepSpatial|stPainter|yyh030806|Reconstructing True 3D Spatial Omics|10.64898/2026.04.28.721395' \
   src/ docs/ ...
```
Zero hits required outside this file and the citation paragraph in README.

## 3D-Specific Fidelity Points (Phase 5 verification)

- [ ] UOT cost (spatial + cosine gene + 10× class penalty) produces equivalent couplings
- [ ] Flow-matching velocity field on joint (x, g, c, z) state matches original trajectories
- [ ] Bidirectional ODE + inverse-CDF time sampling + top-k gene pruning yields same density-preserving volumes
- [ ] Chunked inference (2048 cells) memory profile identical or better on 24 GB 5090
- [ ] Downstream Scanpy 3D domain detection (CellCharter etc.) gives comparable biological results

## Large Assets Not Ported
- `assets/*.html` (WebGL explorers) — will be regenerated with new branding + possibly napari widgets
- `docs/source/` Sphinx site — new MkDocs or Sphinx with fresh tutorials

## Shared flow primitives
The package uses cleaned `flow/` primitives (velocity prediction, Linear/GVP/VP paths, ODE/SDE samplers, EMA), implemented independently. This is an explicit improvement over the original baseline that copied ~800 LOC of transport code.

**Audit sign-off schedule**: per the project audit milestones.

Together with the frozen `DeepSpatial-original/` audit tree (internal monorepo), this document provides complete traceability for the Aether3D publication.
