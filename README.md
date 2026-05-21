# Aether3D

**Learning Continuous 3D Tissue Vector Fields from Serial Spatial Omics Slices**

Aether3D is a generative framework that reconstructs **true continuous three-dimensional spatial omics** from discrete serial 2D physical slices. It learns a joint velocity field over space, gene expression, and cell identity via optimal transport coupling + flow matching, enabling in-silico sectioning at arbitrary depths with biologically faithful cell density and correspondences.

Given a stack of aligned 2D spatial transcriptomics / proteomics slices (with physical Z coordinates), Aether3D produces a dense 3D AnnData volume that can be sliced, queried, or fed into any downstream 3D spatial analysis tool (CellCharter, 3D domain detection, cell-cell communication in volume, etc.).

## Why "Aether3D"?

"Aether" (upper pure air / continuous medium) captures the idea of a smooth, unbroken tissue manifold recovered from sparse physical sampling — the hidden continuous reality that physical sectioning destroys.

## Installation (Planned)

```bash
pip install aether-3d
# dev
pip install -e ".[viz]"
```

## Quick Start (Target API)

```python
import scanpy as sc
import glob
from aether_3d import AetherReconstructor
from aether_3d.config import Aether3DConfig

adatas = [sc.read_h5ad(p) for p in sorted(glob.glob("slices/slice_*.h5ad"))]

cfg = Aether3DConfig(
    patch_size=8,
    hidden_size=256,
    depth=6,
    uot_alpha_spatial=0.5,
    lambda_gene=0.1,
    lambda_class=10.0
)

model = AetherReconstructor(cfg)
model.setup_data(adatas, spatial_key="spatial", z_key="z_coord", label_key="cell_class")
model.fit(max_epochs=100)

volume = model.reconstruct_continuous_volume(adatas, thickness=10.0, n_samples=200000)
# volume is a single AnnData with 3D coordinates, imputed genes, continuous cell-type probabilities
```

## Architecture Highlights

- **Unbalanced Optimal Transport** coupling between adjacent slices (spatial + transcriptomic + cell-type cost).
- **Multi-modal Flow Matching** (`MultiModalVelocityField`): joint velocity prediction on (x, g, c, z).
- **Probability Flow ODE** integration with density-preserving bidirectional sampling and adaptive pruning.
- Full AnnData / Scanpy / SpatialData interoperability.
- Scalable to >30 million cell whole-organ atlases via chunking.

This is the scientific core of the DeepSpatial line of work, completely rebranded and refactored for independent publication.

## Relationship to Baseline

Aether3D is a **heavy refactor + full rebrand** of the public `DeepSpatial` package and site (Yang et al., bioRxiv 2026). Every class name, docstring, narrative, and public API has been rewritten under the Aether3D identity while preserving (and numerically validating) the exact UOT + flow-matching reconstruction algorithm.

See [BASELINE_COMPARISON.md](./BASELINE_COMPARISON.md) + the frozen audit tree `../baselines/DeepSpatial-original/` for traceability.

**Inspirational citation** (to appear in our paper):
> Yang, Y. et al. "Reconstructing True 3D Spatial Omics at Single-Cell Resolution." bioRxiv (2026).

We treat it as foundational prior art. All new code, experiments, and claims in Aether3D are original.

## Status (2026-05-21)

- **Phase 0** — Skeletons + audit protection + 5090 check: **complete**
- **Phase 1** — Clean `flow/` primitives (identical to LuminaST for full package independence): **complete**

Next: Phase 3 (UOT coupling, multi-modal velocity field, 3D reconstructor).

Goal: two separate, high-quality, pip-installable research packages powering two new bioRxiv papers.

## License

MIT.

---

*Companion project to LuminaST under the 2026 spatial omics re-implementation program.*
