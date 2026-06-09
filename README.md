# Aether3D

**Planned continuous 3D tissue reconstruction; current benchmark evidence is `planned (contradicted)` for the continuous-over-2.5D advantage claim.**

Aether3D is a local research package for continuous 3D spatial-omics reconstruction experiments from discrete serial 2D physical slices. It targets joint velocity-field modeling over space, gene expression, and cell identity via optimal transport coupling + flow matching. Current evidence supports synthetic/local-small smoke validation and real-data interoperability checks, but the headline continuous-3D reconstruction advantage claim is `planned (contradicted)`: on the repo's own real leave-one-out benchmarks, the continuous adapter loses to simple 2.5D/linear-interpolation baselines on the primary geometry/topology metrics. Cell-level fidelity, broad baseline advantage, and large-atlas scalability remain gated by [`CLAIM_LEDGER.md`](./CLAIM_LEDGER.md).

Given a schema-valid stack of aligned 2D spatial transcriptomics / proteomics slices with physical Z coordinates, the target API produces a 3D AnnData-style volume for benchmarked virtual slicing and downstream evaluation. Downstream biological use requires the claim-ledger evidence gates to pass.

## Why "Aether3D"?

"Aether" (upper pure air / continuous medium) captures the target modeling idea of a smooth tissue manifold inferred from sparse physical sampling. The current package treats that as a testable hypothesis, not as an already proven biological-fidelity claim.

## Installation

This research package is not published on PyPI yet. Install from a clone or a GitHub branch/commit:

```bash
git clone https://github.com/PeterPonyu/aether-3d
cd aether-3d
pip install -e ".[dev,viz]"
# or: pip install "git+https://github.com/PeterPonyu/aether-3d.git"
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
    alpha_spatial=0.5,
    lambda_g=0.1,
    lambda_c=10.0,
    spatial_key="spatial",
    z_key="z_coord",
    label_key="cell_class",
    seed=42,
)

model = AetherReconstructor(cfg)
model.setup_data(adatas)
model.fit(max_epochs=100)

volume = model.reconstruct_continuous_volume(adatas, thickness=10.0, n_samples=200000)
# volume is a single AnnData with 3D coordinates, imputed genes, continuous cell-type probabilities
```

## Architecture Highlights

- **Unbalanced Optimal Transport** coupling between adjacent slices (spatial + transcriptomic + cell-type cost).
- **Multi-modal Flow Matching** (`MultiModalVelocityField`): joint velocity prediction on (x, g, c, z).
- **Probability Flow ODE** integration with density-preserving bidirectional sampling and adaptive pruning.
- Target AnnData / Scanpy / SpatialData interoperability, gated by export-contract tests.
- Target large-atlas scalability via chunking, gated by scaling-table and memory evidence before any >30M-cell-style claim.

This architecture defines the local Aether3D implementation surface. Publication claims are controlled by the claim ledger; the current reconstruction-advantage headline must stay `planned (contradicted)` until repaired benchmarks beat the 2.5D baselines.

## Prior Art and Audit Boundary

Aether3D is an independent package and manuscript track for continuous 3D spatial-omics reconstruction from serial sections. The public DeepSpatial preprint and repository are treated as prior art for the general research problem and for audit comparison only; Aether3D's user-facing API, documentation, validation plan, figures, and manuscript claims must be written from local evidence.

See [BASELINE_COMPARISON.md](./BASELINE_COMPARISON.md) for the audit boundary and leakage checks. The frozen DeepSpatial audit tree referenced there is kept in the internal monorepo and is not shipped with this standalone clone. Claims graduate to the manuscript only through the project claim ledger and reproducible benchmark artifacts, not by inheriting claims from the reference work.

**Prior-art citation** (to appear in final paper):
> Yang, Y. et al. "Reconstructing True 3D Spatial Omics at Single-Cell Resolution." bioRxiv (2026).

## Status

- **Phase 0** — Skeletons + audit protection: **complete**
- **Phase 1** — Clean `flow/` primitives (independent implementation): **complete**
- **Phase 3** — UOT coupling, multi-modal velocity field, and the 3D reconstructor: **implemented** (synthetic / local-small smoke validated; real-data claims gated by the claim ledger).

Goal: a high-quality, pip-installable research package and manuscript track, with paper claims promoted only once claim-ledger evidence supports them.

## License

MIT.

---

*Part of the 2026 spatial omics re-implementation program.*
