# Aether3D — references (with code) & datasets

Consolidated reference + dataset index. Paper DOIs verified via Crossref and code
repositories via the GitHub API on 2026-06-09. See `manuscript/refs.bib`,
`docs/DATASETS.md`, and the benchmark adapters under `src/aether_3d/benchmarks/adapters/`.

## Reference papers & method baselines (with public code)

| Role | Method | Venue / year | DOI | Code |
|------|--------|--------------|-----|------|
| Prior-art | Yang et al. (DeepSpatial) — Reconstructing True 3D Spatial Omics at Single-Cell Resolution | bioRxiv 2026 | `10.64898/2026.04.28.721395` | audit boundary (not vendored) |
| Baseline ⚙ | 3d-OT — deep geometry-aware heterogeneous ST-slice alignment | Nature Methods 2026 | `10.1038/s41592-026-03034-9` | https://github.com/dbjzs/3d-OT |
| Baseline ⚙ | ASIGN — 3D ST alignment | arXiv 2412.03026 | — | https://github.com/hrlblab/ASIGN |
| Baseline ⚙ | SpatialZ | — | — | https://github.com/senlin-lin/SpatialZ |

Method citations in `manuscript/refs.bib` (alignment / 3D): CellCharter
`10.1038/s41588-023-01588-4` · PASTE `10.1038/s41592-022-01459-6` · PASTE2
`10.1101/gr.277670.123` · STAligner `10.1038/s43588-023-00543-x` · STitch3D
`10.1038/s42256-023-00734-1` · GPSA `10.1038/s41592-023-01972-2` · STalign
`10.1038/s41467-023-43915-7`.

## Datasets (audited registry — `docs/DATASETS.md`)

- Allen/Zhuang ABC Atlas MERFISH whole mouse brain (~147 serial sections, AWS S3)
- MOSTA Stereo-seq mouse embryo (CNGB CNP0001543); Stereo-seq whole mouse brain (CNGB CNP0003837)
- 10x Visium mouse-brain serial sections (4); Visium HD mouse brain / tonsil; liver regeneration GEO **GSE223561**
- STARmap PLUS mouse CNS 3D (Zenodo 8092024); Kuett 2022 3D-IMC (Zenodo 4752030); MERFISH hypothalamus

> Verification: Yang2026 + 3d-OT DOIs confirmed in Crossref; 3d-OT / ASIGN / SpatialZ repos
> live via GitHub API. GEO GSE223561 confirmed accessible (2026-06-09).
