# Aether3D Dataset Integration Guide

Aether3D learns **continuous 3D tissue vector fields from serial 2D spatial-omics
slices with physical Z coordinates**. Optimal-transport coupling links adjacent
physical sections, and multi-modal flow matching integrates a joint velocity
field over space, gene expression, and cell identity to reconstruct a continuous
3D volume from sparse physical sampling. The model therefore only works on data
that ships **multiple aligned 2D sections along a real Z axis** — a single
section cannot define the cross-slice trajectories the model trains on
(`SerialSliceTrajectoryDataset` requires `>= 2` slices).

This guide selects the audited, verified datasets from the canonical
`ST_research/datasets/DATASET_REGISTRY.md` (2026-05-28 audit) that best satisfy
Aether3D's serial / stacked-slice requirement, and maps each onto the existing
`aether_3d_serial_slice` AnnData contract. It complements
[`docs/DATA.md`](./DATA.md), which documents the Python-native plug-and-play
serial picks (MERFISH hypothalamus, Visium breast cancer serial, MOSTA); the
datasets below are the **larger audited registry recommendations** intended for
scaled 3D-reconstruction experiments.

> **Source of truth:** all figures, accessions, and verification marks below are
> copied from `ST_research/datasets/DATASET_REGISTRY.md`. Where that registry
> corrects earlier docs, the corrected values are used here (notably
> **GSE223561 = human liver _regeneration_, ~9.1 GB — NOT liver cancer**).

## Schema contract (recap)

Every dataset is mapped onto the `aether_3d_serial_slice` AnnData schema:

| Schema field | Meaning | Config key |
|---|---|---|
| `obsm['spatial']` | 2D in-plane coordinates of the section | `spatial_key` |
| `obs['z_coord']` | physical depth / order of the section along Z | `z_key` |
| `obs['cell_class']` | cell-type / region label | `label_key` |
| `X` | gene/protein expression (raw counts preferred for DL training) | — |

A serial **stack** is a Python list of these AnnData sections, ordered by
`z_coord`, passed to `AetherReconstructor.setup_data(adatas)`. Per project data
policy: raw integer count matrices + spatial metadata only — **no WSIs, FASTQs,
BAMs, or normalized-only objects**.

---

## Recommended datasets (audited registry)

| # | Dataset | Accession / ID | Platform | Tissue | Sections / slices | Raw-count size | Source | Serial / Z fit |
|---|---------|----------------|----------|--------|-------------------|----------------|--------|----------------|
| 14 | MERFISH Mouse Brain Receptor Map | Vizgen — info.vizgen.com/mouse-brain-data | MERFISH (imaging, single-cell) | mouse brain | **9** (3 coronal × 3 replicates), 483 genes, 734,696 cells | 3–7 GB | ⚠️ **UNVERIFIED** hotlink (see caveat) | native 9-slice stack — the strongest serial source here |
| 4 | Visium HD Mouse Brain | 10x dataset page | Visium HD (2 µm bins) | mouse brain (H&E) | 1 capture (~20k genes, ~30k @ 8 µm) | 4–7 GB | ✅ 10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he | ultra-high-res per-section block; pair ≥2 captures for a stack |
| 5 | Visium HD Tonsil | 10x dataset page | Visium HD (2 µm bins) | tonsil (fresh frozen) | 1 capture (~20k genes, ~30k @ 8 µm) | 2–4 GB | ✅ 10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen | ultra-high-res per-section block; pair ≥2 captures for a stack |
| 17 | GSE223561 — human liver **regeneration** | GEO **GSE223561** | Visium v1 | **human liver regeneration — NOT cancer** | up to 66 (human+mouse SuperSeries), ~18k genes, ~3k spots/section | **~9.1 GB** `RAW.tar` | ✅ ncbi.nlm.nih.gov/geo — "Multimodal decoding of human liver regeneration" | multiple serial Visium sections — assign `z_coord` by section order |

### 14 — MERFISH Mouse Brain Receptor Map (★ serial/3D pick)

- **Access / source:** Vizgen showcase page `info.vizgen.com/mouse-brain-data`.
- **Platform:** MERFISH, imaging-based, single-cell resolution.
- **Scale (registry):** 483 genes × 734,696 cells across **9 sections**
  (3 coronal positions × 3 biological replicates).
- **Raw counts:** `cell_by_gene` count matrix + per-cell metadata (3–7 GB).
- **Serial / Z fit:** this is a **native multi-section stack** — the single best
  fit for Aether3D in the audited set. Use the 3 coronal positions as the ordered
  physical Z axis (replicates give per-Z redundancy for virtual-slice holdout).
- **Schema mapping:** `spatial ← cell metadata `center_x`/`center_y`;
  `z_coord ← coronal-position index × physical section spacing`;
  `cell_class ← provided cell-type annotation` (or cluster if absent);
  `X ← cell_by_gene` counts. Build one AnnData per coronal position, ordered by
  `z_coord`, into the serial stack.
- **⚠️ Caveat:** the direct `hubfs` download hotlink is **UNVERIFIED** — see
  the caveat section below; obtain the real bundle from the canonical page.

### 4 — Visium HD Mouse Brain

- **Access / source:** ✅ 10x dataset page
  `10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he`.
- **Platform:** Visium HD, 2 µm continuous bins (typically analyzed at 8 µm).
- **Scale (registry):** 1 capture, ~20k genes, ~30k bins @ 8 µm, 4–7 GB.
- **Raw counts:** download the `binned_outputs.tar.gz` bundle (raw bin counts);
  **avoid the unverified single-`.h5` hotlink** — use the page bundle.
- **Serial / Z fit:** a single HD capture is **one dense section**, not a stack.
  It serves as an ultra-high-resolution per-section building block: pair ≥2
  adjacent HD captures (assigning increasing `z_coord`) to form a true serial
  stack, or use within-section virtual-slice holdout as a high-resolution
  single-section stress test.
- **Schema mapping:** `spatial ← obsm['spatial']` (bin centers);
  `z_coord ← 0.0` for the first capture, fixed physical spacing for each
  additional capture; `cell_class ← leiden/cluster` (no native label);
  `X ← raw bin counts`.

### 5 — Visium HD Tonsil

- **Access / source:** ✅ 10x dataset page
  `10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen`.
- **Platform:** Visium HD, 2 µm continuous bins (8 µm analysis).
- **Scale (registry):** 1 capture, ~20k genes, ~30k bins @ 8 µm, 2–4 GB.
- **Raw counts:** `binned_outputs.tar.gz` from the dataset page (raw bin counts).
- **Serial / Z fit:** same single-capture caveat as #4 — one dense section.
  Smallest HD download (2–4 GB), so it is the cheapest HD smoke target; pair ≥2
  captures for a stack or use intra-section virtual-slice holdout.
- **Schema mapping:** identical to #4 (`spatial ← bin centers`;
  `z_coord ←` per-capture physical spacing; `cell_class ←` clustering;
  `X ←` raw bin counts).

### 17 — GSE223561 — human liver regeneration serial sections

- **Access / source:** ✅ GEO `GSE223561`, `ncbi.nlm.nih.gov/geo` — series
  "Multimodal decoding of human liver regeneration."
- **Platform:** 10x Visium v1 (spot-based).
- **Scale (registry):** up to 66 sections in the human+mouse SuperSeries,
  ~18k genes, ~3k spots/section, **~9.1 GB** `RAW.tar`.
- **Important:** this is **human liver _regeneration_, NOT liver cancer** — the
  corrected registry description supersedes earlier (HCC/cancer) docs.
- **Raw counts:** `GSE223561_RAW.tar` (raw UMI counts per section).
- **Serial / Z fit:** the SuperSeries contains **multiple Visium sections** of
  the same regenerating-liver tissue, the registry's recommended serial-Visium
  source for Aether3D. Select human sections from one tissue block, order them,
  and assign `z_coord` by section index × physical spacing.
- **Schema mapping:** `spatial ← obsm['spatial']`;
  `z_coord ← ordered section index × physical section spacing`;
  `cell_class ← leiden/cluster` (or supplied annotation if present);
  `X ← raw UMI counts`. Subset to one tissue block / species before stacking
  to keep the Z axis physically meaningful.

---

## ⚠️ UNVERIFIED download URLs

Per the registry audit, the following hotlink is **not confirmed** and must be
replaced with the canonical page bundle before use:

1. **MERFISH (#14)** — `info.vizgen.com/hubfs/v1/...cell_by_gene.csv.gz` is a
   **guessed** hotlink. Obtain the real `cell_by_gene` + metadata bundle from the
   canonical page `info.vizgen.com/mouse-brain-data` (#14). Treat the size
   (3–7 GB) as an estimate until the real bundle is listed.

The 10x Visium HD pages (#4, #5) are verified, but prefer the
`binned_outputs.tar.gz` bundle from each dataset page over any direct single
`.h5` hotlink, which the registry also flags as unverified for HD products.

---

## Local resources

- **Existing data guide:** [`docs/DATA.md`](./DATA.md) — Python-native serial
  picks (MERFISH hypothalamus, Visium breast cancer serial, MOSTA) and the
  `aether_3d_serial_slice` schema contract that this guide reuses.
- **Advanced/optional real-data path:** [`docs_brca_imc_real_data.md`](../docs_brca_imc_real_data.md)
  — Kuett 2022 3D IMC (size-gated, not plug-and-play).
- **Machine-readable data cards:** `data/cards/*.yaml` (one card per dataset;
  `data/cards/` is tracked, `data/raw/` and `data/processed/` are gitignored).
- **Fetch / inspect entry point:** `scripts/data/fetch_aether_datasets.py`
  (`--list`, `--dataset <id> --dry-run`, `--save`). New registry datasets should
  be added as cards + specs here following the existing convention.
- **Verified papers (PDFs + index):** `../../references/`.
- **Provenance + corrected download commands:** `st_dataset_provenance_and_policy.md`
  (3 URLs flagged ⚠️ UNVERIFIED — includes the MERFISH hotlink above).
- **Suggested local cache path:** `~/Desktop/ST_research/data_cache/raw/<dataset_slug>/`
  (raw counts + spatial only).
- **Canonical registry & audit trail:** `ST_research/datasets/DATASET_REGISTRY.md`
  and `ST_research/audits/`.

## Next steps

Ingestion of these datasets is tracked as granular per-dataset GitHub issues
(label `data`) and sequenced in the dataset-integration PR. Each loader must
produce `z_coord`-tagged AnnData sections matching `aether_3d_serial_slice`, add
a `data/cards/*.yaml` card, register in `scripts/data/fetch_aether_datasets.py`,
ship a no-network smoke test, and add a row to this guide.
