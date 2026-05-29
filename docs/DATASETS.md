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

## Extended validation datasets

Independent raw-count + serial/Z verification (2026-05-28) of the originally
recommended datasets against `ST_research/datasets/DATASET_REGISTRY.md`, plus
the genuinely-serial replacements/expansions filed as new ingestion issues.
Aether3D needs **raw integer counts AND multiple aligned sections along a real
Z axis**; a dataset can pass the raw-count test yet still fail the serial/3D
test (single section, or independent samples with no physical Z).

### Verification verdict table (existing recommendations)

| Issue | Dataset | Raw-count artifact | Raw count? | Serial / Z-axis suitability | Verdict | Action |
|---|---|---|:--:|---|:--:|---|
| #66 | MERFISH Mouse Brain Receptor Map (#14) | `cell_by_gene_S#R#.csv` (integer) + `cell_metadata_S#R#.csv` | ✅ | Only **3 coronal Z-levels** (3 slices × 3 *replicates* = redundancy, not depth) — coarse stack, not dense serial | ⚠️ | Keep (coarse 3D); dense-serial upgrade → **#71** |
| #67 | Visium HD Mouse Brain (#4) | `binned_outputs` → `filtered_feature_bc_matrix.h5` (integer) | ✅ | **Single capture = one 2D section**, no Z-axis, cannot stack alone | ❌ serial | Replacement → **#72** |
| #68 | Visium HD Tonsil (#5) | `binned_outputs` → `filtered_feature_bc_matrix.h5` (integer) | ✅ | **Single capture**, no adjacent serial tonsil sections released — least serial-suitable | ❌ serial | Replacement → **#72** |
| #69 | GSE223561 liver regeneration (#17) | `GSE223561_RAW.tar` MTX/TSV (integer UMI) | ✅ | SuperSeries = healthy/POD/APAP across timepoints & 2 species → **independent samples, NOT serial sections**; no physical Z | ❌ serial | Replacement → **#73** |

> **Correction retained:** GSE223561 is human liver **REGENERATION, NOT cancer** (~9.1 GB).

### New genuinely-serial / 3D datasets (filed as ingestion issues)

| Issue | Dataset | Accession / source | Platform | Sections / Z-availability | Raw-count artifact | Serial / Z fit |
|---|---|---|---|---|---|---|
| #71 | Allen/Zhuang 2023 MERFISH whole mouse brain | ABC Atlas `Zhuang-ABCA-1..4` ([portal](https://alleninstitute.github.io/abc_atlas_access/)) | MERFISH (single-cell) | **~147 serial coronal sections** of one brain + sagittal sets; `brain_section_label` + 3D CCF coords | ✅ raw cell-by-gene `.h5ad` + metadata + CCF coords (AWS `allen-brain-cell-atlas`) | ★ dense serial Z-stack — **dense-serial upgrade for #66** |
| #72 | 10x Visium Mouse Brain Serial Sections | [10x datasets](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-0-0) (CC BY 4.0) | Visium v1 (spot) | **4 serial sections** (Sagittal Anterior S1/S2 + Posterior S1/S2) | ✅ per-section `filtered_feature_bc_matrix.h5` (integer UMI) + `spatial/` | genuinely multi-section Visium — **replaces #67/#68 serial gap** |
| #73 | Stereo-seq MOSTA mouse organogenesis | [CNGB MOSTA](https://db.cngb.org/stomics/mosta/) / raw `CNP0001543` | Stereo-seq (sub-cellular) | E16.5 **13 serial sagittal sections of a single embryo** (53 total) → paper's **3D reconstruction** | ✅ `.gef` / `.h5ad` raw counts + spatial | ★ true serial 3D — **replaces #69** |
| #74 | STARmap PLUS adult mouse CNS 3D | Shi et al. 2023 *Nature* ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10709140/)); SODB / Zenodo `8092024` | STARmap PLUS (in-situ, 3D) | native **3D volumes** (194×194×345 nm voxels), per-cell x/y/**z** | ✅ per-cell gene counts + 3D coords (⚠️ exact hotlink unverified — use SODB) | volumetric 3D — serial/3D expansion |

> Per the OT-coupling + flow-matching design, every ingested stack must yield a
> Python list of `z_coord`-ordered AnnData sections (`SerialSliceTrajectoryDataset`
> requires ≥2 slices). Datasets marked ❌ above are retained only as
> single-section / coarse stress targets, never as the primary serial stack.

## Next steps

Ingestion of these datasets is tracked as granular per-dataset GitHub issues
(label `data`) and sequenced in the dataset-integration PR. Each loader must
produce `z_coord`-tagged AnnData sections matching `aether_3d_serial_slice`, add
a `data/cards/*.yaml` card, register in `scripts/data/fetch_aether_datasets.py`,
ship a no-network smoke test, and add a row to this guide.

---

## Consolidated dataset registry (draft PR #70)

Every `[data]` issue for Aether3D is consolidated into draft PR #70 at
**framework + registry + data-card** depth (not full runnable loaders, not
docs-only). Each dataset has a machine-readable card under `data/cards/*.yaml`
and is dispatched by the unified CLI `scripts/data/fetch_aether_datasets.py`
(`--list`, `--dataset <id> --dry-run`, `--fetch`). All entries map onto the
`aether_3d_serial_slice` contract (`obsm['spatial']` / `obs['z_coord']` /
`obs['cell_class']` / raw-count `X`).

| Issue | data_card_id | Platform | URL status | fetch_mode |
|---|---|---|:--:|---|
| #66 | `merfish_mouse_brain_receptor_map` | MERFISH (single-cell) | ⚠️ UNVERIFIED | `unverified` |
| #67 | `visium_hd_mouse_brain` | Visium HD (2 µm bins) | ✅ verified | `external` |
| #68 | `visium_hd_tonsil` | Visium HD (2 µm bins) | ✅ verified | `external` |
| #69 | `gse223561_liver_regeneration_serial` | Visium v1 | ✅ verified | `external` |

**fetch_mode semantics** — `python_native`: squidpy/scanpy one-liner loader
(wired, opt-in via `--fetch`); `external`: verified URL but multi-GB, prints the
canonical URL and never auto-downloads; `manifest_script`: delegates to
`scripts/data/prepare_brca_imc_kuett_2022.py --manifest-only`; `unverified`: the
source URL is a guess — the CLI raises `NotImplementedError` pointing at the
tracking issue and never fabricates a download URL.

> Per project data policy every ingested stack yields a Python list of
> `z_coord`-ordered AnnData sections (`SerialSliceTrajectoryDataset` requires
> ≥2 slices) with **raw integer count matrices + spatial metadata only** — no
> WSIs, FASTQs, BAMs, or normalized-only objects.

### Literature / source citations

No `LITERATURE_LINKS.md` exists in this repo, so the source-paper citations for
each consolidated dataset are recorded here:

- **#66 MERFISH Mouse Brain Receptor Map** — Vizgen showcase dataset,
  `info.vizgen.com/mouse-brain-data` (⚠️ direct hotlink unverified).
- **#67 / #68 Visium HD Mouse Brain / Tonsil** — 10x Genomics Visium HD CytAssist
  public datasets (10x Genomics Public Datasets License).
- **#69 GSE223561** — *Multimodal decoding of human liver regeneration*, GEO
  GSE223561 (human liver **regeneration**, NOT cancer; ~9.1 GB).

### Ingestion roadmap

Sequenced toward production loaders (each closes its `[data]` issue):

1. **Zero-glue Python-native first** — `merfish_hypothalamus_moffitt_2018` (#76)
   and `visium_breast_cancer_serial` (#77): squidpy/scanpy loaders are wired;
   add real download + `z_coord` split + contract tests, then close #76/#77.
2. **Verified-external serial stacks** — `visium_mouse_brain_serial_sections`
   (#72), `mosta_mouse_embryo_serial` (#73/#78/#179),
   `allen_merfish_whole_mouse_brain_serial` (#71), `gse223561_liver_regeneration_serial`
   (#69), `visium_hd_mouse_brain` (#67), `visium_hd_tonsil` (#68): staged manual
   download from the verified URLs, then loader + contract tests.
3. **Resolve UNVERIFIED URLs before ingestion** — `merfish_mouse_brain_receptor_map`
   (#66), `starmap_plus_mouse_cns_3d` (#74), `stereoseq_whole_mouse_brain_serial`
   (#75): confirm the canonical raw-count bundle (the CLI guards these with
   `NotImplementedError`); never invent a URL.
4. **Advanced / blocked** — `brca_imc_kuett_2022` (#79): run
   `scripts/data/prepare_brca_imc_kuett_2022.py --manifest-only`, then a
   `steinbock` pipeline to produce schema-valid AnnData before wiring the loader.
5. **Hygiene** — `.omc/`, `data/raw/`, and `data/processed/` are gitignored;
   `data/cards/` stays tracked (#174).

PR #70 stays a **draft** until at least the Tier-1 python-native loaders ship
runnable `z_coord`-tagged AnnData with passing contract tests.
