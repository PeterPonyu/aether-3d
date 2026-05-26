# Aether3D Round 6: BRCA IMC Kuett 2022 real-data branch

Branch: `feature/round6-brca-imc-kuett-real-data-20260526`

## Goal

Implement the first real-data preparation logic for Aether3D using the Kuett et
al. 3D imaging mass cytometry breast-cancer record. This is the smallest safe
Round 6 step toward replacing `validated-small` synthetic/contract evidence with
real serial-section data gates.

## Source

- Paper: Kuett et al., *Three-dimensional imaging mass cytometry for highly
  multiplexed molecular and cellular mapping of tissues and the tumor
  microenvironment*.
- Data availability: the paper states that IMC high-dimensional TIFF images,
  single-cell masks, and single-cell data for the 3D models are available at
  Zenodo DOI `10.5281/zenodo.4752030`.
- Code availability: the paper points to `BodenmillerGroup/3D_IMC_publication`
  for preprocessing/analysis code.

## What this branch implements

- `data/cards/brca_imc_kuett_2022.yaml`: Aether3D data-card gate.
- `scripts/data/prepare_brca_imc_kuett_2022.py`: Zenodo API discovery,
  manifest writing, and opt-in raw download logic.
- `tests/data/test_prepare_brca_imc.py`: no-network unit tests for Zenodo file
  parsing and manifest gate semantics.
- `.gitignore`: ignores `data/raw/` and `data/processed/` while keeping
  `data/cards/` tracked.

## Current status

Running

```bash
python scripts/data/prepare_brca_imc_kuett_2022.py --manifest-only
```

queries Zenodo record `4752030` and writes a manifest. A live check found 4
files totaling about 6.66 GB, so the script refuses downloads above the default
safety cap unless the user explicitly raises `--max-bytes` after storage and
license review.

This graduates A1 to **real-data source ready** but not metric/paper-ready yet:
AnnData conversion and virtual-slice holdout metrics are still the next gate.
