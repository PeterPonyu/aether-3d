# Aether3D manuscript build

`main.pdf` is built from `main.tex` + `refs.bib` and three composed figures.

## Prerequisite: generate the benchmark JSONs

The figures are composed from `results/benchmark/*.json`, which are **not**
committed (the `results/` tree is gitignored). On a fresh checkout you must
generate them first, from the repository root, in the `dl` environment (these
scripts import `aether_3d`, so the package must be installed editable or run
via `scripts/run.sh`):

```bash
python scripts/ci/run_synthetic_holdout.py   # -> results/benchmark/synthetic_holdout.json
python scripts/ci/run_uot_ablation.py        # -> results/benchmark/uot_ablation.json
python scripts/ci/run_scaling_sweep.py       # -> results/benchmark/scaling_curve.json
```

Without these JSONs, `compose_figures.py` prints
`WARN: missing results/benchmark/...; skipping` and produces no figure PDFs, so
the subsequent `pdflatex` pass fails on the missing `\includegraphics` targets
(issue #100).

## Build the PDF

```bash
cd manuscript
make pdf        # composes figures (from the JSONs above) then runs pdflatex+bibtex+pdflatex x2
```

`make pdf` runs `compose_figures.py` first; the figure PDFs it writes under
`figures/` are gitignored and regenerated on each build.

Requirements: `pdflatex`, `bibtex`, and `matplotlib` (use the `dl` env). The
benchmark JSON generation step above additionally requires the full `aether_3d`
runtime dependencies (incl. `torch`).
