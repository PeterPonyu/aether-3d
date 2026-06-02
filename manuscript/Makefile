.PHONY: pdf figures clean

# Build the manuscript PDF end-to-end:
#   1. (Re)compose every figure from the synthetic benchmark JSONs
#   2. Compile main.tex via explicit pdflatex + bibtex + pdflatex x2 chain.
#      We avoid `latexmk -pdf` here because it stops early when the .aux
#      file changes during the bibtex pass, producing an empty bibliography.
#
# Requires: pdflatex, bibtex, matplotlib (use the dl env).
# Prerequisite on a fresh checkout: generate results/benchmark/*.json first by
# running scripts/ci/run_{synthetic_holdout,uot_ablation,scaling_sweep}.py from
# the repo root, otherwise compose_figures.py has no inputs. See README.md.
pdf: figures
	pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
	bibtex main > /dev/null || true
	pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@echo "[manuscript] PDF built at main.pdf"

# Compose multi-panel figures from results/benchmark/*.json
figures:
	python compose_figures.py

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
	rm -f figures/*.pdf figures/*.png
