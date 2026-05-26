.PHONY: pdf figures clean

# Build the manuscript PDF end-to-end:
#   1. (Re)compose every figure from the synthetic benchmark JSONs
#   2. Run latexmk to compile main.tex
#
# Requires: pdflatex, latexmk, matplotlib (use the dl env)
pdf: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	@echo "[manuscript] PDF built at main.pdf"

# Compose multi-panel figures from results/benchmark/*.json
figures:
	python compose_figures.py

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
	rm -f figures/*.pdf figures/*.png
