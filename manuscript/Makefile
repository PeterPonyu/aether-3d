.PHONY: pdf figures clean

# Build the manuscript PDF end-to-end:
#   1. Compile main.tex via explicit pdflatex + bibtex + pdflatex x2 chain.
#      We avoid `latexmk -pdf` here because it stops early when the .aux
#      file changes during the bibtex pass, producing an empty bibliography.
#
# Requires: pdflatex, bibtex
# TODO(ref-parity): figure-composition step removed with the stale figure
# scripts; figures are placeholders until real computational results exist.
pdf: figures
	pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
	bibtex main > /dev/null || true
	pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@echo "[manuscript] PDF built at main.pdf"

# TODO(ref-parity): figure composition is a no-op until real results land
# (the compose_figures.py generator was removed). main.tex uses placeholders.
figures:
	@echo "[manuscript] figure composition skipped (pending real-data results)"

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
	rm -f figures/*.pdf figures/*.png
