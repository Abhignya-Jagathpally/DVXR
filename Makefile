PYTHON ?= python3

.PHONY: test mmf ablation paper all

test:
	$(PYTHON) -m unittest discover -s tests

mmf:
	$(PYTHON) scripts/run_mmf_full.py --profile
	$(PYTHON) scripts/run_mmf_full.py --realtime
	$(PYTHON) scripts/run_mmf_full.py --insight

ablation:
	$(PYTHON) scripts/run_ablation.py

# Regenerate result tables from outputs/, then build the PDF if pdflatex exists.
paper:
	$(PYTHON) scripts/build_paper_tables.py
	@if command -v pdflatex >/dev/null 2>&1; then \
		cd paper && pdflatex -interaction=nonstopmode main.tex && \
		( command -v bibtex >/dev/null 2>&1 && bibtex main || true ) && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex; \
	else \
		echo "[paper] pdflatex not found — generated tables only, skipped PDF build."; \
	fi

all: ablation mmf paper test
