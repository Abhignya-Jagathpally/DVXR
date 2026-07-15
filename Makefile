PYTHON ?= python3

.PHONY: test audit mmf ablation scoreboard-labram paper all

test:
	$(PYTHON) -m unittest discover -s tests

# The torch-free honesty-critical suite — what CI runs to prove "audit green" on a clean checkout.
audit:
	$(PYTHON) -m unittest tests.test_honesty_audit tests.test_docs_consistency tests.test_evidence \
		tests.test_paper_product tests.test_utility tests.test_external_evidence

mmf:
	$(PYTHON) scripts/run_mmf_full.py --profile
	$(PYTHON) scripts/run_mmf_full.py --realtime
	$(PYTHON) scripts/run_mmf_full.py --insight

ablation:
	$(PYTHON) scripts/run_ablation.py

# Regenerate the LaBraM-inclusive depression/workload scoreboard the honesty audit pins the
# depression headline to. HEAVY: needs torch + cached LaBraM weights; not part of `make all`.
# The committed outputs/_dnh_labram/benchmark_scoreboard.csv is the audited artifact — regenerate
# only when re-benchmarking (deterministic: same seed reproduces the same 1-AUROC cells).
scoreboard-labram:
	OMP_NUM_THREADS=4 $(PYTHON) scripts/run_benchmark.py --tasks mumtaz_depression eegmat_workload \
		--repeats 3 --folds 5 --out outputs/_dnh_labram

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
