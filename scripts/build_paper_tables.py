#!/usr/bin/env python3
"""build_paper_tables.py — emit paper/tables/*.tex.

Two families, both traceable and requiring no LaTeX install:
  - product tables (dvxr.serve.evidence + saved screeners): the honest headline results, the
    do-no-harm-vs-CACMF fusion contribution, DVXR-vs-published-SOTA, and decision-curve utility;
  - legacy CACMF-pipeline tables from outputs/*.csv (ablation, clinical metrics, codebook).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.eval.paper import build_paper_tables, build_product_tables  # noqa: E402


def main() -> int:
    total = 0
    # honest product tables (the paper's core evidence)
    prod = build_product_tables(ROOT / "paper" / "tables", ROOT / "outputs" / "product" / "screeners")
    for name, info in prod.items():
        print(f"[paper] {name}.tex  <-  {info['source']}  ({info['rows']} rows)")
    total += len(prod)
    # legacy CACMF-pipeline tables, if their source CSVs exist
    manifest = build_paper_tables(ROOT / "outputs", ROOT / "paper" / "tables")
    for name, info in manifest.items():
        print(f"[paper] {name}.tex  <-  {info['source']}  ({info['rows']} rows)")
    total += len(manifest)
    if not total:
        print("[paper] nothing to write")
        return 0
    print(f"[paper] wrote {total} table(s) to paper/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
