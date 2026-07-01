#!/usr/bin/env python3
"""build_paper_tables.py — emit paper/tables/*.tex from outputs/ (Goal 4).

Every number traces to a file under outputs/. No LaTeX install required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.eval.paper import build_paper_tables  # noqa: E402


def main() -> int:
    manifest = build_paper_tables(ROOT / "outputs", ROOT / "paper" / "tables")
    if not manifest:
        print("[paper] no source outputs/ found; run run_ablation.py / run_goal1_full.py first")
        return 0
    for name, info in manifest.items():
        print(f"[paper] {name}.tex  <-  {info['source']}  ({info['rows']} rows)")
    print(f"[paper] wrote {len(manifest)} table(s) to paper/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
