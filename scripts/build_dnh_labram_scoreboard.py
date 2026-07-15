#!/usr/bin/env python3
"""build_dnh_labram_scoreboard.py — provenance tool for the depression headline's scoreboard.

`outputs/_dnh_labram/benchmark_scoreboard.csv` is the board the honesty audit pins the depression
claim to (`dvxr.serve.evidence`). It is produced by a real LaBraM benchmark run
(`make scoreboard-labram`), which needs torch + cached weights + the Mumtaz cohort. This tool makes
its provenance checkable and recoverable **without** any of that:

  --verify     (default) assert the committed board's mumtaz_depression base_err equals
               1 - (the committed screener manifest's held-out AUROC). A drift guard that ties the
               board to the served artifact; exits non-zero on mismatch or a missing board.
  --regenerate rebuild a MINIMAL board from the manifest alone (base_err = 1 - manifest AUROC;
               statistics that cannot be reconstructed offline are left blank, never invented) —
               a network-free fallback if the real board is ever lost. Prefer `make scoreboard-labram`
               (a genuine run) when torch/weights/data are available; this is the offline stopgap.

Reuses `dvxr.serve.evidence._manifest_auroc` so the manifest is read the same way the audit does.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.serve.evidence import _manifest_auroc  # noqa: E402

_BOARD = "outputs/_dnh_labram/benchmark_scoreboard.csv"
_MANIFEST = "outputs/product/screeners/mumtaz_depression"
_TASK = "mumtaz_depression"
_HEADER = ["task", "metric", "best_baseline", "base_err", "prop_err", "delta_abs", "RER_pct",
           "RER_CI_low", "RER_CI_high", "p_wilcoxon", "p_holm", "cliffs_delta", "n_folds",
           "meets_>=50%"]
_TOL = 5e-3


def _manifest_base_err() -> float:
    auroc = _manifest_auroc(_MANIFEST)
    if auroc is None:
        raise SystemExit(f"[dnh-scoreboard] cannot read manifest AUROC at {_MANIFEST}/manifest.json")
    return round(1.0 - auroc, 4)


def verify() -> int:
    """Assert the committed board's depression base_err matches the manifest-derived value."""
    board = ROOT / _BOARD
    if not board.exists():
        print(f"[dnh-scoreboard] MISSING committed board {_BOARD} — regenerate with "
              f"`make scoreboard-labram` (real run) or `--regenerate` (offline fallback).")
        return 1
    derived = _manifest_base_err()
    with open(board, newline="") as fh:
        row = next((r for r in csv.DictReader(fh) if r["task"] == _TASK), None)
    if row is None:
        print(f"[dnh-scoreboard] board present but has no {_TASK} row.")
        return 1
    board_err = float(row["base_err"])
    ok = abs(board_err - derived) <= _TOL
    print(f"[dnh-scoreboard] committed base_err={board_err}  manifest-derived (1-AUROC)={derived}  "
          f"->  {'MATCH ✓' if ok else 'MISMATCH ✗'}")
    return 0 if ok else 2


def regenerate() -> int:
    """Rebuild a minimal board from the manifest alone (offline fallback). Blanks what it can't derive."""
    base_err = _manifest_base_err()
    out = ROOT / _BOARD
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**{c: "" for c in _HEADER}, "task": _TASK, "metric": "1-AUROC",
             "best_baseline": "labram", "base_err": f"{base_err:.4f}"}]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"[dnh-scoreboard] wrote MINIMAL board from manifest -> {_BOARD} "
          f"(base_err={base_err:.4f}; unreconstructable stats left blank). "
          f"Regenerate the full board with a real run: `make scoreboard-labram`.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true", help="(default) check board vs manifest")
    g.add_argument("--regenerate", action="store_true", help="rebuild a minimal board offline")
    args = ap.parse_args(argv)
    return regenerate() if args.regenerate else verify()


if __name__ == "__main__":
    raise SystemExit(main())
