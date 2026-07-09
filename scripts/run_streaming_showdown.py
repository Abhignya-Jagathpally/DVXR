#!/usr/bin/env python3
"""run_streaming_showdown.py — the honest streaming / partial-observation win-hunt.

Sweeps modality dropout and compares the proposed models (CACMF fusion, a modality-
dropout-trained variant, and the soft-prompt LLM when available) against the strongest
floor (xgboost's native NaN handling, else a mean-impute linear floor). Reports the
crossover level where the proposed model genuinely beats the floor with a CI-backed
margin — or, honestly, that no such crossover exists. Writes outputs/streaming_showdown_*.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.bench.streaming_eval import partial_observation_showdown, write_showdown  # noqa: E402
from dvxr.bench.tasks import TASK_BUILDERS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="streaming partial-observation showdown")
    ap.add_argument("--tasks", nargs="+", default=["wesad_stress", "cgmacros_diabetes"],
                    choices=list(TASK_BUILDERS))
    ap.add_argument("--models", nargs="+", default=["fused", "fused_robust"],
                    help="proposed models to pit against the floor (add 'llm' if a local LLM is present)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-combos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args(argv)

    for name in args.tasks:
        print(f"[showdown] {name}: models={args.models} ...", flush=True)
        task = TASK_BUILDERS[name]()
        res = partial_observation_showdown(
            task, seed=args.seed, n_repeats=args.repeats, n_folds=args.folds,
            models=tuple(args.models), max_combos=args.max_combos)
        info = write_showdown(res, out_dir=args.out)
        verdict = (f"WIN at k={res['crossover_k']} ({res['crossover_model']})"
                   if res["crossover_k"] is not None else "no CI-backed crossover")
        print(f"[showdown]   -> {verdict}; wrote {info['md']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
