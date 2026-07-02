#!/usr/bin/env python3
"""run_benchmark.py — the actual science: real labels, held-out subjects, CIs.

Runs the CACMF fused model against real baselines (trivial, classical GBM, best
single modality, real pretrained SOTA encoder) on real-label tasks under repeated
subject/patient-held-out CV, with bootstrap CIs, paired Wilcoxon + Holm, and a
true retrain-without-modality ablation. Emits outputs/benchmark_scoreboard.{csv,md}.

Reports reality — the fused model is NOT assumed to win, and tasks that miss the
>=50% relative-error-reduction bar are printed as misses.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.bench.ablation import ablation_table, modality_ablation  # noqa: E402
from dvxr.bench.run import finalize, run_task  # noqa: E402
from dvxr.bench.scoreboard import write_scoreboard  # noqa: E402
from dvxr.bench.tasks import TASK_BUILDERS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CACMF real-label benchmark")
    ap.add_argument("--tasks", nargs="+", default=["stress", "glucose", "mortality"],
                    choices=list(TASK_BUILDERS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-sota", action="store_true", help="skip real SOTA encoders")
    ap.add_argument("--ablate", action="store_true",
                    help="also run the true modality ablation (multimodal tasks)")
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args(argv)

    results, ablation_by_task = [], {}
    for name in args.tasks:
        print(f"[bench] building task {name!r} ...", flush=True)
        task = TASK_BUILDERS[name]()
        print(f"[bench] running {name}: n={task.n} subjects={len(set(task.subject_ids))} "
              f"modalities={task.modalities}", flush=True)
        res = run_task(task, n_repeats=args.repeats, n_folds=args.folds,
                       seed=args.seed, include_sota=not args.no_sota)
        r = res.relativity
        print(f"[bench]   best_baseline={res.best_baseline} RER={r.rer_pct:.1f}% "
              f"CI=({r.rer_ci[0]:.1f},{r.rer_ci[1]:.1f}) p={r.p_wilcoxon:.4f}", flush=True)
        results.append(res)
        if args.ablate and len(task.modalities) > 1:
            print(f"[bench]   ablating modalities of {name} ...", flush=True)
            rows = modality_ablation(task, n_repeats=min(3, args.repeats),
                                     n_folds=args.folds, seed=args.seed)
            ablation_by_task[name] = ablation_table(rows, task.metric)

    results = finalize(results)
    meta = {"repeats": args.repeats, "folds": args.folds, "seed": args.seed,
            "sota": not args.no_sota}
    info = write_scoreboard(results, out_dir=args.out,
                            ablation_by_task=ablation_by_task or None, meta=meta)
    print(f"[bench] wrote {info['md']} + {info['csv']} ({info['n_tasks']} tasks)")
    for res in results:
        rel = res.relativity
        print(f"  - {res.task}: RER {rel.rer_pct:.1f}% "
              f"(meets >=50%: {rel.meets_target()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
