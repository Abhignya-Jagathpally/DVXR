#!/usr/bin/env python3
"""Ablation: does the 1-SE candidate-selection rule close DNH's held-out do-no-harm gap?

Compares dnh_gated (strict argmin best-candidate) vs dnh_gated (1-SE rule: prefer the
simpler candidate within one bootstrap SE) on the SAME 5x5 subject-held-out folds, plus
the best single modality and the concat-GBM floor for reference. Focused + fast: it runs
only the two DNH variants + references (no rep:fused/pca/neural/vq), so it isolates the
rule's effect without a full 30-min sweep.

    python3 scripts/run_dnh_ablation.py --profile mh --repeats 5 --folds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.bench.baselines import error_metric, _single_fn        # noqa: E402
from dvxr.bench.gated_fusion import pred_dnh_gated               # noqa: E402
from dvxr.bench.protocol import repeated_group_folds, relativity  # noqa: E402
from dvxr.bench.tasks import TASK_BUILDERS                        # noqa: E402

PROFILES = {
    "mh": ["stress", "wesad_stress", "deap_anxiety", "deap_arousal",
           "eegmat_workload", "mumtaz_depression"],
    "clinical": ["stress", "glucose", "mortality"],
}


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DNH 1-SE ablation")
    ap.add_argument("--profile", choices=list(PROFILES), default="mh")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    tasks = args.tasks or PROFILES[args.profile]
    md = ["# DNH 1-SE ablation — does the simpler-candidate rule close the held-out gap?\n",
          f"{args.repeats}x{args.folds} subject-held-out CV. Error = 1-AUROC (lower better). "
          "`dnh_strict` = best candidate by inner-CV argmin; `dnh_1se` = 1-SE rule (prefer the "
          "simpler candidate within one bootstrap SE). RER vs. best single modality > 0 means "
          "do-no-harm holds on held-out subjects.\n",
          "| task | best_single | dnh_strict | dnh_1se | strict RER vs single | 1se RER vs single |",
          "|---|---|---|---|---|---|"]
    for name in tasks:
        print(f"[abl] {name} ...", flush=True)
        task = TASK_BUILDERS[name]()
        folds = repeated_group_folds(task.subject_ids, args.repeats, args.folds, args.seed)
        singles = {m: _single_fn(m) for m in task.modalities}
        strict_err, se_err, single_err = [], [], {m: [] for m in task.modalities}
        for tr, te in folds:
            yte = task.y[te]
            strict_err.append(error_metric(task, yte,
                              pred_dnh_gated(task, tr, te, seed=args.seed, strict=True)))
            se_err.append(error_metric(task, yte,
                          pred_dnh_gated(task, tr, te, seed=args.seed, strict=False)))
            for m in task.modalities:
                single_err[m].append(error_metric(task, yte, singles[m](task, tr, te, seed=args.seed)))
        bs = min(task.modalities, key=lambda m: _mean(single_err[m]))
        bs_err = _mean(single_err[bs])
        ds, dse = _mean(strict_err), _mean(se_err)
        def rer(base, val):
            return 100 * (base - val) / base if base else float("nan")
        md.append(f"| {name} | single:{bs} {bs_err:.4f} | {ds:.4f} | {dse:.4f} | "
                  f"{rer(bs_err, ds):+.1f}% | {rer(bs_err, dse):+.1f}% |")
        print(f"[abl]   best_single(single:{bs})={bs_err:.4f} strict={ds:.4f} "
              f"1se={dse:.4f} | RER strict {rer(bs_err, ds):+.1f}% 1se {rer(bs_err, dse):+.1f}%",
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dnh_ablation_1se.md").write_text("\n".join(md) + "\n")
    print(f"[abl] wrote {out/'dnh_ablation_1se.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
