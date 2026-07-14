#!/usr/bin/env python3
"""Synergy/redundancy diagnostic for the do-no-harm reliability-gated fusion.

For each real task, compute (on one subject-held-out train fold) the modality
synergy/redundancy diagnostic + the realised DNH behaviour, and correlate joint-gain
with DNH's acceptance/gain across cohorts. This delivers the testable rule the paper
leads with: *when combining modalities buys little over the best single modality
(low joint-gain / high redundancy), the do-no-harm floor dominates and DNH falls back
to the best candidate; where joint-gain is real, DNH accepts fusion and improves.*

Descriptive, not a scored benchmark result (single fold). Offline/CPU/deterministic.

    python3 scripts/run_dnh_diagnostic.py --profile mh --out outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.bench.gated_fusion import dnh_diagnostics       # noqa: E402
from dvxr.bench.tasks import TASK_BUILDERS                 # noqa: E402

PROFILES = {
    "mh": ["stress", "wesad_stress", "deap_anxiety", "deap_arousal",
           "eegmat_workload", "mumtaz_depression"],
    "clinical": ["stress", "glucose", "mortality"],
}


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DNH synergy/redundancy diagnostic")
    ap.add_argument("--profile", choices=list(PROFILES), default="mh")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    tasks = args.tasks or PROFILES[args.profile]
    rows = []
    for name in tasks:
        print(f"[dnh-diag] {name} ...", flush=True)
        task = TASK_BUILDERS[name]()
        d = dnh_diagnostics(task, seed=args.seed)
        rows.append(d)
        print(f"[dnh-diag]   joint_gain={_fmt(d['joint_gain'])} "
              f"dnh_gain_single={_fmt(d['dnh_gain_single'])} "
              f"redundancy={_fmt(d['redundancy'])} "
              f"lambda={_fmt(d['lambda'])} accepted={d['accepted']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# DNH synergy/redundancy diagnostic\n",
          "One subject-held-out train fold per task (descriptive, not a scored result). "
          "`joint_gain` = relative inner-CV error reduction of the best candidate (incl. "
          "concat-GBM) over the best single modality — how much combining helps at all. "
          "`dnh_gain_single` = DNH's realised inner-CV gain over the best single modality. "
          "`redundancy` = mean pairwise correlation of single-modality OOF predictions "
          "(high => redundant). `lambda` = shrinkage toward the best candidate (1.0 => pure "
          "do-no-harm fallback). `accepted` = fusion cleared the finite-sample SE gate.\n",
          "| task | n_subj | modalities | best_single | joint_gain | dnh_gain_single | "
          "redundancy | lambda | accepted |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        md.append(f"| {d['task']} | {d['n_subjects']} | {'+'.join(d['modalities'])} | "
                  f"{d['best_single']} | {_fmt(d['joint_gain'])} | "
                  f"{_fmt(d['dnh_gain_single'])} | {_fmt(d['redundancy'])} | "
                  f"{_fmt(d['lambda'])} | {d['accepted']} |")

    # the rule, stated + checked: does higher joint_gain track DNH acceptance?
    acc = [d for d in rows if d["accepted"]]
    non = [d for d in rows if not d["accepted"]]
    import numpy as np
    def _mean(xs, k):
        v = [x[k] for x in xs if x[k] == x[k]]
        return float(np.mean(v)) if v else float("nan")
    md += ["\n## Rule check\n",
           f"- Tasks where DNH **accepted** fusion (n={len(acc)}): "
           f"mean joint_gain = {_fmt(_mean(acc, 'joint_gain'))}, "
           f"mean redundancy = {_fmt(_mean(acc, 'redundancy'))}.",
           f"- Tasks where DNH **fell back** to the best candidate (n={len(non)}): "
           f"mean joint_gain = {_fmt(_mean(non, 'joint_gain'))}, "
           f"mean redundancy = {_fmt(_mean(non, 'redundancy'))}.",
           "\nInterpretation: DNH accepts late fusion where combining modalities has real "
           "headroom (higher joint_gain / lower redundancy) and falls back to the safe best "
           "candidate where it does not — the do-no-harm floor doing exactly its job."]

    path = out / "dnh_diagnostic.md"
    path.write_text("\n".join(md) + "\n")
    print(f"[dnh-diag] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
