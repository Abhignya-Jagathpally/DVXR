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


def _write_llm_interpretability(task_name: str, task, out_dir: str) -> None:
    """Interpretability artifact for the LLM predictor: per-modality attribution
    (how much each modality moves the frozen-LLM representation) + backend used."""
    import json
    from pathlib import Path

    from dvxr.llm.predictor import modality_attribution
    try:
        attr = modality_attribution(task)
    except Exception as exc:  # never fail the benchmark on the explainer
        print(f"[bench]   interpretability skipped for {task_name}: {exc}", flush=True)
        return
    payload = {"task": task_name, "backend": task.extra.get("_llm_backend", ""),
               "modality_attribution": attr}
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"llm_interpretability_{task_name}.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# LLM predictor interpretability — {task_name}",
             f"\nBackend: `{payload['backend']}`\n",
             "Modality attribution — share of the frozen-LLM representation shift when each "
             "modality is replaced by its learned absent token (higher = more influential):\n"]
    for m, v in sorted(attr.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{m}**: {v:.3f}")
    (p / f"llm_interpretability_{task_name}.md").write_text("\n".join(lines) + "\n")
    print(f"[bench]   wrote LLM interpretability for {task_name}: {attr}", flush=True)


def _write_llm_robustness(task_name: str, task, out_dir: str,
                          repeats: int, folds: int, seed: int) -> None:
    """Interoperability artifact: missing-modality robustness of the LLM predictor —
    train on all modalities, test with each dropped. Graceful degradation, no crash."""
    import json
    from pathlib import Path

    from dvxr.llm.predictor import missing_modality_robustness
    try:
        rows = missing_modality_robustness(task, seed=seed, n_repeats=repeats, n_folds=folds)
    except Exception as exc:
        print(f"[bench]   robustness skipped for {task_name}: {exc}", flush=True)
        return
    if not rows:  # forecast task — no classification robustness curve
        return
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"llm_robustness_{task_name}.json").write_text(json.dumps(rows, indent=2))
    base = next(r for r in rows if r["dropped"] == "none")["err"]
    lines = [f"# LLM predictor — missing-modality robustness ({task_name})",
             "\nThe shared head is trained ONCE on all modalities; at test time each "
             "modality is individually replaced by its learned absent token. A single-"
             "modality model cannot do this at all — it needs its one modality present.\n",
             f"Full-modality test error (1-AUROC): **{base:.4f}**\n",
             "| dropped at test | 1-AUROC | degradation |",
             "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dropped']} | {r['err']:.4f} | {r['degradation']:+.4f} |")
    (p / f"llm_robustness_{task_name}.md").write_text("\n".join(lines) + "\n")
    print(f"[bench]   wrote LLM robustness for {task_name} (base err {base:.4f})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CACMF real-label benchmark")
    ap.add_argument("--tasks", nargs="+", default=["stress", "glucose", "mortality"],
                    choices=list(TASK_BUILDERS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--loso", action="store_true",
                    help="true leave-one-subject-out: n_folds = #subjects, n_repeats = 1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-sota", action="store_true", help="skip real SOTA encoders")
    ap.add_argument("--llm", action="store_true",
                    help="also evaluate the rep:llm soft-prompt LLM predictor (needs a local LLM)")
    ap.add_argument("--ablate", action="store_true",
                    help="also run the true modality ablation (multimodal tasks)")
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args(argv)

    results, ablation_by_task = [], {}
    for name in args.tasks:
        print(f"[bench] building task {name!r} ...", flush=True)
        task = TASK_BUILDERS[name]()
        n_subj = len(set(task.subject_ids))
        repeats, folds = args.repeats, args.folds
        if args.loso:
            repeats, folds = 1, n_subj  # true leave-one-subject-out
        reps = None  # default sweep
        if args.llm:
            from dvxr.bench.representations import DEFAULT_REPS, llm_available
            if llm_available():
                reps = DEFAULT_REPS + ["llm"]
            else:
                print("[bench]   --llm requested but no local LLM available; skipping rep:llm "
                      "(set DVXR_LLM_ALLOW_DOWNLOAD=1 or pre-download the model).", flush=True)
        print(f"[bench] running {name}: n={task.n} subjects={n_subj} "
              f"modalities={task.modalities} (repeats={repeats}, folds={folds})", flush=True)
        res = run_task(task, n_repeats=repeats, n_folds=folds,
                       seed=args.seed, include_sota=not args.no_sota, representations=reps)
        if args.llm and reps and "_llm_emb" in task.extra:
            _write_llm_interpretability(name, task, args.out)
            _write_llm_robustness(name, task, args.out, repeats, folds, args.seed)
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
