#!/usr/bin/env python3
"""Slice C ablation: does making VQ->LLM soft tokens IN-DISTRIBUTION help the frozen LLM?

The LLM-in-the-predictive-path (dvxr/llm/predictor.py) is the repo's weakest config because its
VQ-codebook -> LLM projection is an UNTRAINED random matrix, so the frozen Qwen reads
out-of-distribution soft prompts. Slice C's hypothesis: projecting VQ tokens onto convex
combinations of the LLM's REAL token embeddings (DVXR_LLM_INDIST=1) puts soft prompts inside the
model's own embedding distribution, letting the frozen LLM contribute real signal.

This runs, on one cohort, the frozen-LLM embedding under BOTH projections through subject-held-out
CV with the same shared head, and reports 1-AUROC vs the band-power single:eeg reference. An honest
negative (LLM still weak even in-distribution) is a valid, reportable outcome.

    python3 scripts/run_llm_indist_ablation.py --task eegmat_workload --repeats 3 --folds 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.bench.baselines import error_metric, _single_fn        # noqa: E402
from dvxr.bench.protocol import repeated_group_folds             # noqa: E402
from dvxr.bench.representations import _fit_head                 # noqa: E402
from dvxr.bench.tasks import TASK_BUILDERS                       # noqa: E402


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _llm_cv(task_name, indist: bool, repeats, folds, seed):
    # env must be set BEFORE get_reader() builds/caches the reader
    os.environ["DVXR_LLM_INDIST"] = "1" if indist else ""
    # fresh modules-free import path: build task, compute embeddings, CV
    from dvxr.llm.predictor import llm_window_embeddings
    task = TASK_BUILDERS[task_name]()
    emb = llm_window_embeddings(task, seed=seed)                 # (N, hidden), cached on task
    folds_ = repeated_group_folds(task.subject_ids, repeats, folds, seed)
    errs = []
    for tr, te in folds_:
        pred = _fit_head(task.kind, emb[tr], task.y[tr], emb[te], seed=seed)
        errs.append(error_metric(task, task.y[te], pred))
    return _mean(errs), task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Slice C LLM in-distribution ablation")
    ap.add_argument("--task", default="eegmat_workload")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    print(f"[llm-abl] {args.task}: random projection ...", flush=True)
    rand_err, task = _llm_cv(args.task, False, args.repeats, args.folds, args.seed)
    print(f"[llm-abl]   rep:llm(random)   1-AUROC = {rand_err:.4f}", flush=True)

    print(f"[llm-abl] {args.task}: in-distribution projection ...", flush=True)
    indist_err, _ = _llm_cv(args.task, True, args.repeats, args.folds, args.seed)
    print(f"[llm-abl]   rep:llm(indist)   1-AUROC = {indist_err:.4f}", flush=True)

    # reference: band-power single:eeg on the same folds
    folds_ = repeated_group_folds(task.subject_ids, args.repeats, args.folds, args.seed)
    eeg_fn = _single_fn("eeg") if "eeg" in task.modalities else None
    eeg_err = float("nan")
    if eeg_fn is not None:
        errs = [error_metric(task, task.y[te], eeg_fn(task, tr, te, seed=args.seed))
                for tr, te in folds_]
        eeg_err = _mean(errs)

    def rer(base, val):
        return 100 * (base - val) / base if base and np.isfinite(base) else float("nan")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Slice C — LLM in-distribution projection ablation ({args.task})\n",
          f"{args.repeats}x{args.folds} subject-held-out CV, error = 1-AUROC (lower better).\n",
          "| config | 1-AUROC | vs random | vs single:eeg |",
          "|---|---|---|---|",
          f"| rep:llm (random proj) | {rand_err:.4f} | — | {rer(eeg_err, rand_err):+.1f}% |",
          f"| rep:llm (in-distribution) | {indist_err:.4f} | {rer(rand_err, indist_err):+.1f}% | "
          f"{rer(eeg_err, indist_err):+.1f}% |",
          f"| single:eeg (band-power ref) | {eeg_err:.4f} | — | — |",
          "",
          f"In-distribution vs random: **{rer(rand_err, indist_err):+.1f}%** relative error change. "
          f"{'Helps' if indist_err < rand_err else 'Does not help'}. "
          f"{'Beats' if indist_err < eeg_err else 'Still loses to'} the band-power EEG baseline."]
    (out / "llm_indist_ablation.md").write_text("\n".join(md) + "\n")
    print("[llm-abl] wrote", out / "llm_indist_ablation.md", flush=True)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
