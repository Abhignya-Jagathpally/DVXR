"""dvxr.bench.run — tie tasks + representations + baselines + protocol together.

For each task: run every configuration (5 representations + the baselines) through
repeated subject/patient-held-out CV, collect paired per-fold errors, pick the
best NON-fused opponent by mean CV error, and compute the relativity of the fused
proposed model against it (RER%, bootstrap CI, paired Wilcoxon, Cliff's delta).

Honesty is structural: the proposed model is only ever compared to the best
opponent the field could muster on the same folds, and we never assume it wins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

log = logging.getLogger("dvxr.bench")

# Fraction of NaN folds above which a config is flagged "unstable" (M2).
UNSTABLE_NAN_FRAC = 0.20
# Headline protocol label (M3): the current headline is single-level repeated CV;
# selection of the best opponent and RER evaluation share the folds. A nested-CV /
# frozen-test headline is DEFERRED (see CHANGES.md) — this string keeps that honest.
PROTOCOL_LABEL = "repeated subject/patient-held-out 5x5 CV (single-level; " \
                 "opponent selection and RER share folds — nested CV deferred)"

from dvxr.bench.baselines import baseline_configs, error_metric
from dvxr.bench.protocol import RelativityResult, holm_correction, relativity, repeated_group_folds
from dvxr.bench.representations import REPRESENTATIONS, evaluate_representation, pred_fused_e2e
from dvxr.bench.tasks import BenchTask
from dvxr.calibration import expected_calibration_error, fit_temperature_scaler

# The proposed model: CACMF encoder+VQ+cross-modal fusion produces the joint latent
# h, which the SAME shared head (used by every representation) is trained on. The
# encoder genuinely feeds the head (fixes B1) and every config is apples-to-apples.
PROPOSED = "rep:fused"
# also reported (transparency): CACMF predicting via its OWN end-to-end head.
PROPOSED_E2E = "cacmf_e2e"


@dataclass
class TaskResult:
    task: str
    metric: str
    per_config_fold_err: Dict[str, List[float]]
    best_baseline: str
    relativity: RelativityResult
    backend_note: str = ""
    protocol: str = PROTOCOL_LABEL
    failures: Dict[str, int] = field(default_factory=dict)   # config -> # failed folds
    unstable: List[str] = field(default_factory=list)        # configs NaN on >20% folds
    # classification only: pooled out-of-fold ECE per config, raw and after post-hoc
    # temperature scaling (how calibrated the probabilities are / could be).
    per_config_ece: Dict[str, float] = field(default_factory=dict)
    per_config_ece_ts: Dict[str, float] = field(default_factory=dict)

    def config_means(self) -> Dict[str, float]:
        return {c: float(np.nanmean(e)) for c, e in self.per_config_fold_err.items()}


def run_task(task: BenchTask, n_repeats: int = 5, n_folds: int = 5,
             seed: int = 7, include_sota: bool = True,
             representations: List[str] | None = None) -> TaskResult:
    reps = representations or list(REPRESENTATIONS.keys())
    folds = repeated_group_folds(task.subject_ids, n_repeats, n_folds, seed)

    baselines = baseline_configs(task, include_sota=include_sota)
    per_config: Dict[str, List[float]] = {f"rep:{r}": [] for r in reps}
    per_config[PROPOSED_E2E] = []
    per_config.update({b: [] for b in baselines})
    failures: Dict[str, int] = {}
    # pooled out-of-fold (truth, prob) per config for classification ECE
    pred_pool: Dict[str, List[tuple]] = {}

    def _run(name, thunk, fold_i):
        # M2: never swallow silently — log the exception with config+fold, count it,
        # record NaN so the sweep continues but the failure is visible.
        try:
            pred = thunk()
            if task.kind == "classification":
                pred_pool.setdefault(name, []).append(
                    (np.asarray(task.y[te], dtype=int), np.asarray(pred, dtype=float)))
            return error_metric(task, task.y[te], pred)
        except Exception as exc:
            failures[name] = failures.get(name, 0) + 1
            log.warning("config %s failed on fold %d (%s): %s",
                        name, fold_i, type(exc).__name__, exc)
            return float("nan")

    for fi, (tr, te) in enumerate(folds):
        for r in reps:
            per_config[f"rep:{r}"].append(
                _run(f"rep:{r}", lambda r=r: evaluate_representation(task, r, tr, te, seed=seed), fi))
        per_config[PROPOSED_E2E].append(
            _run(PROPOSED_E2E, lambda: pred_fused_e2e(task, tr, te, seed=seed), fi))
        for name, fn in baselines.items():
            per_config[name].append(
                _run(name, lambda fn=fn: fn(task, tr, te, seed=seed), fi))

    # M2: flag configs that NaN'd on >20% of folds; do not let them silently average.
    n = max(1, len(folds))
    unstable = sorted(c for c, e in per_config.items()
                      if np.mean([not np.isfinite(v) for v in e]) > UNSTABLE_NAN_FRAC)
    if failures:
        log.warning("run_task(%s): failure counts by config = %s", task.name, failures)
    if unstable:
        log.warning("run_task(%s): UNSTABLE configs (NaN on >%.0f%% of folds) = %s",
                    task.name, 100 * UNSTABLE_NAN_FRAC, unstable)

    # best opponent by mean CV error (selection on CV). Unstable configs are NOT
    # eligible to be the "best baseline" — a config that mostly failed must not win.
    proposed_key = PROPOSED
    field = {k: v for k, v in per_config.items()
             if k != proposed_key and k not in unstable}
    means = {k: float(np.nanmean(v)) for k, v in field.items()
             if np.isfinite(np.nanmean(v))}
    best = min(means, key=means.get)

    rel = relativity(task.name, task.metric, best,
                     per_config[proposed_key], per_config[best], seed=seed)
    note = str(task.extra.get("_sota_backend", "")) if include_sota else ""

    # Pooled out-of-fold ECE per config (classification only): how calibrated each
    # model's probabilities are (raw), and after post-hoc temperature scaling.
    ece: Dict[str, float] = {}
    ece_ts: Dict[str, float] = {}
    for name, chunks in pred_pool.items():
        if not chunks:
            continue
        yt = np.concatenate([c[0] for c in chunks])
        pp = np.clip(np.concatenate([c[1] for c in chunks]), 0.0, 1.0)
        if len(np.unique(yt)) < 2:
            continue
        ece[name] = float(expected_calibration_error(yt, pp))
        ece_ts[name] = float(expected_calibration_error(yt, fit_temperature_scaler(pp, yt).predict(pp)))

    return TaskResult(task.name, task.metric, per_config, best, rel, note,
                      protocol=PROTOCOL_LABEL, failures=failures, unstable=unstable,
                      per_config_ece=ece, per_config_ece_ts=ece_ts)


def finalize(results: List[TaskResult]) -> List[TaskResult]:
    """Apply Holm correction across tasks to the paired Wilcoxon p-values."""
    holm = holm_correction({r.task: r.relativity.p_wilcoxon for r in results})
    for r in results:
        r.relativity.p_holm = holm[r.task]
    return results
