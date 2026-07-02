"""dvxr.bench.run — tie tasks + representations + baselines + protocol together.

For each task: run every configuration (5 representations + the baselines) through
repeated subject/patient-held-out CV, collect paired per-fold errors, pick the
best NON-fused opponent by mean CV error, and compute the relativity of the fused
proposed model against it (RER%, bootstrap CI, paired Wilcoxon, Cliff's delta).

Honesty is structural: the proposed model is only ever compared to the best
opponent the field could muster on the same folds, and we never assume it wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from dvxr.bench.baselines import baseline_configs, error_metric
from dvxr.bench.protocol import RelativityResult, holm_correction, relativity, repeated_group_folds
from dvxr.bench.representations import REPRESENTATIONS, evaluate_representation, pred_fused_e2e
from dvxr.bench.tasks import BenchTask

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

    for tr, te in folds:
        for r in reps:
            try:
                pred = evaluate_representation(task, r, tr, te, seed=seed)
                per_config[f"rep:{r}"].append(error_metric(task, task.y[te], pred))
            except Exception:                            # keep the sweep alive
                per_config[f"rep:{r}"].append(float("nan"))
        try:
            pred = pred_fused_e2e(task, tr, te, seed=seed)
            per_config[PROPOSED_E2E].append(error_metric(task, task.y[te], pred))
        except Exception:
            per_config[PROPOSED_E2E].append(float("nan"))
        for name, fn in baselines.items():
            try:
                pred = fn(task, tr, te, seed=seed)
                per_config[name].append(error_metric(task, task.y[te], pred))
            except Exception:
                per_config[name].append(float("nan"))

    # best opponent by mean CV error (selection on CV, not per-fold test).
    # Proposed = rep:fused; it competes against every other config.
    proposed_key = PROPOSED
    field = {k: v for k, v in per_config.items() if k != proposed_key}
    means = {k: float(np.nanmean(v)) for k, v in field.items()
             if np.isfinite(np.nanmean(v))}
    best = min(means, key=means.get)

    rel = relativity(task.name, task.metric, best,
                     per_config[proposed_key], per_config[best], seed=seed)
    note = str(task.extra.get("_sota_backend", "")) if include_sota else ""
    return TaskResult(task.name, task.metric, per_config, best, rel, note)


def finalize(results: List[TaskResult]) -> List[TaskResult]:
    """Apply Holm correction across tasks to the paired Wilcoxon p-values."""
    holm = holm_correction({r.task: r.relativity.p_wilcoxon for r in results})
    for r in results:
        r.relativity.p_holm = holm[r.task]
    return results
