"""dvxr.bench.ablation — TRUE modality ablation (retrain without the modality).

The old harness "ablated" a modality by zero-filling its columns, which leaves the
model keying off correlated features and shifts the input distribution. Here we
actually rebuild and retrain the fused model on the remaining modalities only, so
a modality's contribution is the honest held-out error change when it is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np

from dvxr.bench.baselines import error_metric
from dvxr.bench.protocol import bootstrap_ci, repeated_group_folds
from dvxr.bench.representations import evaluate_representation
from dvxr.bench.tasks import BenchTask


@dataclass
class AblationRow:
    dropped: str
    err_with_all: float
    err_without: float
    contribution: float          # err_without - err_with_all  (>0 => modality helps)
    ci: tuple


def _task_without(task: BenchTask, modality: str) -> BenchTask:
    feats = {m: a for m, a in task.features.items() if m != modality}
    names = {m: n for m, n in task.feature_names.items() if m != modality}
    return replace(task, features=feats, feature_names=names, extra=dict(task.extra))


def modality_ablation(task: BenchTask, n_repeats: int = 3, n_folds: int = 5,
                      seed: int = 7) -> List[AblationRow]:
    """Per-modality contribution to the fused model (retrain-without, not zero-fill)."""
    if len(task.modalities) < 2:
        return []                                    # nothing to ablate
    folds = repeated_group_folds(task.subject_ids, n_repeats, n_folds, seed)

    full = [error_metric(task, task.y[te],
                         evaluate_representation(task, "fused", tr, te, seed=seed))
            for tr, te in folds]
    full_mean = float(np.nanmean(full))

    rows: List[AblationRow] = []
    for m in task.modalities:
        reduced = _task_without(task, m)
        errs = [error_metric(reduced, reduced.y[te],
                             evaluate_representation(reduced, "fused", tr, te, seed=seed))
                for tr, te in folds]
        diff = np.asarray(errs) - np.asarray(full)   # paired per-fold
        rows.append(AblationRow(
            dropped=m, err_with_all=full_mean,
            err_without=float(np.nanmean(errs)),
            contribution=float(np.nanmean(diff)),
            ci=bootstrap_ci(diff[np.isfinite(diff)], seed=seed)))
    rows.sort(key=lambda r: r.contribution, reverse=True)
    return rows


def ablation_table(rows: List[AblationRow], metric: str) -> Dict[str, list]:
    return {
        "dropped_modality": [r.dropped for r in rows],
        f"err_without ({metric})": [round(r.err_without, 4) for r in rows],
        f"err_with_all ({metric})": [round(r.err_with_all, 4) for r in rows],
        "contribution": [round(r.contribution, 4) for r in rows],
        "ci_low": [round(r.ci[0], 4) for r in rows],
        "ci_high": [round(r.ci[1], 4) for r in rows],
    }
