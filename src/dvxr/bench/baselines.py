"""dvxr.bench.baselines — the opponents the proposed model must actually beat.

  * majority / persistence — trivial floor (no-skill)
  * classical              — HistGradientBoosting on raw features (the strong
                             classical model; this is essentially the current repo)
  * single:<modality>      — one modality's raw features -> shared head
  * sota:<fm>              — a real pretrained foundation model as a FROZEN feature
                             extractor -> shared head (MOMENT / CGM-JEPA / Bio_ClinicalBERT)

SOTA embeddings use the RAW frozen-FM output (no PCA). The FM is unsupervised (never
sees labels), so its forward pass over all rows is leakage-free; ALL supervised fitting
(the per-fold StandardScaler + head) happens on train indices only. This fixes the
earlier transductive-PCA leak (M5): no projection is fit over test rows.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from dvxr.bench.representations import _concat, _fit_head
from dvxr.bench.tasks import BenchTask


# ------------------------------------------------------------------- metrics
def error_metric(task: BenchTask, y_true, pred) -> float:
    """Primary ERROR (lower is better) for the task."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=float)
    if task.kind == "classification":
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(1.0 - roc_auc_score(y_true, pred))
    return float(np.mean(np.abs(pred - y_true)))         # MAE


# --------------------------------------------------------------- trivial floor
def pred_trivial(task, tr, te, seed=7):
    if task.baseline_hint == "persistence":
        col = task.extra["persistence_col"]
        return task.features["cgm"][te][:, col]          # last observed glucose
    return np.full(len(te), float(np.mean(task.y[tr])))  # majority prevalence


def pred_classical(task, tr, te, seed=7):
    X = _concat(task)
    if task.kind == "classification":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(random_state=seed).fit(X[tr], task.y[tr])
        if len(m.classes_) < 2:
            return np.full(len(te), float(m.classes_[0]))
        return m.predict_proba(X[te])[:, list(m.classes_).index(1)]
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed).fit(
        X[tr], task.y[tr]).predict(X[te])


def _single_fn(modality: str) -> Callable:
    def fn(task, tr, te, seed=7):
        X = task.features[modality]
        return _fit_head(task.kind, X[tr], task.y[tr], X[te], seed=seed)
    return fn


# --------------------------------------------------------------- SOTA (frozen)
_FM_FOR_TASK = {"stress": "wearable_phys", "glucose": "cgm", "mortality": "ehr"}


def _sota_embeddings(task: BenchTask) -> np.ndarray:
    """Cache the RAW frozen-FM embedding for every row (no PCA -> no transductive leak).

    Only the FM forward pass runs here (unsupervised, label-free); dimensionality is
    handled by the per-fold shared head (StandardScaler + logistic/ridge on train only).
    """
    if "_sota_emb" in task.extra:
        return task.extra["_sota_emb"]
    import pandas as pd

    from dvxr.config import DEFAULTS
    from dvxr.encoders.base import make_primary_backend

    modality = _FM_FOR_TASK.get(task.name, "wearable_phys")
    X = _concat(task)
    cols = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=cols)
    cfg = DEFAULTS.with_(d=16, use_real_weights=True, allow_download=True, seed=7)
    backend = make_primary_backend(modality, cfg)
    emb = np.asarray(backend._embed(frame, cols), dtype=float)   # RAW, no PCA
    task.extra["_sota_emb"] = emb
    task.extra["_sota_backend"] = getattr(backend, "name",
                                          getattr(backend, "used_encoder", type(backend).__name__))
    return emb


def pred_sota(task, tr, te, seed=7):
    # per-fold: StandardScaler + head fit on TRAIN raw-FM embedding only (no leak)
    emb = _sota_embeddings(task)
    return _fit_head(task.kind, emb[tr], task.y[tr], emb[te], seed=seed)


# --------------------------------------------------------------- config set
def baseline_configs(task: BenchTask, include_sota: bool = True) -> Dict[str, Callable]:
    """All non-fused opponents for a task (name -> predictor)."""
    cfgs: Dict[str, Callable] = {
        task.baseline_hint: pred_trivial,
        "classical_gbm": pred_classical,
    }
    for m in task.modalities:
        cfgs[f"single:{m}"] = _single_fn(m)
    if include_sota:
        cfgs["sota"] = pred_sota
    return cfgs
