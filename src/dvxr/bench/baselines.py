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


# ------------------------------------------------- stronger classical floors
def pred_xgboost(task, tr, te, seed=7):
    """XGBoost on concatenated raw features — the tuned-GBM floor a fusion/LLM method
    must beat. Import-guarded: raises cleanly if xgboost is absent (run.py logs + skips)."""
    import xgboost as xgb  # noqa: F401 (guarded)

    X = _concat(task)
    if task.kind == "classification":
        if len(np.unique(task.y[tr])) < 2:
            return np.full(len(te), float(np.mean(task.y[tr])))
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=2)
        m.fit(X[tr], task.y[tr])
        return m.predict_proba(X[te])[:, list(m.classes_).index(1)]
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2)
    return m.fit(X[tr], task.y[tr]).predict(X[te])


def pred_tabpfn(task, tr, te, seed=7):
    """TabPFN-v2 — SOTA on small tabular. Classification only; capped train size.
    Import-guarded (raises cleanly if tabpfn is absent)."""
    from tabpfn import TabPFNClassifier  # noqa: F401 (guarded)

    if task.kind != "classification" or len(np.unique(task.y[tr])) < 2:
        raise RuntimeError("tabpfn: classification with 2 classes only")
    X = _concat(task)
    # TabPFN v2 handles up to ~10k rows / 500 features; subsample train if larger.
    idx = tr
    if len(tr) > 3000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(tr, size=3000, replace=False)
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(X[idx][:, :500], task.y[idx])
    proba = clf.predict_proba(X[te][:, :500])
    classes = list(clf.classes_)
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]


def pred_ridge_history(task, tr, te, seed=7):
    """CGM forecast floor: ridge on the glucose-history feature vector (beyond bare
    persistence). Forecast tasks only."""
    if task.kind != "forecast":
        raise RuntimeError("ridge_history: forecast tasks only")
    X = task.features["cgm"]
    return _fit_head("forecast", X[tr], task.y[tr], X[te], seed=seed)


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
    if backend is None or not hasattr(backend, "_embed"):
        # e.g. CGM-JEPA cannot load as an HF text model — fail cleanly so run.py logs
        # it once per fold and marks the sota config "unstable" (never silent).
        raise RuntimeError(f"no real SOTA backend available for modality {modality!r}")
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
    # stronger floors — import-guarded predictors raise cleanly when their dep is absent,
    # so run.py logs+skips them and the offline suite still passes. Only register the
    # ones whose dep is importable, so absent deps don't spam per-fold failures.
    if _importable("xgboost"):
        cfgs["xgboost"] = pred_xgboost
    if task.kind == "classification" and _importable("tabpfn"):
        cfgs["tabpfn"] = pred_tabpfn
    if task.kind == "forecast":
        cfgs["ridge_history"] = pred_ridge_history
    # raw-signal lever: a multimodal 1D-CNN over RAW windows (the honest path past the
    # summary-stat ceiling, C2). Registered only when the task carries per-modality raw
    # windows in extra["raw"] — so it competes head-to-head against the summary-stat floor
    # on the exact same folds. Classification only (the CNN head is a classifier).
    if task.kind == "classification" and isinstance(task.extra.get("raw"), dict):
        from dvxr.bench.raw_seq import pred_rawcnn
        cfgs["raw_cnn"] = pred_rawcnn
    if include_sota:
        cfgs["sota"] = pred_sota
    return cfgs


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
