"""dvxr.eval.ablation — Goal-3 fused-vs-single-modality ablation (ARCHITECTURE §A4/§A7).

For each task we evaluate, on a subject-held-out split:
  * each single modality alone,
  * each of the five fusion strategies,
  * each of the three aggregation baselines,
using a frozen (seeded) CACMF encoder + a linear probe (logistic / ridge). This is a
standard, fast, deterministic linear-probe protocol. We DO NOT assert fused ≥ single —
the harness reports whatever actually happens.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dvxr.config import AGGREGATIONS, DEFAULTS, FUSION_STRATEGIES
from dvxr.encoders.base import _torch_available
from dvxr.eval.metrics import classification_metrics, forecast_metrics
from dvxr.eval.splits import subject_holdout_split
from dvxr.fusion.aggregate import AGGREGATORS

_METRIC_KEYS = ["auroc", "auprc", "f1", "accuracy", "ece",
               "mae", "interval_radius", "coverage"]


def _encode_all(model, feats):
    import torch
    with torch.no_grad():
        ft = {m: torch.tensor(feats[m], dtype=torch.float32) for m in feats}
        z = model.encode(ft)
        return {m: z[m].numpy() for m in z}


def _fuse_h(model, feats):
    import torch
    with torch.no_grad():
        ft = {m: torch.tensor(feats[m], dtype=torch.float32) for m in feats}
        z = model.encode(ft)
        return model.cacmf.fuse_result(z).h.numpy()          # request-local (no _last*)


def _clf_proba(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)          # (Nte, n_classes)
    # normalize to a 2-column [neg, pos] layout
    if p.shape[1] == 1:
        only = int(clf.classes_[0])
        col = np.zeros((p.shape[0], 2))
        col[:, only] = 1.0
        return col
    return p


def _ridge(Xtr, ytr, Xte, Xtr2):
    from sklearn.linear_model import Ridge
    r = Ridge().fit(Xtr, ytr)
    return r.predict(Xte), r.predict(Xtr2)


def _row(task, ctype, cname, present, ntr, nte, metrics):
    row = {"task": task, "config_type": ctype, "config_name": cname,
           "present_modalities": present, "n_train": ntr, "n_test": nte}
    for k in _METRIC_KEYS:
        row[k] = metrics.get(k, float("nan"))
    return row


def make_synthetic_dataset(n_subjects: int = 12, per_subject: int = 12,
                           seed: int = 0) -> Dict:
    """A small multimodal dataset with subject structure + learnable labels.

    A classification task (stress) driven by a wearable feature + subject bias, and a
    forecast task (glucose) driven by a CGM feature. Used for offline ablation demos.
    """
    rng = np.random.default_rng(seed)
    dims = {"eeg": 10, "wearable_phys": 8, "cgm": 6}
    feats = {m: [] for m in dims}
    sids, stress, glucose = [], [], []
    for s in range(n_subjects):
        bias = rng.normal(0, 0.7)
        for _ in range(per_subject):
            row = {m: rng.normal(0, 1, d).astype(np.float32) for m, d in dims.items()}
            for m in dims:
                feats[m].append(row[m])
            sids.append(f"subj{s}")
            stress.append(int(row["wearable_phys"][0] + bias > 0))
            glucose.append(120.0 + 30.0 * row["cgm"][0] + 5.0 * bias)
    feats = {m: np.vstack(v).astype(np.float32) for m, v in feats.items()}
    return {
        "features": feats,
        "subject_ids": np.array(sids),
        # synthetic fixtures co-register every modality on the same synthetic subject, so they are
        # genuinely synchronized and the fusion gate lets them through (see dvxr.cohort).
        "cohort_id": "synthetic",
        "tasks": {
            "stress_detection": {"kind": "classification", "y": np.array(stress)},
            "glucose": {"kind": "forecast", "y": np.array(glucose)},
        },
    }


def run_ablation(dataset: Dict, config=DEFAULTS, test_frac: float = 0.3,
                 seed: int = 7,
                 fusion_strategies: Optional[List[str]] = None,
                 aggregations: Optional[List[str]] = None) -> pd.DataFrame:
    """Return one row per (task × configuration) with held-out metrics."""
    if not _torch_available():
        raise RuntimeError("run_ablation requires torch")

    from dvxr.tasks.model import build_multitask_model

    feats = dataset["features"]
    sids = np.asarray(dataset["subject_ids"])
    tasks = dataset["tasks"]
    mods = list(feats.keys())
    present = "|".join(mods)
    input_dims = {m: feats[m].shape[1] for m in mods}

    # Honesty gate (spec §1.B/§4): a multimodal fusion claim is only valid on a cohort that genuinely
    # co-registers those modalities on the same subjects. When the dataset declares its cohort, refuse
    # to compute fused rows over a modality set the cohort does not synchronize (e.g. EEG+CGM, which no
    # public cohort co-registers). Single-modality rows are always allowed.
    cohort_id = dataset.get("cohort_id")
    if cohort_id is not None and len(mods) > 1:
        from dvxr.cohort import require_synchronized_for_fusion
        require_synchronized_for_fusion(cohort_id, mods)

    tr, te = subject_holdout_split(sids, test_frac, seed)

    base = build_multitask_model(config, input_dims)
    base.eval()
    z = _encode_all(base, feats)

    strat_h = {}
    for s in (fusion_strategies or FUSION_STRATEGIES):
        m_s = build_multitask_model(config.with_(fusion_strategy=s), input_dims)
        m_s.eval()
        strat_h[s] = _fuse_h(m_s, feats)

    rows: List[dict] = []
    for task, spec in tasks.items():
        y = np.asarray(spec["y"])
        kind = spec.get("kind", "classification")

        if kind == "classification":
            ytr, yte = y[tr].astype(int), y[te].astype(int)
            if len(np.unique(ytr)) < 2:
                continue
            for m in mods:
                prob = _clf_proba(z[m][tr], ytr, z[m][te])[:, 1]
                rows.append(_row(task, "single", m, present, len(tr), len(te),
                                 classification_metrics(yte, prob)))
            for s, h in strat_h.items():
                prob = _clf_proba(h[tr], ytr, h[te])[:, 1]
                rows.append(_row(task, "fusion", s, present, len(tr), len(te),
                                 classification_metrics(yte, prob)))
            per_mod = {m: _clf_proba(z[m][tr], ytr, z[m][te]) for m in mods}
            for agg in (aggregations or AGGREGATIONS):
                combined = AGGREGATORS[agg](per_mod)
                rows.append(_row(task, "aggregation", agg, present, len(tr), len(te),
                                 classification_metrics(yte, combined[:, 1])))

        elif kind == "forecast":
            ytr, yte = y[tr].astype(float), y[te].astype(float)
            for m in mods:
                pred_te, pred_tr = _ridge(z[m][tr], ytr, z[m][te], z[m][tr])
                rows.append(_row(task, "single", m, present, len(tr), len(te),
                                 forecast_metrics(yte, pred_te, ytr, pred_tr)))
            for s, h in strat_h.items():
                pred_te, pred_tr = _ridge(h[tr], ytr, h[te], h[tr])
                rows.append(_row(task, "fusion", s, present, len(tr), len(te),
                                 forecast_metrics(yte, pred_te, ytr, pred_tr)))
            preds = [_ridge(z[m][tr], ytr, z[m][te], z[m][tr])[0] for m in mods]
            pred_te = np.mean(preds, axis=0)
            pred_tr = np.mean([_ridge(z[m][tr], ytr, z[m][tr], z[m][tr])[0]
                               for m in mods], axis=0)
            rows.append(_row(task, "aggregation", "ensemble_avg", present,
                             len(tr), len(te),
                             forecast_metrics(yte, pred_te, ytr, pred_tr)))

    return pd.DataFrame(rows)


def ablation_summary(df: pd.DataFrame) -> str:
    """Markdown: fused vs best-single per task (honest — no winner is assumed)."""
    lines = ["# CACMF Ablation Summary\n",
             "Fused vs single-modality vs aggregation on subject-held-out splits. "
             "No configuration is assumed to win — these are measured results.\n"]
    for task, g in df.groupby("task"):
        lines.append(f"\n## {task}\n")
        cls = "auroc" in g and g["auroc"].notna().any()
        metric = "auroc" if cls else "mae"
        better = "higher" if cls else "lower"
        cols = ["config_type", "config_name"]
        for c in ([metric, "f1", "accuracy", "ece"] if cls
                  else [metric, "coverage", "interval_radius"]):
            if c not in cols:
                cols.append(c)
        sub = g[cols].copy()
        try:
            lines.append(sub.to_markdown(index=False))
        except Exception:
            lines.append("```\n" + sub.to_string(index=False) + "\n```")
        lines.append(f"\n_(primary metric: {metric}, {better} is better)_\n")
    return "\n".join(lines)
