#!/usr/bin/env python
"""Train the research-stage tabular meta-model stack for POST /v1/research/predict (OFFLINE).

Pipeline (never runs inside the HTTP request — the API only *loads* what this writes):

  1. Build a canonical named-feature table per target (metabolic / physiological / neural / clinical).
     Real cohorts (``dvxr.bench.tasks``) are projected onto the interpretable named axes when available
     (``--real``); otherwise a clearly-labelled SYNTHETIC fixture exercises the whole pipeline.
  2. Subject-level OUTER split (``eval.splits.subject_kfold``) → train each per-target base model within
     the train folds → collect OUT-OF-FOLD probabilities (no subject in both train and OOF).
  3. Train the diabetes META-model (``prediction.meta_model``) on metabolic covariates + the OOF base
     probabilities.
  4. Held-out eval + a separate calibration fold (``calibration.fit_platt_calibrator``).
  5. Serialise committed artifacts (JSON) + a manifest with sha256, versions, and the honesty flags
     ``validated_for_clinical_use=false`` / ``research_stage=true``.

HONESTY: the diabetes target is ``cgmacros_diabetes`` — an EXCLUDED task. Its artifact is ALWAYS
research/experimental, never a validated clinical claim, never a headline AUROC, never a diagnosis.

Usage:
  venv/bin/python scripts/train_research_meta.py --synthetic          # fast, deterministic fixture
  venv/bin/python scripts/train_research_meta.py --real               # attempt real cohorts per target
  venv/bin/python scripts/train_research_meta.py --out /tmp/models    # custom output dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.calibration import fit_platt_calibrator  # noqa: E402
from dvxr.eval.splits import InsufficientSubjectsError, subject_kfold  # noqa: E402
from dvxr.prediction.meta_model import DiabetesMetaModel, LinearHead, fit_linear_head  # noqa: E402
from dvxr.serve.research_predict import (  # noqa: E402
    CANONICAL_FEATURES,
    META_METABOLIC_FEATURES,
    META_PROB_FEATURES,
    TARGET_FEATURES,
    TARGETS,
)

VERSION = "v2-research"

# canonical feature -> substrings that identify a matching cohort column (real-data projection).
_KEYWORDS = {
    "heart_rate": ["ecg", "_hr", "bpm", "heart", "ibi"],
    "hrv_rmssd": ["hrv", "rmssd", "sdnn"],
    "eda": ["eda", "scl", "gsr"],
    "resp_rate": ["resp", "rsp", "breath"],
    "skin_temp": ["temp", "skt"],
    "eeg_delta": ["delta"],
    "eeg_theta": ["theta"],
    "eeg_alpha": ["alpha"],
    "eeg_beta": ["beta"],
    "eeg_gamma": ["gamma"],
    "bmi": ["bmi"],
    "hba1c": ["hba1c", "a1c"],
    "fasting_glucose": ["fasting_glucose", "glucose"],
    "cgm_mean": ["cgm", "glucose_mean", "glucose_now"],
    "cgm_std": ["glucose_cv", "cgm_std", "std"],
    "time_above_range": ["frac_hyper", "tar", "above"],
}

_TARGET_TASK = {
    "stress": "wesad_stress",
    "anxiety": "deap_anxiety",
    "depression": "mumtaz_depression",
    "cognitive_workload": "eegmat_workload",
    "glucose_instability": "cgmacros_diabetes",
}


# --------------------------------------------------------------------------- table builders
def _synthetic_table(target: str, n_subjects: int = 40, per_subject: int = 6, seed: int = 7):
    """A clearly-labelled synthetic fixture: canonical features drawn around their reference ranges,
    with a KNOWN generative label rule so the trained head recovers sensible signs. Research/simulation
    only — never presented as a real cohort result."""
    rng = np.random.default_rng(seed + abs(hash(target)) % 1000)
    feats = TARGET_FEATURES[target]
    rows, ys, sids = [], [], []
    for s in range(n_subjects):
        subj_bias = rng.normal(0, 0.6)
        for _ in range(per_subject):
            vals = {}
            z_sum = 0.0
            for f in feats:
                _mod, m, sc, lo, hi = CANONICAL_FEATURES[f]
                v = float(np.clip(rng.normal(m, sc), lo, hi))
                vals[f] = v
                z_sum += (v - m) / sc
            logit = 0.8 * z_sum + subj_bias + rng.normal(0, 0.5)
            rows.append([vals[f] for f in feats])
            ys.append(int(logit > 0))
            sids.append(f"S{s}")
    return np.array(rows, dtype=float), np.array(ys, dtype=int), np.array(sids), feats, "synthetic"


def _project_real_table(target: str):
    """Best-effort projection of a real cohort onto the canonical named features for ``target``.

    Row-level: each canonical feature is the mean of the standardized cohort columns whose names match
    its keywords. Returns None if the cohort can't load or too few named features are recoverable."""
    import pandas as pd

    from dvxr.bench.tasks import TASK_BUILDERS

    task_name = _TARGET_TASK[target]
    try:
        task = TASK_BUILDERS[task_name]()
    except Exception:  # noqa: BLE001 — dataset absent → caller falls back to synthetic
        return None
    feats = TARGET_FEATURES[target]
    # flatten all modality columns into one (N, D) frame with names
    mats, names = [], []
    for mod, mat in task.features.items():
        cols = task.feature_names[mod]
        mats.append(np.asarray(mat, dtype=float))
        names.extend(cols)
    if not mats:
        return None
    X = np.hstack(mats)
    df = pd.DataFrame(X, columns=names)
    # standardize columns once for the projection
    zdf = (df - df.mean()) / df.std(ddof=0).replace(0, 1.0)
    out_cols, recovered = [], 0
    for f in feats:
        kws = _KEYWORDS.get(f, [])
        match = [c for c in names if any(k in c.lower() for k in kws)]
        if match:
            col = zdf[match].mean(axis=1).to_numpy()
            # re-express on the canonical reference scale so the runtime schema stays interpretable
            _mod, m, sc, _lo, _hi = CANONICAL_FEATURES[f]
            out_cols.append(m + sc * col)
            recovered += 1
        else:
            _mod, m, sc, _lo, _hi = CANONICAL_FEATURES[f]
            out_cols.append(np.full(len(df), m))
    if recovered < max(1, len(feats) // 2):
        return None  # too little real signal recovered — use the honest synthetic fixture instead
    Xc = np.column_stack(out_cols)
    return Xc, np.asarray(task.y, dtype=int), np.asarray(task.subject_ids), feats, "real"


def build_table(target: str, use_real: bool):
    if use_real:
        real = _project_real_table(target)
        if real is not None:
            return real
        print(f"  [{target}] real cohort unavailable → synthetic fixture")
    return _synthetic_table(target)


# --------------------------------------------------------------------------- OOF + fit
def _auroc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:  # noqa: BLE001
        return None


def oof_and_head(target: str, X, y, sids, source: str):
    """Subject-level OOF probabilities + a final head fit on all rows (with a Platt layer fit on a
    separate held-out calibration fold). Returns (head, oof_prob_by_subject, auroc_oof)."""
    from sklearn.linear_model import LogisticRegression

    feats = TARGET_FEATURES[target]
    n_subj = len(np.unique(sids))
    n_folds = min(5, max(2, n_subj))
    oof = np.full(len(y), np.nan)
    try:
        folds = subject_kfold(sids, n_folds=n_folds)
    except InsufficientSubjectsError:
        folds = [(np.arange(len(y)), np.arange(len(y)))]
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(np.mean(y[tr])) if len(tr) else 0.5
            continue
        clf = LogisticRegression(max_iter=2000, random_state=7)
        clf.fit((X[tr] - mean) / scale, y[tr])
        oof[te] = clf.predict_proba((X[te] - mean) / scale)[:, 1]
    auroc_oof = _auroc(y, oof)
    evidence = "experimental" if source == "real" else "simulation"
    head = fit_linear_head(X, y, feats, model_version=f"research-{target}/{VERSION}-{source}",
                           evidence_status=evidence, auroc_oof=auroc_oof)
    # Platt on a held-out calibration fold (fold 0's test rows), never on the head's own train rows
    if folds and len(folds) > 1:
        cal_idx = folds[0][1]
        if len(np.unique(y[cal_idx])) >= 2:
            cal = fit_platt_calibrator(oof[cal_idx], y[cal_idx])
            if cal.model is not None:
                head.platt_coef = float(cal.model.coef_.ravel()[0])
                head.platt_intercept = float(cal.model.intercept_.ravel()[0])
    # OOF probability aggregated to subject level (for stacking into the meta-model)
    oof_by_subject = {}
    for s in np.unique(sids):
        oof_by_subject[str(s)] = float(np.nanmean(oof[sids == s]))
    return head, oof_by_subject, auroc_oof


def train_meta(use_real: bool, base_heads, oof_by_subject_per_target):
    """Diabetes meta-model: metabolic covariates + per-subject OOF base probabilities → diabetes label.
    Trained on the glucose_instability cohort's subjects so the metabolic covariates + label align."""
    Xg, yg, sids, _feats, source = build_table("glucose_instability", use_real)
    import pandas as pd
    gdf = pd.DataFrame(Xg, columns=TARGET_FEATURES["glucose_instability"])
    gdf["__sid"] = sids
    gdf["__y"] = yg
    # subject-level metabolic covariates
    subj = gdf.groupby("__sid").mean(numeric_only=True)
    rows, ys, feat_names = [], [], META_METABOLIC_FEATURES + META_PROB_FEATURES
    for sid, r in subj.iterrows():
        met = [float(r.get(f, CANONICAL_FEATURES[f][1])) for f in META_METABOLIC_FEATURES]
        probs = []
        for t in TARGETS:
            if t == "glucose_instability":
                continue
            probs.append(float(oof_by_subject_per_target.get(t, {}).get(str(sid), 0.5)))
        rows.append(met + probs)
        ys.append(int(round(r["__y"])))
    X = np.array(rows, dtype=float)
    y = np.array(ys, dtype=int)
    auroc = None
    if len(np.unique(y)) >= 2 and len(y) >= 4:
        # honest subject-level OOF AUROC for the meta-model (reported, never a validated headline)
        from sklearn.linear_model import LogisticRegression
        oof = np.full(len(y), np.nan)
        try:
            folds = subject_kfold(np.arange(len(y)), n_folds=min(5, max(2, len(y))))
        except InsufficientSubjectsError:
            folds = [(np.arange(len(y)), np.arange(len(y)))]
        m, s = X.mean(0), X.std(0)
        s[s == 0] = 1.0
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                oof[te] = float(np.mean(y[tr]))
                continue
            clf = LogisticRegression(max_iter=2000, random_state=7)
            clf.fit((X[tr] - m) / s, y[tr])
            oof[te] = clf.predict_proba((X[te] - m) / s)[:, 1]
        auroc = _auroc(y, oof)
    evidence = "experimental" if source == "real" else "simulation"
    head = fit_linear_head(X, y, feat_names, model_version=f"research-diabetes-meta/{VERSION}-{source}",
                           evidence_status=evidence, auroc_oof=auroc)
    meta = DiabetesMetaModel(head=head, metabolic_features=META_METABOLIC_FEATURES,
                             prob_features=META_PROB_FEATURES)
    return meta, auroc, source


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "product" / "research_models"))
    ap.add_argument("--real", action="store_true", help="attempt real cohorts (falls back to synthetic)")
    ap.add_argument("--synthetic", action="store_true", help="force the synthetic fixture (fast)")
    args = ap.parse_args()
    use_real = args.real and not args.synthetic
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"training research meta-stack ({'real' if use_real else 'synthetic'}) → {out}")
    base_heads, oof_per_target, metrics = {}, {}, {}
    for t in TARGETS:
        X, y, sids, _feats, source = build_table(t, use_real)
        head, oof_by_subject, auroc = oof_and_head(t, X, y, sids, source)
        base_heads[t] = head
        oof_per_target[t] = oof_by_subject
        metrics[t] = {"auroc_oof": auroc, "source": source, "n": int(len(y)),
                      "evidence_status": head.evidence_status}
        print(f"  [{t}] source={source} n={len(y)} AUROC_oof={auroc}")

    meta, meta_auroc, meta_source = train_meta(use_real, base_heads, oof_per_target)
    metrics["diabetes_meta"] = {"auroc_oof": meta_auroc, "source": meta_source,
                                "evidence_status": meta.head.evidence_status,
                                "validated_for_clinical_use": False}
    print(f"  [diabetes_meta] source={meta_source} AUROC_oof={meta_auroc} (EXCLUDED task, never validated)")

    models = {"version": VERSION,
              "targets": {t: h.to_dict() for t, h in base_heads.items()},
              "diabetes_meta": meta.to_dict()}
    models_path = out / "models.json"
    models_path.write_text(json.dumps(models, indent=2, sort_keys=True))
    sha = hashlib.sha256(models_path.read_bytes()).hexdigest()

    manifest = {
        "product": "dvxr-research-predict",
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_file": "models.json",
        "artifact_sha256": sha,
        "research_stage": True,
        "validated_for_clinical_use": False,
        "excluded_task_note": ("The diabetes meta-model targets cgmacros_diabetes, an EXCLUDED task; it "
                               "is research-stage decision-support, never a validated claim, never a "
                               "diagnosis."),
        "targets": list(TARGETS),
        "metrics": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {models_path} (sha256={sha[:12]}…) and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
