"""dvxr.eval.glucose_ablation — the honest, single-cohort glucose-excursion ablation (Gate 8).

The scientific question the product must answer is prospective: on held-out participants, does adding a
modality improve 30/60-minute glucose-excursion detection over CGM alone? On the real CGMacros cohort we
can honestly evaluate the arms that CGMacros co-registers on the SAME subjects — CGM-only and
CGM+wearable (Fitbit HR/METs). Every arm that needs EEG (cgm+eeg, the full fused product) is
**not evaluable** — no cohort co-registers EEG+CGM — and is reported as ``cannot_evaluate``, never a
number. This is the release gate: the fused headline stays gated until synchronized pilot data exists.

Design (spec §9): subject-held-out outer split, a SEPARATE subject-held-out calibration fold (Platt),
identical target/threshold across arms, multiple seeds, and a paired bootstrap CI over test rows for the
CGM+wearable − CGM-only AUROC delta.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dvxr.calibration import fit_platt_calibrator
from dvxr.eval.clinical_metrics import (
    auprc,
    brier_score,
    expected_calibration_error,
    threshold_at_fixed_false_alert_rate,
)
from dvxr.eval.splits import subject_holdout_split
from dvxr.prediction.service import cgm_history_features
from dvxr.targets import ExcursionThresholds, build_excursion_labels

#: honestly-evaluable arms on CGMacros (same-subject synchrony for these modalities) → channel set.
HONEST_ARMS: Dict[str, Tuple[str, ...]] = {
    "cgm_only": ("glucose",),
    "cgm_wearable": ("glucose", "hr", "mets"),
}
#: arms that require EEG — no public cohort co-registers EEG+CGM, so they cannot be evaluated here.
GATED_ARMS: Dict[str, str] = {
    "cgm_eeg": "no public cohort co-registers EEG with CGM on the same subjects (spec §1.B)",
    "fused_eeg_cgm_wearable": "the fused product requires synchronized same-subject EEG+wearable+CGM "
                              "pilot data, which does not exist in this deployment",
}


def load_cgmacros(root: Optional[str] = None, max_subjects: Optional[int] = None) -> pd.DataFrame:
    """Load CGMacros into (subject_id, timestamp, glucose, hr, mets). Glucose=Libre GL (mg/dL),
    wearable = Fitbit HR + METs — all same-subject. Returns empty frame if the data is absent."""
    root = root or "data/real/cgmacros"
    files = sorted(glob.glob(os.path.join(root, "**", "CGMacros-*", "CGMacros-*.csv"), recursive=True))
    files = [f for f in files if os.path.basename(f).startswith("CGMacros-")]
    frames = []
    for f in files:
        sid = os.path.basename(f).replace("CGMacros-", "").replace(".csv", "")
        try:
            d = pd.read_csv(f, usecols=["Timestamp", "Libre GL", "HR", "METs"])
        except (ValueError, FileNotFoundError):
            continue
        d = d.rename(columns={"Timestamp": "timestamp", "Libre GL": "glucose",
                              "HR": "hr", "METs": "mets"})
        d["subject_id"] = f"cgm{sid}"
        frames.append(d)
        if max_subjects and len(frames) >= max_subjects:
            break
    if not frames:
        return pd.DataFrame(columns=["subject_id", "timestamp", "glucose", "hr", "mets"])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["timestamp", "glucose"])


def _wearable_features(hist: pd.DataFrame, channels: Sequence[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for ch in channels:
        if ch == "glucose":
            continue
        v = pd.to_numeric(hist.get(ch), errors="coerce").to_numpy(dtype=float) if ch in hist else np.array([])
        v = v[~np.isnan(v)]
        feats[f"{ch}_mean"] = float(np.mean(v)) if len(v) else 0.0
        feats[f"{ch}_std"] = float(np.std(v)) if len(v) else 0.0
        feats[f"{ch}_max"] = float(np.max(v)) if len(v) else 0.0
        feats[f"{ch}__present"] = 1.0 if len(v) else 0.0        # missing != zero
    return feats


def build_arm_matrix(cgm: pd.DataFrame, examples: pd.DataFrame, channels: Sequence[str], *,
                     thresholds: ExcursionThresholds):
    """Feature matrix for one arm: CGM history features (+ wearable summaries when the arm includes
    them). Returns (X, y, subject_ids, keys, feature_names) over the reportable (uncensored) examples,
    where ``keys[i] = "subject|anchor_iso|horizon"`` is the STABLE example id used to pair arms exactly
    (never by positional truncation)."""
    rep = examples[examples["censored"] == False]              # noqa: E712
    rows, ys, subs, keys = [], [], [], []
    names: Optional[List[str]] = None
    # index CGM by subject once (avoids re-scanning/re-sorting the whole cohort per anchor)
    by_subject = {sid: g.sort_values("timestamp") for sid, g in cgm.groupby("subject_id")}
    hist_min = pd.Timedelta(minutes=thresholds.history_minutes)
    for _, ex in rep.iterrows():
        sid = str(ex["subject_id"])
        anchor = pd.Timestamp(ex["anchor_time"])
        g = by_subject.get(sid)
        if g is None:
            continue
        hist = g[(g.timestamp >= anchor - hist_min) & (g.timestamp <= anchor)]  # causal multi-channel
        if len(hist) == 0:
            continue
        feats = dict(cgm_history_features(hist[["timestamp", "glucose"]], thresholds=thresholds))
        feats.update(_wearable_features(hist, channels))
        if any(np.isnan(v) for k, v in feats.items() if k.startswith("cgm_")):
            continue
        if names is None:
            names = sorted(feats.keys())
        rows.append([feats[k] for k in names])
        ys.append(int(ex["label"]))
        subs.append(sid)
        keys.append(f"{sid}|{anchor.isoformat()}|{int(ex['horizon_minutes'])}")
    if not rows:
        return np.empty((0, 0)), np.array([]), np.array([]), np.array([]), names or []
    return (np.array(rows, dtype=float), np.array(ys, dtype=int), np.array(subs),
            np.array(keys, dtype=object), names)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


@dataclass
class ArmResult:
    arm: str
    n_test: int
    auroc: float
    sensitivity_at_far10: float
    false_alerts_per_day: float
    brier: float


@dataclass
class AblationReport:
    honest: Dict[str, dict] = field(default_factory=dict)
    gated: Dict[str, dict] = field(default_factory=dict)
    paired_delta: Optional[dict] = None
    n_subjects: int = 0
    threshold_version: str = ""


def _subject_folds(subjects: Sequence[str], n_folds: int, seed: int) -> Dict[str, int]:
    """Assign every unique subject to one of ``n_folds`` folds (shared across arms so both arms train/
    test on the SAME participant partition — the precondition for exact-key pairing)."""
    uniq = sorted(set(str(s) for s in subjects))
    rng = np.random.default_rng(seed)
    order = np.array(uniq, dtype=object)
    rng.shuffle(order)
    return {s: (i % n_folds) for i, s in enumerate(order)}


def _fit_arm_kfold(X, y, subs, keys, fold_of, *, n_folds, cal_frac, seed, far):
    """K-fold over SUBJECTS (disjoint test folds). Per fold: fit on train subjects, Platt-calibrate on a
    held-out CALIBRATION subset of the training subjects, freeze the alert threshold on that calibration
    fold (target FAR), then predict the test fold. Returns per-example {key: {y, p, subject, threshold}}
    — each participant appears once across the pooled test folds."""
    from sklearn.ensemble import GradientBoostingClassifier
    subs = np.array([str(s) for s in subs])
    out: Dict[str, dict] = {}
    for k in range(n_folds):
        te = np.array([i for i in range(len(subs)) if fold_of.get(subs[i]) == k])
        tr_pool = np.array([i for i in range(len(subs)) if fold_of.get(subs[i]) != k])
        if len(te) == 0 or len(tr_pool) == 0:
            continue
        tr_i, cal_i = subject_holdout_split(subs[tr_pool], test_frac=cal_frac, seed=seed + k)
        tr, cal = tr_pool[tr_i], tr_pool[cal_i]
        if min(len(np.unique(y[tr])), len(np.unique(y[cal])), len(np.unique(y[te]))) < 2:
            continue
        clf = GradientBoostingClassifier(random_state=seed)
        clf.fit(X[tr], y[tr])
        calib = fit_platt_calibrator(clf.predict_proba(X[cal])[:, 1], y[cal])
        p_cal = calib.predict(clf.predict_proba(X[cal])[:, 1])
        # freeze the operating threshold on the CALIBRATION fold — never on the test predictions
        thr = threshold_at_fixed_false_alert_rate(y[cal], p_cal, target_far=far)
        p_te = calib.predict(clf.predict_proba(X[te])[:, 1])
        for j, i in enumerate(te):
            out[str(keys[i])] = {"y": int(y[i]), "p": float(p_te[j]),
                                 "subject": subs[i], "threshold": float(thr)}
    return out


def _person_days(cgm: pd.DataFrame, subjects: Sequence[str]) -> float:
    """Actual observed participant-time (days) for ``subjects`` — the honest denominator for
    false-alerts-per-participant-day (never a placeholder 1.0)."""
    total = 0.0
    want = set(str(s) for s in subjects)
    for sid, g in cgm.groupby("subject_id"):
        if str(sid) not in want:
            continue
        t = pd.to_datetime(g["timestamp"]).dropna()
        if len(t) >= 2:
            total += (t.max() - t.min()) / pd.Timedelta(days=1)
    return float(total)


def _horizon_of(key: str) -> Optional[int]:
    """Recover the horizon from a pooled example key ``subject|anchor_iso|horizon``."""
    try:
        return int(str(key).rsplit("|", 1)[1])
    except (IndexError, ValueError):
        return None


def _metrics_over(items: Sequence[dict], cgm: pd.DataFrame, arm: str) -> dict:
    y = np.array([r["y"] for r in items], dtype=int)
    p = np.array([r["p"] for r in items], dtype=float)
    alert = np.array([1 if r["p"] >= r["threshold"] else 0 for r in items], dtype=int)
    subjects = {r["subject"] for r in items}
    pdays = _person_days(cgm, subjects)
    false_alerts = int(((alert == 1) & (y == 0)).sum())
    sens = float(alert[y == 1].mean()) if (y == 1).any() else float("nan")
    return {
        "n_test": int(len(y)), "n_subjects_test": int(len(subjects)),
        "auroc": round(_auroc(y, p), 4),
        "auprc": round(auprc(y, p), 4),
        "sensitivity_at_frozen_threshold": round(sens, 4),
        "false_alerts_per_participant_day": round(false_alerts / pdays, 4) if pdays > 0 else None,
        "brier": round(brier_score(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "modality_scope": arm,
    }


def _arm_metrics(pooled: Dict[str, dict], cgm: pd.DataFrame, arm: str) -> dict:
    """Pooled arm metrics PLUS a per-horizon breakdown. Pooling 30- and 60-minute examples into one
    number hides that they are two different conditional tasks (P0-4), so each horizon is reported
    separately alongside the pooled figures."""
    out = _metrics_over(list(pooled.values()), cgm, arm)
    by_h: Dict[int, list] = {}
    for k, r in pooled.items():
        h = _horizon_of(k)
        if h is not None:
            by_h.setdefault(h, []).append(r)
    out["per_horizon"] = {int(h): _metrics_over(items, cgm, arm) for h, items in sorted(by_h.items())}
    return out


def run_glucose_ablation(cgm: pd.DataFrame, *, thresholds: ExcursionThresholds = ExcursionThresholds(),
                         seeds: Sequence[int] = (1, 2, 3), test_frac: float = 0.3,
                         cal_frac: float = 0.25, anchor_stride: int = 8,
                         max_anchors_per_subject: int = 60, n_folds: int = 5,
                         far: float = 0.1, participant_days: float = 1.0) -> AblationReport:
    """Honest CGM-only vs CGM+wearable ablation on ``cgm``; every EEG/fused arm is gated.

    Methodology (corrected): subject K-fold with DISJOINT test folds (each participant once), a
    calibration subset carved from each fold's training subjects, the alert threshold FROZEN on the
    calibration fold, arms paired by EXACT example key, and a PARTICIPANT-level bootstrap for the delta
    CI over real observed person-time. ``seeds[0]`` seeds the fold shuffle (no cross-seed row pooling)."""
    rep = AblationReport(n_subjects=int(cgm["subject_id"].nunique()),
                         threshold_version=thresholds.version)
    for arm, reason in GATED_ARMS.items():
        rep.gated[arm] = {"status": "cannot_evaluate", "reason": reason}

    anchors = []
    for sid, g in cgm.groupby("subject_id"):
        t = pd.to_datetime(g["timestamp"]).sort_values()
        anchors += list(t.iloc[thresholds.history_minutes // 5::anchor_stride])[:max_anchors_per_subject]
    examples = build_excursion_labels(cgm, thresholds=thresholds, anchors=sorted(set(anchors)),
                                      subject_col="subject_id")

    n_folds = max(2, min(n_folds, int(cgm["subject_id"].nunique())))
    fold_of = _subject_folds(cgm["subject_id"].tolist(), n_folds, seed=int(seeds[0]))

    per_arm: Dict[str, Dict[str, dict]] = {}
    for arm, channels in HONEST_ARMS.items():
        X, y, subs, keys, _names = build_arm_matrix(cgm, examples, channels, thresholds=thresholds)
        if len(y) < 20 or len(np.unique(y)) < 2:
            rep.honest[arm] = {"status": "insufficient_data"}
            continue
        pooled = _fit_arm_kfold(X, y, subs, keys, fold_of, n_folds=n_folds, cal_frac=cal_frac,
                                seed=int(seeds[0]), far=far)
        if not pooled:
            rep.honest[arm] = {"status": "insufficient_data"}
            continue
        per_arm[arm] = pooled
        rep.honest[arm] = _arm_metrics(pooled, cgm, arm)

    if all(a in per_arm for a in HONEST_ARMS):
        rep.paired_delta = _paired_auroc_delta(per_arm["cgm_only"], per_arm["cgm_wearable"])
    return rep


def _paired_auroc_delta(pooled_b: Dict[str, dict], pooled_a: Dict[str, dict],
                        n_boot: int = 500, seed: int = 0) -> dict:
    """AUROC(cgm_wearable) − AUROC(cgm_only) paired by EXACT example key, with a PARTICIPANT-level
    bootstrap CI (resample subjects with replacement — the correlated unit — never individual rows)."""
    keys = sorted(set(pooled_b) & set(pooled_a))          # exact-key pairing, not truncation
    if not keys:
        return {"metric": "auroc(cgm_wearable) - auroc(cgm_only)", "point": float("nan"),
                "ci95": [float("nan"), float("nan")], "adds_value": False, "n_paired": 0}
    y = np.array([pooled_b[k]["y"] for k in keys], dtype=int)
    pb = np.array([pooled_b[k]["p"] for k in keys], dtype=float)
    pa = np.array([pooled_a[k]["p"] for k in keys], dtype=float)
    subj = np.array([pooled_b[k]["subject"] for k in keys], dtype=object)
    # group row indices by subject for the participant bootstrap
    by_subj: Dict[str, list] = {}
    for i, s in enumerate(subj):
        by_subj.setdefault(s, []).append(i)
    subjects = np.array(sorted(by_subj), dtype=object)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(subjects), len(subjects))
        idx = np.concatenate([by_subj[subjects[j]] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(_auroc(y[idx], pa[idx]) - _auroc(y[idx], pb[idx]))
    deltas = np.array([d for d in deltas if not np.isnan(d)])
    lo, hi = ((float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
              if len(deltas) else (float("nan"), float("nan")))
    point = _auroc(y, pa) - _auroc(y, pb)
    return {
        "metric": "auroc(cgm_wearable) - auroc(cgm_only)",
        "point": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_paired": int(len(keys)),
        "bootstrap_unit": "participant",
        "adds_value": bool(lo > 0),               # honest: wearable adds only if the CI clears 0
    }
