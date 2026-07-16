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
    brier_score,
    false_alerts_per_participant_day,
    sensitivity_at_fixed_false_alert_rate,
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
                     thresholds: ExcursionThresholds) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Feature matrix for one arm: CGM history features (+ wearable summaries when the arm includes
    them). Returns (X, y, subject_ids, feature_names) over the reportable (uncensored) examples."""
    rep = examples[examples["censored"] == False]              # noqa: E712
    rows, ys, subs = [], [], []
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
    if not rows:
        return np.empty((0, 0)), np.array([]), np.array([]), names or []
    return np.array(rows, dtype=float), np.array(ys, dtype=int), np.array(subs), names


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


def _fit_predict_arm(X, y, subs, seed, test_frac, cal_frac):
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    from sklearn.ensemble import GradientBoostingClassifier
    tr_all, te = subject_holdout_split(subs, test_frac=test_frac, seed=seed)
    # a SEPARATE subject-held-out calibration fold carved from the training subjects (never the test)
    tr, cal = subject_holdout_split(subs[tr_all], test_frac=cal_frac, seed=seed + 100)
    tr, cal = tr_all[tr], tr_all[cal]
    if min(len(np.unique(y[tr])), len(np.unique(y[cal])), len(np.unique(y[te]))) < 2:
        return None
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X[tr], y[tr])
    calib = fit_platt_calibrator(clf.predict_proba(X[cal])[:, 1], y[cal])
    p_te = calib.predict(clf.predict_proba(X[te])[:, 1])
    return {"subjects": subs[te], "y": y[te], "p": np.asarray(p_te)}


def run_glucose_ablation(cgm: pd.DataFrame, *, thresholds: ExcursionThresholds = ExcursionThresholds(),
                         seeds: Sequence[int] = (1, 2, 3), test_frac: float = 0.3,
                         cal_frac: float = 0.25, anchor_stride: int = 8,
                         max_anchors_per_subject: int = 60,
                         participant_days: float = 1.0) -> AblationReport:
    """Run the honest CGM-only vs CGM+wearable ablation on ``cgm``; gate every EEG/fused arm."""
    rep = AblationReport(n_subjects=int(cgm["subject_id"].nunique()),
                         threshold_version=thresholds.version)
    for arm, reason in GATED_ARMS.items():
        rep.gated[arm] = {"status": "cannot_evaluate", "reason": reason}

    # anchors: subsample per subject for tractability (deterministic)
    anchors = []
    for sid, g in cgm.groupby("subject_id"):
        t = pd.to_datetime(g["timestamp"]).sort_values()
        anchors += list(t.iloc[thresholds.history_minutes // 5::anchor_stride])[:max_anchors_per_subject]
    examples = build_excursion_labels(cgm, thresholds=thresholds, anchors=sorted(set(anchors)),
                                      subject_col="subject_id")

    pooled: Dict[str, Dict[str, list]] = {a: {"y": [], "p": []} for a in HONEST_ARMS}
    for arm, channels in HONEST_ARMS.items():
        X, y, subs, _names = build_arm_matrix(cgm, examples, channels, thresholds=thresholds)  # once
        for seed in seeds:
            out = _fit_predict_arm(X, y, subs, seed, test_frac, cal_frac)
            if out is None:
                continue
            pooled[arm]["y"].append(out["y"])
            pooled[arm]["p"].append(out["p"])

    for arm in HONEST_ARMS:
        if not pooled[arm]["y"]:
            rep.honest[arm] = {"status": "insufficient_data"}
            continue
        y = np.concatenate(pooled[arm]["y"]); p = np.concatenate(pooled[arm]["p"])
        sfar = sensitivity_at_fixed_false_alert_rate(y, p, target_far=0.1)
        rep.honest[arm] = {
            "n_test": int(len(y)),
            "auroc": round(_auroc(y, p), 4),
            "sensitivity_at_far10": round(sfar["sensitivity"], 4),
            "false_alerts_per_participant_day": round(
                false_alerts_per_participant_day(y, p, sfar["threshold"], participant_days), 4),
            "brier": round(brier_score(y, p), 4),
            "modality_scope": arm,
        }

    # paired delta (cgm_wearable - cgm_only) with a bootstrap CI over test rows, when both ran
    if all(pooled[a]["y"] for a in HONEST_ARMS):
        rep.paired_delta = _paired_auroc_delta(pooled)
    return rep


def _paired_auroc_delta(pooled, n_boot: int = 200, seed: int = 0) -> dict:
    yb = np.concatenate(pooled["cgm_only"]["y"]); pb = np.concatenate(pooled["cgm_only"]["p"])
    ya = np.concatenate(pooled["cgm_wearable"]["y"]); pa = np.concatenate(pooled["cgm_wearable"]["p"])
    n = min(len(yb), len(ya))
    yb, pb, pa = yb[:n], pb[:n], pa[:n]           # aligned pooled rows (same target)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(yb[idx])) < 2:
            continue
        deltas.append(_auroc(yb[idx], pa[idx]) - _auroc(yb[idx], pb[idx]))
    deltas = np.array([d for d in deltas if not np.isnan(d)])
    lo, hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))) if len(deltas) else (float("nan"), float("nan"))
    point = _auroc(ya, pa) - _auroc(yb, pb)
    return {
        "metric": "auroc(cgm_wearable) - auroc(cgm_only)",
        "point": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "adds_value": bool(lo > 0),               # honest: wearable adds only if the CI clears 0
    }
