"""dvxr.eval.clinical_metrics — the clinically-relevant evaluation metrics (spec §9).

AUROC/AUPRC measure ranking; a glucose early-warning product also has to answer "how many false alerts
per participant-day?", "what sensitivity at a prespecified false-alert rate?", "how early are events
detected?". These are additive, pure-NumPy, and deterministic; they compose with the existing
`dvxr.eval.metrics`. A chronological personalization split enforces the causal baseline→evaluation
ordering (spec §7) for within-person evaluation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def brier_score(y_true: Sequence[int], prob: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def rmse(y_true: Sequence[float], pred: Sequence[float]) -> float:
    y, p = np.asarray(y_true, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


def mae(y_true: Sequence[float], pred: Sequence[float]) -> float:
    y, p = np.asarray(y_true, float), np.asarray(pred, float)
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


def bias(y_true: Sequence[float], pred: Sequence[float]) -> float:
    y, p = np.asarray(y_true, float), np.asarray(pred, float)
    return float(np.mean(p - y)) if len(y) else float("nan")


def threshold_at_fixed_false_alert_rate(y_true: Sequence[int], prob: Sequence[float],
                                        target_far: float) -> float:
    """The lowest probability threshold whose false-alert rate (FPR on negatives) is <= target_far.

    Lower threshold ⇒ more alerts ⇒ higher FPR and higher sensitivity, so we want the smallest
    threshold that still respects the false-alert budget."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    neg = p[y == 0]
    if len(neg) == 0:
        return 1.0
    # candidate thresholds = every observed score (ascending) + 1.0; an alarm fires when p >= t.
    # Return the SMALLEST threshold (most sensitive) whose false-alert rate meets the budget. Scores
    # above max(neg) are needed to reach FAR 0, so all scores — not just negatives — are candidates.
    for t in np.sort(np.unique(np.concatenate([p, [1.0]]))):
        far = float(np.mean(neg >= t))
        if far <= target_far:
            return float(t)
    return 1.0


def sensitivity_at_fixed_false_alert_rate(y_true: Sequence[int], prob: Sequence[float],
                                          target_far: float = 0.1) -> Dict[str, float]:
    """Sensitivity (TPR) at the threshold meeting a prespecified false-alert rate (spec §9)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    t = threshold_at_fixed_false_alert_rate(y, p, target_far)
    pos = p[y == 1]
    neg = p[y == 0]
    sens = float(np.mean(pos >= t)) if len(pos) else float("nan")
    far = float(np.mean(neg >= t)) if len(neg) else float("nan")
    return {"threshold": t, "sensitivity": sens, "false_alert_rate": far, "target_far": target_far}


def false_alerts_per_participant_day(y_true: Sequence[int], prob: Sequence[float],
                                     threshold: float, participant_days: float) -> float:
    """False positives divided by total participant-days monitored (alert-fatigue metric, spec §9)."""
    if participant_days <= 0:
        return float("nan")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    fp = int(np.sum((p >= threshold) & (y == 0)))
    return fp / float(participant_days)


def event_lead_times(event_times: Sequence[float], alarm_times: Sequence[float],
                     horizon: float) -> List[float]:
    """For each event, the lead time = event_time − earliest alarm within ``horizon`` before it.

    Times are numeric (e.g. minutes). An event with no preceding alarm inside the horizon is a MISS
    (excluded from the lead-time list; count misses separately)."""
    alarms = np.sort(np.asarray(alarm_times, dtype=float))
    leads: List[float] = []
    for et in np.asarray(event_times, dtype=float):
        window = alarms[(alarms >= et - horizon) & (alarms <= et)]
        if len(window):
            leads.append(float(et - window[0]))
    return leads


def median_event_lead_time(event_times: Sequence[float], alarm_times: Sequence[float],
                           horizon: float) -> float:
    leads = event_lead_times(event_times, alarm_times, horizon)
    return float(np.median(leads)) if leads else float("nan")


def fraction_events_detected_early(event_times: Sequence[float], alarm_times: Sequence[float],
                                   horizon: float, min_lead: float = 15.0) -> float:
    """Fraction of events with an alarm at least ``min_lead`` before the event (spec §9)."""
    events = np.asarray(event_times, dtype=float)
    if len(events) == 0:
        return float("nan")
    leads = event_lead_times(events, alarm_times, horizon)
    early = sum(1 for l in leads if l >= min_lead)
    return early / float(len(events))


def chronological_personalization_split(
    subject_ids: Sequence,
    timestamps: Sequence,
    baseline_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-subject causal split: each subject's EARLIEST ``baseline_frac`` of rows (by time) is the
    baseline/calibration set; the later rows are evaluation (spec §7). Returns (baseline_idx, eval_idx)
    into the input order. A subject's future never appears in its own baseline."""
    sids = np.asarray(subject_ids)
    ts = np.asarray(timestamps)
    order = np.arange(len(sids))
    base, evl = [], []
    for s in np.unique(sids):
        idx = order[sids == s]
        idx = idx[np.argsort(ts[idx], kind="stable")]
        cut = max(1, int(round(len(idx) * baseline_frac)))
        base.extend(idx[:cut].tolist())
        evl.extend(idx[cut:].tolist())
    return np.array(sorted(base), dtype=int), np.array(sorted(evl), dtype=int)
