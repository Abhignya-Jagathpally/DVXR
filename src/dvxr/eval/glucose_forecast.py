"""dvxr.eval.glucose_forecast — CGM-only continuous glucose forecast with a calibrated interval.

The excursion classifier (`dvxr.eval.glucose_ablation`) answers a *binary* question — "will a
threshold-crossing happen in (t, t+h]?". This module answers the complementary *continuous* one: what
will the glucose **level** be at t+h, and with what honestly-calibrated uncertainty? It is deliberately
narrow and honest:

  * **CGM-only.** Features are the causal CGM-history summaries already used by the product
    (`cgm_history_features`); no wearable, no EEG, no fused signal. The forecast is a single-modality
    baseline and is labelled as such — it makes no EEG or fused claim.
  * **Strictly future target.** For an anchor ``t`` and horizon ``h`` the label is the observed glucose
    at the sample closest to ``t+h`` (within tolerance); an anchor whose future is not observed near
    ``t+h`` is *censored*, never imputed. The label sample is always strictly after ``t``.
  * **Split-conformal interval.** The prediction interval is ``yhat ± q`` where ``q`` is the
    finite-sample ``(n+1)`` conformal quantile of the absolute residuals on a **held-out calibration
    fold of subjects** — never read on the test rows. On exchangeable held-out data this guarantees
    marginal coverage ``>= 1 - alpha`` (Vovk et al.; Angelopoulos & Bates 2021).
  * **Subject-held-out evaluation.** Subject K-fold with disjoint test folds; each participant is scored
    once. Coverage and width are reported alongside RMSE/MAE/bias over real held-out participants.

Nothing here fabricates data or crosses cohorts. On a cohort without enough subjects/samples a horizon
is reported ``insufficient_data`` rather than a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dvxr.eval.clinical_metrics import (
    bias,
    interval_coverage,
    mae,
    mean_interval_width,
    rmse,
)
from dvxr.prediction.service import cgm_history_features
from dvxr.targets import ExcursionThresholds

MODALITY_SCOPE = "cgm_only"


def _conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """The finite-sample split-conformal quantile of the absolute residuals.

    With ``n`` calibration residuals, the conformal radius is the ``k``-th smallest residual where
    ``k = ceil((n+1)(1-alpha))`` (the ``(n+1)`` correction is what upgrades an ordinary empirical
    quantile to a *guaranteed* ``>= 1-alpha`` coverage bound). When ``k > n`` the guarantee needs an
    infinite radius, so we return ``inf`` (the honest "cannot certify a finite interval here")."""
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])


def _grouped_conformal_quantile(residuals: Sequence[float], groups: Sequence, alpha: float) -> float:
    """Participant-blocked conformal radius (honest under within-participant correlation).

    Ordinary split conformal assumes the calibration residuals are exchangeable, but many overlapping
    windows from the SAME participant are strongly correlated — treating them as independent
    under-covers on a NEW participant. This reduces each participant to a single conformity score (that
    participant's own ``1-alpha`` residual quantile — a per-person "typical worst case") and takes the
    finite-sample conformal quantile OVER PARTICIPANTS, so the exchangeable unit is the participant. With
    ``m`` participants the radius is the ``ceil((m+1)(1-alpha))``-th smallest per-participant score, or
    ``inf`` when ``m`` is too small to certify a finite interval."""
    res = np.asarray(residuals, dtype=float)
    grp = np.asarray(groups, dtype=object)
    if len(res) == 0:
        return float("inf")
    per_group = []
    for g in np.unique(grp):
        gr = res[grp == g]
        if len(gr):
            per_group.append(float(np.quantile(gr, 1.0 - alpha)))
    return _conformal_quantile(per_group, alpha)


def build_forecast_examples(
    cgm: pd.DataFrame,
    *,
    thresholds: ExcursionThresholds = ExcursionThresholds(),
    anchors: Optional[Sequence[pd.Timestamp]] = None,
    time_col: str = "timestamp",
    glucose_col: str = "glucose",
    subject_col: Optional[str] = None,
) -> pd.DataFrame:
    """Continuous-forecast label table: one row per (subject, anchor, horizon). ``label`` is the glucose
    at the sample closest to ``anchor + horizon`` within ``target_tolerance_minutes``; an anchor whose
    future is not observed near ``t+h`` is ``censored=True`` (``label=NaN``) — never imputed. The label
    sample time (``label_time``) is always strictly after ``anchor``."""
    df = cgm[[c for c in (subject_col, time_col, glucose_col) if c]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[glucose_col] = pd.to_numeric(df[glucose_col], errors="coerce")
    df = df.dropna(subset=[time_col, glucose_col])
    if subject_col is None:
        df["__subject__"] = "single"
        subject_col = "__subject__"
    df = df.sort_values([subject_col, time_col], kind="stable")

    tol = pd.Timedelta(minutes=thresholds.target_tolerance_minutes)
    hist_win = pd.Timedelta(minutes=thresholds.history_minutes)
    rows: List[dict] = []
    for sid, g in df.groupby(subject_col, sort=True):
        tindex = pd.DatetimeIndex(g[time_col].to_numpy())
        vals = g[glucose_col].to_numpy(dtype=float)
        anchor_list = list(tindex) if anchors is None else [pd.Timestamp(a) for a in anchors]
        for anchor in anchor_list:
            anchor = pd.Timestamp(anchor)
            n_hist = int(((tindex >= anchor - hist_win) & (tindex <= anchor)).sum())
            if n_hist < thresholds.min_history_samples:
                continue
            for h in thresholds.horizons_minutes:
                target_t = anchor + pd.Timedelta(minutes=h)
                fut = (tindex > anchor)
                if not fut.any():
                    rows.append(_censored_row(sid, anchor, h, n_hist, thresholds, "no_future_samples"))
                    continue
                fut_t = tindex[fut]
                fut_v = vals[fut.to_numpy() if hasattr(fut, "to_numpy") else fut]
                # sample closest to t+h; censor if none is within tolerance of the horizon
                j = int(np.argmin(np.abs((fut_t - target_t) / pd.Timedelta(minutes=1))))
                if abs(fut_t[j] - target_t) > tol:
                    rows.append(_censored_row(sid, anchor, h, n_hist, thresholds,
                                              "insufficient_future_coverage"))
                    continue
                rows.append({
                    "subject_id": str(sid), "anchor_time": anchor.isoformat(),
                    "horizon_minutes": int(h), "label": float(fut_v[j]),
                    "label_time": pd.Timestamp(fut_t[j]).isoformat(),
                    "censored": False, "censor_reason": None,
                    "n_history_samples": n_hist, "threshold_version": thresholds.version,
                })
    cols = ["subject_id", "anchor_time", "horizon_minutes", "label", "label_time",
            "censored", "censor_reason", "n_history_samples", "threshold_version"]
    return pd.DataFrame(rows, columns=cols)


def _censored_row(sid, anchor, h, n_hist, thr, reason) -> dict:
    return {"subject_id": str(sid), "anchor_time": pd.Timestamp(anchor).isoformat(),
            "horizon_minutes": int(h), "label": float("nan"), "label_time": None,
            "censored": True, "censor_reason": reason, "n_history_samples": n_hist,
            "threshold_version": thr.version}


def build_forecast_matrix(cgm: pd.DataFrame, examples: pd.DataFrame, *,
                          thresholds: ExcursionThresholds):
    """CGM-only feature matrix for the reportable (uncensored) forecast examples. Returns
    ``(X, y, subjects, keys, horizons, names)`` where ``keys[i] = "subject|anchor_iso|horizon"``."""
    rep = examples[examples["censored"] == False]              # noqa: E712
    by_subject = {sid: g.sort_values("timestamp") for sid, g in cgm.groupby("subject_id")}
    hist_min = pd.Timedelta(minutes=thresholds.history_minutes)
    rows, ys, subs, keys, horizons = [], [], [], [], []
    names: Optional[List[str]] = None
    for _, ex in rep.iterrows():
        sid = str(ex["subject_id"])
        anchor = pd.Timestamp(ex["anchor_time"])
        g = by_subject.get(sid)
        if g is None:
            continue
        hist = g[(g.timestamp >= anchor - hist_min) & (g.timestamp <= anchor)]
        if len(hist) == 0:
            continue
        feats = dict(cgm_history_features(hist[["timestamp", "glucose"]], thresholds=thresholds))
        if any(np.isnan(v) for v in feats.values()):
            continue
        if names is None:
            names = sorted(feats.keys())
        rows.append([feats[k] for k in names])
        ys.append(float(ex["label"]))
        subs.append(sid)
        horizons.append(int(ex["horizon_minutes"]))
        keys.append(f"{sid}|{anchor.isoformat()}|{int(ex['horizon_minutes'])}")
    if not rows:
        return (np.empty((0, 0)), np.array([]), np.array([]), np.array([]),
                np.array([]), names or [])
    return (np.array(rows, dtype=float), np.array(ys, dtype=float), np.array(subs, dtype=object),
            np.array(keys, dtype=object), np.array(horizons, dtype=int), names)


@dataclass
class ForecastReport:
    per_horizon: Dict[int, dict] = field(default_factory=dict)
    target_coverage: float = 0.0
    n_subjects: int = 0
    threshold_version: str = ""
    modality_scope: str = MODALITY_SCOPE
    method: str = "split-conformal (CGM-only)"


def _subject_folds(subjects: Sequence[str], n_folds: int, seed: int) -> Dict[str, int]:
    uniq = sorted(set(str(s) for s in subjects))
    rng = np.random.default_rng(seed)
    order = np.array(uniq, dtype=object)
    rng.shuffle(order)
    return {s: (i % n_folds) for i, s in enumerate(order)}


def _forecast_one_horizon(X, y, subs, *, alpha, n_folds, cal_frac, seed, grouped=True):
    """Subject K-fold conformal forecast for one horizon. Per fold: fit a regressor on the train
    subjects, size the conformal radius ``q`` on a held-out CALIBRATION subset of the training subjects,
    then predict the test fold and form ``yhat ± q``. With ``grouped=True`` (default) the radius is
    PARTICIPANT-blocked — the exchangeable unit is the participant, not the correlated overlapping window
    — which is the honest choice under within-participant correlation. Pools one prediction per test
    example. Returns (y_true, y_hat, lower, upper, test_subjects)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from dvxr.eval.splits import subject_holdout_split

    subs = np.array([str(s) for s in subs])
    fold_of = _subject_folds(subs, n_folds, seed)
    yt, yh, lo, hi, ts = [], [], [], [], []
    for k in range(n_folds):
        te = np.array([i for i in range(len(subs)) if fold_of.get(subs[i]) == k])
        tr_pool = np.array([i for i in range(len(subs)) if fold_of.get(subs[i]) != k])
        if len(te) == 0 or len(tr_pool) < 2:
            continue
        tr_i, cal_i = subject_holdout_split(subs[tr_pool], test_frac=cal_frac, seed=seed + k)
        tr, cal = tr_pool[tr_i], tr_pool[cal_i]
        if len(tr) == 0 or len(cal) == 0:
            continue
        reg = GradientBoostingRegressor(random_state=seed)
        reg.fit(X[tr], y[tr])
        resid = np.abs(y[cal] - reg.predict(X[cal]))            # conformity scores on calibration fold
        # radius frozen off the test rows; participant-blocked so correlated windows don't overstate n
        q = (_grouped_conformal_quantile(resid, subs[cal], alpha) if grouped
             else _conformal_quantile(resid, alpha))
        pred = reg.predict(X[te])
        yt.extend(y[te].tolist()); yh.extend(pred.tolist())
        lo.extend((pred - q).tolist()); hi.extend((pred + q).tolist())
        ts.extend(subs[te].tolist())
    return (np.array(yt), np.array(yh), np.array(lo), np.array(hi), np.array(ts, dtype=object))


def run_glucose_forecast(cgm: pd.DataFrame, *,
                         thresholds: ExcursionThresholds = ExcursionThresholds(),
                         seed: int = 1, alpha: float = 0.1, n_folds: int = 5,
                         cal_frac: float = 0.25, anchor_stride: int = 8,
                         max_anchors_per_subject: int = 60, grouped: bool = True) -> ForecastReport:
    """CGM-only continuous glucose forecast with a ``1-alpha`` conformal interval, evaluated per horizon
    on subject-held-out folds. ``grouped=True`` (default) uses a PARTICIPANT-blocked conformal radius so
    the interval is honest under within-participant correlation. Reports RMSE/MAE/bias, empirical
    coverage, interval width, and the fraction of unscorable/infinite intervals per horizon. A horizon
    without enough subjects/rows is ``insufficient_data`` (never a number)."""
    rep = ForecastReport(target_coverage=round(1.0 - alpha, 4),
                         n_subjects=int(cgm["subject_id"].nunique()),
                         threshold_version=thresholds.version,
                         method=("participant-blocked conformal (CGM-only)" if grouped
                                 else "split-conformal (CGM-only)"))
    anchors = []
    for sid, g in cgm.groupby("subject_id"):
        t = pd.to_datetime(g["timestamp"]).sort_values()
        anchors += list(t.iloc[thresholds.history_minutes // 5::anchor_stride])[:max_anchors_per_subject]
    examples = build_forecast_examples(cgm, thresholds=thresholds, anchors=sorted(set(anchors)),
                                       subject_col="subject_id")
    X, y, subs, keys, horizons, _names = build_forecast_matrix(cgm, examples, thresholds=thresholds)
    n_folds = max(2, min(n_folds, int(cgm["subject_id"].nunique())))

    for h in thresholds.horizons_minutes:
        m = horizons == h
        if m.sum() < 20 or len(np.unique(subs[m])) < n_folds:
            rep.per_horizon[int(h)] = {"status": "insufficient_data",
                                       "n_examples": int(m.sum())}
            continue
        yt, yh, lo, hi, ts = _forecast_one_horizon(
            X[m], y[m], subs[m], alpha=alpha, n_folds=n_folds, cal_frac=cal_frac, seed=seed,
            grouped=grouped)
        if len(yt) == 0:
            rep.per_horizon[int(h)] = {"status": "insufficient_data", "n_examples": int(m.sum())}
            continue
        finite = np.isfinite(hi - lo)
        # coverage is honestly computed only where the interval is FINITE; an infinite interval trivially
        # "covers" and would inflate coverage, so report the unscorable fraction explicitly instead.
        frac_infinite = round(float((~finite).mean()), 4) if len(finite) else float("nan")
        cov = round(interval_coverage(yt[finite], lo[finite], hi[finite]), 4) if finite.any() else None
        rep.per_horizon[int(h)] = {
            "horizon_minutes": int(h), "n_test": int(len(yt)),
            "n_subjects_test": int(len(set(ts.tolist()))),
            "rmse": round(rmse(yt, yh), 3), "mae": round(mae(yt, yh), 3),
            "bias": round(bias(yt, yh), 3),
            "coverage": cov, "coverage_basis": "finite_intervals_only",
            "target_coverage": round(1.0 - alpha, 4),
            "conformal": "participant-blocked" if grouped else "split",
            "fraction_infinite_intervals": frac_infinite,
            "fraction_scorable": round(float(finite.mean()), 4) if len(finite) else float("nan"),
            "mean_interval_width": round(mean_interval_width(lo[finite], hi[finite]), 3)
            if finite.any() else None,
            "median_interval_width": round(float(np.median((hi - lo)[finite])), 3)
            if finite.any() else None,
            "modality_scope": MODALITY_SCOPE,
        }
    return rep
