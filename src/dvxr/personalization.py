"""
personalization.py — per-subject adaptation for the goal1_pipeline.

Public API
----------
per_subject_normalize(frame, feature_cols, subject_col="subject_id") -> pd.DataFrame
PersonalizedCalibrator
    .fit(subject_ids, probabilities, truths) -> None
    .predict(subject_ids, probabilities) -> np.ndarray
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def per_subject_normalize(
    frame: pd.DataFrame,
    feature_cols: list[str],
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """DEPRECATED (leak-prone): z-score each feature WITHIN each subject using ALL of that subject's
    rows. Because it draws on a subject's future rows to normalize the present, it must NOT be used in
    any reportable evaluation — use :class:`SubjectBaselineNormalizer` with an explicit baseline cutoff
    instead (spec §7). Retained only for legacy callers; emits a DeprecationWarning.

    For each (subject, feature) pair the mean and std are computed from the
    rows that belong to that subject. Constant features (std == 0) are set to
    0 rather than NaN so that downstream code always receives a clean frame.

    Parameters
    ----------
    frame:
        Input DataFrame. Must contain *subject_col* and every column named in
        *feature_cols*.
    feature_cols:
        Names of numeric feature columns to normalize.
    subject_col:
        Column that identifies subjects (default ``"subject_id"``).

    Returns
    -------
    pd.DataFrame
        A copy of *frame* with the feature columns replaced by their
        per-subject z-scores.  All other columns are preserved unchanged.
    """
    warnings.warn(
        "per_subject_normalize uses a subject's full history (future-leaking); use "
        "SubjectBaselineNormalizer with an explicit baseline cutoff for reportable runs.",
        DeprecationWarning, stacklevel=2)
    if subject_col not in frame.columns:
        raise ValueError(f"subject_col '{subject_col}' not found in frame")
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"feature_cols not found in frame: {missing}")

    out = frame.copy()

    for col in feature_cols:
        col_vals = out[col].astype(float)
        subject_mean = col_vals.groupby(out[subject_col]).transform("mean")
        subject_std = col_vals.groupby(out[subject_col]).transform("std").fillna(0.0)
        # Where std == 0 (constant feature within a subject) → output 0
        normalized = np.where(subject_std == 0.0, 0.0, (col_vals - subject_mean) / subject_std)
        out[col] = normalized.astype(float)

    return out


class SubjectBaselineNormalizer:
    """Leak-safe per-subject normalization (spec §7): fit statistics on a subject's BASELINE
    (past-only) window, then transform later windows with those frozen stats.

    ``per_subject_normalize`` above computes each subject's mean/std from *all* their rows, which
    uses future observations to normalize the present — a look-ahead leak when the later windows are
    the evaluation set. This class separates fit (baseline period only) from transform, and records
    the baseline cutoff, so a prediction at time t is normalized using only data at or before the
    baseline cutoff. Unseen subjects fall back to the pooled baseline statistics.
    """

    def __init__(self) -> None:
        self._stats: dict = {}                    # subject -> {col: (mean, std)}
        self._pooled: dict = {}                   # col -> (mean, std)
        self._cols: list[str] = []
        self.baseline_cutoff = None
        self._baseline_samples: dict = {}         # subject -> count of baseline rows
        self._baseline_period = None              # (min_time, max_time) of the baseline slice
        self._min_baseline_samples = 1

    def fit(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
        subject_col: str = "subject_id",
        time_col: str | None = None,
        baseline_cutoff=None,
        *,
        strict: bool = False,
        min_baseline_samples: int = 1,
    ) -> "SubjectBaselineNormalizer":
        """Fit per-subject mean/std on the baseline slice ONLY.

        If ``time_col`` and ``baseline_cutoff`` are given, only rows with ``time_col <= cutoff`` are
        used (the causal baseline). Otherwise the whole frame is treated as the baseline — callers
        that pass an already-past-only frame get the same guarantee.

        ``strict=True`` (reportable mode) REQUIRES an explicit ``time_col`` + ``baseline_cutoff`` so a
        leak cannot slip in via an implicit whole-frame baseline. ``min_baseline_samples`` is the floor
        below which a subject is not personalized (it falls back to the pooled baseline and is flagged
        via :meth:`personalization_status`)."""
        if strict and (time_col is None or baseline_cutoff is None):
            raise ValueError(
                "strict personalization requires an explicit time_col and baseline_cutoff "
                "(no implicit whole-frame baseline in a reportable run)")
        self._cols = list(feature_cols)
        self._min_baseline_samples = int(min_baseline_samples)
        base = frame
        if time_col is not None and baseline_cutoff is not None:
            base = frame[frame[time_col] <= baseline_cutoff]
            self.baseline_cutoff = baseline_cutoff
        if base.empty:
            raise ValueError("baseline slice is empty — cannot fit a subject baseline")
        if time_col is not None and time_col in base.columns and len(base):
            self._baseline_period = (base[time_col].min(), base[time_col].max())
        self._baseline_samples = {str(sid): int(len(g)) for sid, g in base.groupby(subject_col)}
        for col in self._cols:
            vals = base[col].astype(float)
            self._pooled[col] = (float(vals.mean()), float(vals.std() or 0.0))
            for sid, g in vals.groupby(base[subject_col]):
                if len(g) < self._min_baseline_samples:
                    continue                       # too little baseline → pooled fallback, flagged
                self._stats.setdefault(str(sid), {})[col] = (float(g.mean()), float(g.std() or 0.0))
        return self

    def personalization_status(self, subject_id=None) -> dict:
        """Report whether a subject is genuinely personalized or fell back to the pooled baseline.

        Honest surfacing (spec §7): a subject with too few baseline samples is NOT personalized, and
        the report says so via ``fallback_used=True`` rather than silently pooling."""
        if subject_id is None:
            return {
                "personalization_status": "fitted" if self._stats else "pooled_only",
                "n_subjects_personalized": len(self._stats),
                "baseline_period": self._baseline_period,
                "min_baseline_samples": self._min_baseline_samples,
            }
        sid = str(subject_id)
        n = self._baseline_samples.get(sid, 0)
        personalized = sid in self._stats
        return {
            "personalization_status": "subject_specific" if personalized else "pooled_fallback",
            "baseline_samples": n,
            "baseline_period": self._baseline_period,
            "fallback_used": not personalized,
        }

    def transform(
        self,
        frame: pd.DataFrame,
        subject_col: str = "subject_id",
    ) -> pd.DataFrame:
        """Apply frozen baseline stats. Unknown subjects use the pooled baseline; constant features
        (std==0) map to 0 (never NaN)."""
        if not self._cols:
            raise ValueError("normalizer is not fitted")
        out = frame.copy()
        for col in self._cols:
            vals = out[col].astype(float).to_numpy()
            pooled_std = self._pooled[col][1]
            means, scales = [], []
            for sid in out[subject_col]:
                m, s = self._stats.get(str(sid), {}).get(col, self._pooled[col])
                # constant baseline (std==0) ⇒ fall back to the pooled std, then to 1.0, so a
                # departure from a flat baseline stays visible instead of being zeroed out.
                scale = s if s > 0 else (pooled_std if pooled_std > 0 else 1.0)
                means.append(m)
                scales.append(scale)
            out[col] = (vals - np.asarray(means)) / np.asarray(scales)
        return out


class PersonalizedCalibrator:
    """Per-subject probability recalibration on top of a global model.

    For each subject seen during ``fit``, a tiny logistic recalibration model
    is fitted on that subject's held-out calibration slice.  At prediction
    time the appropriate per-subject recalibrator is applied; unseen subjects
    fall back to the global (population-level) recalibrator.

    All operations are deterministic (``random_state=0``).

    Usage
    -----
    >>> cal = PersonalizedCalibrator()
    >>> cal.fit(subject_ids, raw_probs, truths)
    >>> adjusted = cal.predict(subject_ids, raw_probs)
    """

    def __init__(self) -> None:
        self._global_calibrator: LogisticRegression | None = None
        self._subject_calibrators: dict[str, LogisticRegression] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_single(X: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
        """Fit a single-feature logistic model; return None if it cannot be fit."""
        if len(X) < 2 or len(np.unique(y)) < 2:
            return None
        lr = LogisticRegression(max_iter=500, random_state=0, C=1.0)
        lr.fit(X.reshape(-1, 1), y)
        return lr

    @staticmethod
    def _apply(calibrator: LogisticRegression | None, probs: np.ndarray) -> np.ndarray:
        if calibrator is None:
            return np.clip(probs, 0.0, 1.0)
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        subject_ids: Sequence[str],
        probabilities: Sequence[float],
        truths: Sequence[int],
    ) -> None:
        """Fit per-subject calibrators plus a global fallback.

        Parameters
        ----------
        subject_ids:
            Array-like of subject identifiers, one per sample.
        probabilities:
            Raw (uncalibrated) model probabilities in [0, 1], one per sample.
        truths:
            Binary ground-truth labels (0 / 1), one per sample.
        """
        sids = np.asarray(subject_ids, dtype=str)
        probs = np.asarray(probabilities, dtype=float)
        y = np.asarray(truths, dtype=int)

        # Global calibrator (all data)
        self._global_calibrator = self._fit_single(probs, y)

        # Per-subject calibrators
        self._subject_calibrators = {}
        for sid in np.unique(sids):
            mask = sids == sid
            sub_cal = self._fit_single(probs[mask], y[mask])
            if sub_cal is not None:
                self._subject_calibrators[sid] = sub_cal

    def predict(
        self,
        subject_ids: Sequence[str],
        probabilities: Sequence[float],
    ) -> np.ndarray:
        """Return calibrated probabilities in [0, 1], one per sample.

        Subjects not seen during ``fit`` are handled by the global calibrator.

        Parameters
        ----------
        subject_ids:
            Array-like of subject identifiers, one per sample.
        probabilities:
            Raw model probabilities in [0, 1], one per sample.

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in [0, 1].
        """
        if self._global_calibrator is None:
            raise RuntimeError("PersonalizedCalibrator.fit() must be called before predict().")

        sids = np.asarray(subject_ids, dtype=str)
        probs = np.asarray(probabilities, dtype=float)
        result = np.empty(len(probs), dtype=float)

        for i, (sid, p) in enumerate(zip(sids, probs)):
            cal = self._subject_calibrators.get(sid, self._global_calibrator)
            result[i] = self._apply(cal, np.array([p]))[0]

        return np.clip(result, 0.0, 1.0)
