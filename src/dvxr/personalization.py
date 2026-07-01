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

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def per_subject_normalize(
    frame: pd.DataFrame,
    feature_cols: list[str],
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Z-score each feature WITHIN each subject.

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
