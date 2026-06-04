"""
biomarkers.py — explainable neural + physiological biomarkers for goal1_pipeline.

Public API
----------
physiological_biomarkers(events) -> pd.DataFrame
neural_biomarker_saliency(frame, feature_cols, top_n=10) -> pd.DataFrame
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])


def _hrv_metrics(rr_intervals: np.ndarray) -> dict[str, float]:
    """Compute SDNN and RMSSD from an array of RR (or IBI) intervals in ms."""
    if len(rr_intervals) < 2:
        return {"hrv_sdnn": float("nan"), "hrv_rmssd": float("nan")}
    sdnn = float(np.std(rr_intervals, ddof=1)) if len(rr_intervals) > 1 else float("nan")
    successive_diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else float("nan")
    return {"hrv_sdnn": sdnn, "hrv_rmssd": rmssd}


def _rr_from_heart_rate(hr_values: np.ndarray) -> np.ndarray:
    """Convert heart rate (bpm) to approximate RR intervals (ms)."""
    hr_valid = hr_values[hr_values > 0]
    if len(hr_valid) == 0:
        return np.array([])
    return 60_000.0 / hr_valid


def _eda_metrics(eda_values: np.ndarray) -> dict[str, float]:
    """Compute EDA tonic mean and SCR rate proxy."""
    if len(eda_values) == 0:
        return {"eda_tonic_mean": float("nan"), "eda_scr_rate": float("nan")}
    tonic_mean = float(np.mean(eda_values))
    # SCR proxy: count of upward zero-crossings of the de-meaned signal
    detrended = eda_values - np.mean(eda_values)
    crossings = int(np.sum((detrended[:-1] < 0) & (detrended[1:] >= 0)))
    # Express as rate per minute (assume 4 Hz sampling if unknown; use len)
    scr_rate = float(crossings)  # raw count — caller interprets relative to duration
    return {"eda_tonic_mean": tonic_mean, "eda_scr_rate": scr_rate}


def _resp_rate(resp_values: np.ndarray, sampling_rate_hz: float) -> dict[str, float]:
    """Estimate respiration rate in breaths/min via zero-crossing counting."""
    if len(resp_values) < 4 or sampling_rate_hz <= 0:
        return {"resp_rate_bpm": float("nan")}
    detrended = resp_values - np.mean(resp_values)
    crossings = int(np.sum((detrended[:-1] < 0) & (detrended[1:] >= 0)))
    # Two zero-crossings per breath cycle
    duration_min = len(resp_values) / sampling_rate_hz / 60.0
    rate = (crossings / 2.0) / duration_min if duration_min > 0 else float("nan")
    return {"resp_rate_bpm": float(rate)}


def _glucose_metrics(glucose_values: np.ndarray) -> dict[str, float]:
    """Compute coefficient of variation and time-in-range for glucose (mg/dL)."""
    if len(glucose_values) == 0:
        return {"glucose_cv": float("nan"), "glucose_tir_70_180": float("nan")}
    mean_g = float(np.mean(glucose_values))
    std_g = float(np.std(glucose_values))
    cv = (std_g / mean_g) if mean_g != 0 else float("nan")
    tir = float(np.mean((glucose_values >= 70) & (glucose_values <= 180)))
    return {"glucose_cv": cv, "glucose_tir_70_180": tir}


def _eeg_band_ratio(eeg_values: np.ndarray, sampling_rate_hz: float) -> dict[str, float]:
    """Compute beta/alpha power ratio from an EEG channel."""
    try:
        from scipy import signal as sp_signal
        if len(eeg_values) < max(16, int(sampling_rate_hz)) or sampling_rate_hz <= 0:
            return {"eeg_beta_alpha_ratio": float("nan")}
        freqs, psd = sp_signal.welch(
            eeg_values, fs=sampling_rate_hz,
            nperseg=min(len(eeg_values), int(sampling_rate_hz * 2))
        )
        alpha_mask = (freqs >= 8) & (freqs < 13)
        beta_mask = (freqs >= 13) & (freqs < 30)
        alpha_power = float(np.trapezoid(psd[alpha_mask], freqs[alpha_mask])) if alpha_mask.any() else 0.0
        beta_power = float(np.trapezoid(psd[beta_mask], freqs[beta_mask])) if beta_mask.any() else 0.0
        ratio = beta_power / alpha_power if alpha_power > 0 else float("nan")
        return {"eeg_beta_alpha_ratio": ratio}
    except Exception:
        return {"eeg_beta_alpha_ratio": float("nan")}


# ---------------------------------------------------------------------------
# physiological_biomarkers
# ---------------------------------------------------------------------------

def physiological_biomarkers(events: pd.DataFrame) -> pd.DataFrame:
    """Compute interpretable physiological biomarkers per (subject_id, session_id).

    Modality coverage
    -----------------
    * **HRV** (SDNN & RMSSD) — from ``ppg/heart_rate``, ``ecg``, or
      ``heart_rate`` channel; falls back to deriving pseudo-RR from bpm.
    * **EDA** — tonic mean and SCR-rate proxy from the ``eda`` modality.
    * **Respiration** — rate (breaths/min) from the ``resp`` modality.
    * **Glucose** — CV and time-in-range 70–180 mg/dL from ``cgm/glucose``.
    * **EEG** — beta/alpha power ratio from the ``eeg`` modality.

    Missing modality → NaN for those columns (not dropped).

    Parameters
    ----------
    events:
        Canonical events DataFrame (does NOT require prior ``validate_events``
        but all canonical columns must be present).

    Returns
    -------
    pd.DataFrame with one row per (subject_id, session_id) and biomarker
    columns.
    """
    from .schemas import validate_events as _validate
    events = _validate(events)

    rows: list[dict] = []

    for (subject_id, session_id), grp in events.groupby(["subject_id", "session_id"], sort=False):
        row: dict = {"subject_id": subject_id, "session_id": session_id}

        # ---- HRV ----
        rr_computed = False
        for (mod, chan), sub in grp.groupby(["modality", "channel"], sort=False):
            vals = sub["value"].to_numpy(dtype=float)
            if mod in ("ppg", "ecg") and "heart_rate" not in chan:
                # Treat raw ppg/ecg as already in RR-like units if values
                # are in a plausible IBI range (300–2000 ms), else skip.
                if vals.mean() > 200:
                    row.update(_hrv_metrics(vals))
                    rr_computed = True
                    break
            if chan == "heart_rate" or mod == "ppg":
                rr = _rr_from_heart_rate(vals)
                if len(rr) >= 2:
                    row.update(_hrv_metrics(rr))
                    rr_computed = True
                    break

        if not rr_computed:
            # Try any heart_rate-labelled channel
            hr_rows = grp[grp["channel"] == "heart_rate"]
            if not hr_rows.empty:
                rr = _rr_from_heart_rate(hr_rows["value"].to_numpy(dtype=float))
                row.update(_hrv_metrics(rr))
            else:
                row.update({"hrv_sdnn": float("nan"), "hrv_rmssd": float("nan")})

        # ---- EDA ----
        eda_rows = grp[grp["modality"] == "eda"]
        if not eda_rows.empty:
            row.update(_eda_metrics(eda_rows["value"].to_numpy(dtype=float)))
        else:
            row.update({"eda_tonic_mean": float("nan"), "eda_scr_rate": float("nan")})

        # ---- Respiration ----
        resp_rows = grp[grp["modality"] == "resp"]
        if not resp_rows.empty:
            rate = float(resp_rows["sampling_rate_hz"].median())
            row.update(_resp_rate(resp_rows["value"].to_numpy(dtype=float), rate))
        else:
            row.update({"resp_rate_bpm": float("nan")})

        # ---- Glucose ----
        glucose_rows = grp[
            (grp["modality"] == "cgm") & (grp["channel"] == "glucose")
        ]
        if glucose_rows.empty:
            glucose_rows = grp[grp["channel"] == "glucose"]
        if not glucose_rows.empty:
            row.update(_glucose_metrics(glucose_rows["value"].to_numpy(dtype=float)))
        else:
            row.update({"glucose_cv": float("nan"), "glucose_tir_70_180": float("nan")})

        # ---- EEG beta/alpha ----
        eeg_rows = grp[grp["modality"] == "eeg"]
        if not eeg_rows.empty:
            rate_eeg = float(eeg_rows["sampling_rate_hz"].median())
            # Concatenate all channels for a quick aggregate ratio
            eeg_vals = eeg_rows["value"].to_numpy(dtype=float)
            row.update(_eeg_band_ratio(eeg_vals, rate_eeg))
        else:
            row.update({"eeg_beta_alpha_ratio": float("nan")})

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id", "session_id",
                "hrv_sdnn", "hrv_rmssd",
                "eda_tonic_mean", "eda_scr_rate",
                "resp_rate_bpm",
                "glucose_cv", "glucose_tir_70_180",
                "eeg_beta_alpha_ratio",
            ]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# neural_biomarker_saliency
# ---------------------------------------------------------------------------

def neural_biomarker_saliency(
    frame: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-*n* most salient features via gradient saliency or variance.

    Attempts to use ``goal1_pipeline.neural_encoders.NeuralBiosignalEncoder``
    (fitted via ``fit_transform`` + ``gradient_saliency``).  If the module is
    absent, torch is unavailable, or any error occurs, falls back to ranking
    features by variance (``method = "variance_fallback"``).

    Parameters
    ----------
    frame:
        DataFrame containing at least the columns in *feature_cols*.
    feature_cols:
        Numeric feature columns to rank.
    top_n:
        Maximum number of features to return.

    Returns
    -------
    pd.DataFrame with columns ``feature``, ``saliency``, ``method`` and at
    most *top_n* rows, sorted by descending saliency.
    """
    top_n = max(1, int(top_n))
    feature_cols = [c for c in feature_cols if c in frame.columns]

    if not feature_cols:
        return pd.DataFrame(columns=["feature", "saliency", "method"])

    # ---- Attempt neural path ----
    try:
        import torch  # noqa: F401 — presence check
        from goal1_pipeline.neural_encoders import NeuralBiosignalEncoder  # type: ignore

        encoder = NeuralBiosignalEncoder()
        encoder.fit_transform(frame, feature_cols)
        attribution_df = encoder.gradient_saliency(frame, feature_cols)

        # gradient_saliency returns per-row attribution: shape (n_rows, n_features)
        # with the feature names as columns. Aggregate to one score per feature by
        # taking the mean absolute attribution across rows.
        if isinstance(attribution_df, pd.DataFrame) and not attribution_df.empty:
            shared = [c for c in feature_cols if c in attribution_df.columns]
            if not shared:
                raise ValueError("gradient_saliency returned no recognized feature columns")
            per_feature = attribution_df[shared].abs().mean(axis=0)
            result = pd.DataFrame({"feature": per_feature.index, "saliency": per_feature.values})
            result = result.sort_values("saliency", ascending=False).head(top_n)
            result["method"] = "neural_saliency"
            return result.reset_index(drop=True)

        raise ValueError("gradient_saliency returned empty result; falling back")

    except Exception:
        pass  # Fall through to variance fallback

    # ---- Variance fallback ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        variances = frame[feature_cols].var(numeric_only=True)

    result = pd.DataFrame({
        "feature": variances.index,
        "saliency": variances.values,
    })
    result = result.sort_values("saliency", ascending=False).head(top_n)
    result["method"] = "variance_fallback"
    return result.reset_index(drop=True)
