"""
realtime.py — streaming / real-time monitoring for goal1_pipeline.

Public API
----------
RealtimeMonitor(trained_stress_model=None, window_seconds=30)
    .update(new_events: pd.DataFrame) -> dict
stream_predictions(events, trained_stress_model, step_seconds=30,
                   window_seconds=30) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import latest_stress_feature_row, _safe_slope
from .schemas import validate_events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CGM_MODALITIES = {"cgm", "glucose"}
_CGM_CHANNELS = {"glucose"}


def _latest_glucose(events: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (latest_glucose_value, short_term_trend_slope) from events.

    Filters for cgm/glucose rows, takes the most recent window (up to last
    ~30 min of readings), and computes a linear slope over time as the trend.
    Returns (None, None) if no CGM data is present.
    """
    cgm = events[
        (events["modality"].isin(_CGM_MODALITIES)) | (events["channel"].isin(_CGM_CHANNELS))
    ].copy()
    # Narrow further: must contain numeric glucose readings
    cgm = cgm[cgm["channel"] == "glucose"] if "glucose" in cgm["channel"].values else cgm
    if cgm.empty:
        return None, None

    cgm = cgm.sort_values("timestamp_utc")
    recent_end = cgm["timestamp_utc"].max()
    recent_start = recent_end - pd.Timedelta(minutes=30)
    recent = cgm[cgm["timestamp_utc"] >= recent_start]
    if recent.empty:
        recent = cgm.tail(5)

    glucose_now = float(recent["value"].iloc[-1])

    values = recent["value"].to_numpy(dtype=float)
    glucose_trend = _safe_slope(values)

    return glucose_now, glucose_trend


def _run_stress_prediction(events: pd.DataFrame, trained: Any, window_seconds: int) -> dict:
    """Return stress probability and label dict or empty dict on failure."""
    try:
        latest = latest_stress_feature_row(events, window_seconds=window_seconds)
        aligned = latest.reindex(columns=trained.feature_columns, fill_value=0.0)
        raw_probability = float(trained.model.predict_proba(aligned)[0, 1])
        if trained.calibrator is not None:
            probability = float(trained.calibrator.predict([raw_probability])[0])
        else:
            probability = raw_probability
        return {
            "stress_probability": probability,
            "stress_label": "stress" if probability >= 0.5 else "non_stress",
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# RealtimeMonitor
# ---------------------------------------------------------------------------

class RealtimeMonitor:
    """Stateful rolling-buffer monitor for real-time stress and glucose tracking.

    Parameters
    ----------
    trained_stress_model:
        A ``TrainedModel`` instance (from ``goal1_pipeline.models``) or ``None``.
        When ``None`` the stress fields are omitted from the output dict.
    window_seconds:
        Length of the feature window used for stress prediction.

    Usage
    -----
    >>> monitor = RealtimeMonitor(trained_stress_model=model, window_seconds=30)
    >>> result = monitor.update(new_event_batch)
    >>> print(result["glucose_now"], result["stress_probability"])
    """

    def __init__(
        self,
        trained_stress_model: Any = None,
        window_seconds: int = 30,
    ) -> None:
        self._trained = trained_stress_model
        self._window_seconds = int(window_seconds)
        self._buffer: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, new_events: pd.DataFrame) -> dict:
        """Append *new_events* to the buffer and return the latest summary.

        Parameters
        ----------
        new_events:
            A canonical events DataFrame (must pass ``validate_events``).

        Returns
        -------
        dict with keys:
            * ``timestamp``          — ISO-8601 string of the latest event.
            * ``stress_probability`` — float in [0,1] (only if model given).
            * ``stress_label``       — "stress"/"non_stress" (only if model given).
            * ``glucose_now``        — float or ``None`` if no CGM data.
            * ``glucose_trend``      — float slope (mg/dL per sample) or ``None``.
        """
        new_events = validate_events(new_events)

        if self._buffer is None:
            self._buffer = new_events
        else:
            self._buffer = validate_events(
                pd.concat([self._buffer, new_events], ignore_index=True)
            )

        result: dict[str, Any] = {
            "timestamp": str(self._buffer["timestamp_utc"].max().isoformat()),
        }

        # Stress prediction
        if self._trained is not None:
            result.update(_run_stress_prediction(self._buffer, self._trained, self._window_seconds))

        # Glucose
        glucose_now, glucose_trend = _latest_glucose(self._buffer)
        result["glucose_now"] = glucose_now
        result["glucose_trend"] = glucose_trend

        return result

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer = None


# ---------------------------------------------------------------------------
# stream_predictions
# ---------------------------------------------------------------------------

def stream_predictions(
    events: pd.DataFrame,
    trained_stress_model: Any,
    step_seconds: int = 30,
    window_seconds: int = 30,
) -> pd.DataFrame:
    """Simulate real-time predictions by walking the event timeline in steps.

    The function advances a cursor from the earliest event timestamp to the
    latest, moving ``step_seconds`` at a time.  At each step it evaluates the
    window ending at the cursor and records stress prediction + glucose stats.

    Parameters
    ----------
    events:
        Full canonical events DataFrame.
    trained_stress_model:
        A ``TrainedModel`` instance or ``None`` (glucose-only output).
    step_seconds:
        Time step between successive prediction points.
    window_seconds:
        Width of the feature window used for each stress prediction.

    Returns
    -------
    pd.DataFrame with one row per step and columns: ``timestamp``,
    ``stress_probability``, ``stress_label``, ``glucose_now``,
    ``glucose_trend``.  Empty steps (no events in window) are skipped.
    """
    events = validate_events(events)

    t_min = events["timestamp_utc"].min()
    t_max = events["timestamp_utc"].max()

    rows: list[dict] = []
    cursor = t_min + pd.Timedelta(seconds=window_seconds)

    while cursor <= t_max:
        window_start = cursor - pd.Timedelta(seconds=window_seconds)
        window_events = events[
            (events["timestamp_utc"] >= window_start) & (events["timestamp_utc"] <= cursor)
        ]

        if window_events.empty:
            cursor += pd.Timedelta(seconds=step_seconds)
            continue

        row: dict[str, Any] = {"timestamp": cursor.isoformat()}

        # Stress prediction
        if trained_stress_model is not None:
            row.update(_run_stress_prediction(window_events, trained_stress_model, window_seconds))

        # Glucose
        glucose_now, glucose_trend = _latest_glucose(window_events)
        row["glucose_now"] = glucose_now
        row["glucose_trend"] = glucose_trend

        rows.append(row)
        cursor += pd.Timedelta(seconds=step_seconds)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "stress_probability", "stress_label", "glucose_now", "glucose_trend"]
        )

    return pd.DataFrame(rows)
