"""dvxr.realtime.monitor — FusedRealtimeMonitor (ARCHITECTURE §A1 Stage 6).

A multi-modal rolling-buffer monitor that ingests streaming wearable/EEG windows AND
CGM readings, runs a CACMF-style fuse+aggregate over the present modalities, and emits
per update: stress probability + band, other task risks when their modality is present,
glucose value + short-horizon forecast + interval, and the list of present modalities.
``.update()`` / ``.reset()`` stay compatible with ``RealtimeMonitor``. Deterministic.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from dvxr.calibration import risk_band
from dvxr.config import DEFAULTS
from dvxr.fusion.aggregate import confidence_weighted
from dvxr.realtime.base import RealtimeMonitor, _latest_glucose, _run_stress_prediction
from dvxr.realtime.intervention import evaluate_interventions

# fine-grained event modality -> canonical CACMF modality
_CANON = {
    "eeg": "eeg", "eeg_bandpower": "eeg",
    "eda": "wearable_phys", "gsr": "wearable_phys", "ppg": "wearable_phys",
    "hr": "wearable_phys", "heart_rate": "wearable_phys", "temp": "wearable_phys",
    "motion": "wearable_phys", "resp": "wearable_phys", "spo2": "wearable_phys",
    "acc": "wearable_phys",
    "cgm": "cgm", "glucose": "cgm",
    "ehr": "ehr", "behavior": "behavior",
}

_TASK_MODALITY = {
    "cognitive_workload": "eeg",
    "diabetes_complication": "cgm",
    "clinical_risk": "ehr",
}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def canonical_modalities(events: pd.DataFrame) -> List[str]:
    return sorted({_CANON.get(m, m) for m in events["modality"].unique()})


def _canon_rows(buf: pd.DataFrame, canon: str) -> pd.DataFrame:
    keep = buf["modality"].map(lambda m: _CANON.get(m, m) == canon)
    return buf[keep]


def _recent_vs_global(buf: pd.DataFrame, canon: str, window: int = 30) -> Optional[float]:
    """Standardized (recent-mean - global-mean) for a modality's values -> logistic prob."""
    rows = _canon_rows(buf, canon)
    if rows.empty:
        return None
    vals = rows.sort_values("timestamp_utc")["value"].to_numpy(dtype=float)
    if len(vals) < 3:
        return 0.5
    g_mean, g_std = float(vals.mean()), float(vals.std()) or 1.0
    recent = vals[-max(3, min(window, len(vals))):]
    z = (float(recent.mean()) - g_mean) / g_std
    return float(_sig(1.5 * z))


class FusedRealtimeMonitor(RealtimeMonitor):
    def __init__(
        self,
        trained_stress_model: Any = None,
        config=DEFAULTS,
        window_seconds: int = 30,
        forecast_horizon: int = 6,
        glucose_interval: float = 15.0,
        task_predictors: Optional[Dict[str, Callable[[pd.DataFrame], float]]] = None,
    ) -> None:
        super().__init__(trained_stress_model, window_seconds)
        self.config = config
        self.forecast_horizon = forecast_horizon
        self.glucose_interval = glucose_interval
        self.task_predictors = dict(task_predictors or {})

    # -- fused stress: aggregate per-modality stress signals via confidence weighting --
    def _stress_signals(self, buf: pd.DataFrame) -> Dict[str, np.ndarray]:
        signals: Dict[str, np.ndarray] = {}
        if self._trained is not None:
            pred = _run_stress_prediction(buf, self._trained, self._window_seconds)
            if "stress_probability" in pred:
                p = float(pred["stress_probability"])
                signals["model"] = np.array([[1 - p, p]])
        for canon in ("wearable_phys", "eeg"):
            p = _recent_vs_global(buf, canon)
            if p is not None:
                signals[canon] = np.array([[1 - p, p]])
        return signals

    def update(self, new_events: pd.DataFrame) -> dict:
        base = super().update(new_events)          # timestamp, (stress), glucose_now/trend
        buf = self._buffer
        present = canonical_modalities(buf)
        result: Dict[str, Any] = dict(base)
        result["present_modalities"] = present

        # fused + aggregated stress
        signals = self._stress_signals(buf)
        if signals:
            fused = confidence_weighted(signals)   # (1, 2)
            sp = float(fused[0, 1])
            result["stress_probability"] = sp
            result["stress_label"] = "stress" if sp >= 0.5 else "non_stress"
            result["stress_band"] = risk_band(sp)
            result["stress_modalities"] = sorted(signals.keys())

        # glucose short-horizon forecast + interval
        gnow = result.get("glucose_now")
        if gnow is not None:
            gtr = result.get("glucose_trend") or 0.0
            fc = gnow + gtr * self.forecast_horizon
            result["glucose_forecast"] = fc
            result["glucose_lower"] = fc - self.glucose_interval
            result["glucose_upper"] = fc + self.glucose_interval

        # per-task proxy risks for present modalities (documented proxies)
        for task, modality in _TASK_MODALITY.items():
            if modality not in present:
                continue
            fn = self.task_predictors.get(task)
            if fn is not None:
                val = fn(buf)
            elif task == "diabetes_complication":
                val = self._diabetes_proxy(buf)
            else:  # cognitive_workload / clinical_risk fall back to activity contrast
                val = _recent_vs_global(buf, modality)
            if val is not None:
                result[f"{task}_risk"] = float(val)

        result["interventions"] = [r.message for r in evaluate_interventions(result)]
        return result

    @staticmethod
    def _diabetes_proxy(buf: pd.DataFrame) -> Optional[float]:
        rows = _canon_rows(buf, "cgm")
        if rows.empty:
            return None
        vals = rows["value"].to_numpy(dtype=float)
        return float((vals > 180).mean())   # time-above-180 proxy (documented)

    def reset(self) -> None:
        super().reset()


def stream_fused_predictions(
    events: pd.DataFrame,
    monitor: Optional[FusedRealtimeMonitor] = None,
    step_seconds: int = 30,
    window_seconds: int = 30,
) -> pd.DataFrame:
    """Replay ``events`` as a stream through a FusedRealtimeMonitor -> one row/step."""
    from dvxr.schemas import validate_events
    events = validate_events(events)
    monitor = monitor or FusedRealtimeMonitor(window_seconds=window_seconds)
    monitor.reset()

    t_min = events["timestamp_utc"].min()
    t_max = events["timestamp_utc"].max()
    cursor = t_min + pd.Timedelta(seconds=window_seconds)
    rows: List[dict] = []
    last_cursor = t_min
    while cursor <= t_max:
        step_events = events[(events["timestamp_utc"] > last_cursor)
                             & (events["timestamp_utc"] <= cursor)]
        if not step_events.empty:
            res = monitor.update(step_events)
            res = dict(res)
            res["present_modalities"] = "|".join(res.get("present_modalities", []))
            res["stress_modalities"] = "|".join(res.get("stress_modalities", []))
            res["interventions"] = "|".join(res.get("interventions", []))
            rows.append(res)
        last_cursor = cursor
        cursor += pd.Timedelta(seconds=step_seconds)

    return pd.DataFrame(rows)
