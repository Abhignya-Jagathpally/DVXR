from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .schemas import validate_events


SIGNAL_MODALITIES = {"eeg", "eda", "ppg", "resp", "temp", "motion", "ecg", "emg", "physiology"}


def build_stress_windows(
    events: pd.DataFrame,
    window_seconds: int = 30,
    step_seconds: int = 15,
    label_name: str = "stress_state",
) -> pd.DataFrame:
    """Convert event streams into fixed-window features for stress/workload models."""
    return build_signal_windows(
        events=events,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        label_name=label_name,
    )


def build_deap_arousal_windows(
    events: pd.DataFrame,
    window_seconds: int = 30,
    step_seconds: int = 30,
) -> pd.DataFrame:
    """Convert DEAP EEG/peripheral streams into arousal-classification windows."""
    return build_signal_windows(
        events=events,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        label_name="arousal",
    )


def build_signal_windows(
    events: pd.DataFrame,
    window_seconds: int = 30,
    step_seconds: int = 15,
    label_name: str = "stress_state",
    modalities: set[str] | None = None,
) -> pd.DataFrame:
    """Convert signal events into fixed-window features for a labeled prediction task."""
    events = validate_events(events)
    modalities = modalities or SIGNAL_MODALITIES
    rows: list[dict] = []

    for (subject_id, session_id), group in events.groupby(["subject_id", "session_id"], sort=False):
        signal_group = group[group["modality"].isin(modalities)]
        if signal_group.empty:
            continue

        start = signal_group["timestamp_utc"].min()
        end = signal_group["timestamp_utc"].max()
        cursor = start

        while cursor + pd.Timedelta(seconds=window_seconds) <= end:
            stop = cursor + pd.Timedelta(seconds=window_seconds)
            window = signal_group[(signal_group["timestamp_utc"] >= cursor) & (signal_group["timestamp_utc"] < stop)]
            if not window.empty:
                row = {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "window_start": cursor,
                    "window_end": stop,
                }
                row.update(_window_features(window))
                row["target"] = _window_label(window, label_name)
                if row["target"]:
                    rows.append(row)
            cursor += pd.Timedelta(seconds=step_seconds)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No windows were created. Check timestamps, labels, and modalities.")
    return frame.fillna(0.0)


def build_raw_windows(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    modalities: list[str] | set[str] | None = None,
    samples: int = 64,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Extract per-modality RAW signal windows aligned 1:1 with ``windows``.

    Companion to :func:`build_signal_windows`: while that returns per-window *summary
    statistics* (the floor's input), this returns the per-window *raw waveform* — the
    lever past the summary-stat ceiling. For every row of ``windows`` (which must carry
    ``subject_id``, ``session_id``, ``window_start``, ``window_end``) and every modality,
    each channel's samples inside the window are resampled onto a fixed ``samples``-point
    grid (linear interpolation) and packed **channel-major** into a flat ``(C*samples,)``
    vector, so ``arr.reshape(N, C, samples)`` recovers ``(window, channel, time)``.

    Channels per modality are the sorted unique channels present in ``events`` for that
    modality, fixed across all windows so ``C`` is constant. Returns
    ``(raw, channels)`` where ``raw[m]`` has shape ``(len(windows), C_m*samples)`` and
    ``channels[m] == C_m`` — the layout :func:`dvxr.bench.raw_seq.pred_rawcnn` consumes.

    NOTE (fidelity): the raw here is whatever the loader emitted — for DEAP that is the
    decimated preprocessed signal (~8 Hz effective), so this is a *raw-waveform* contrast
    against summary stats on the same signal, not full-rate EEG. Full-rate raw is future
    work (see Slice H notes / BENCHMARK_FINDINGS C2).
    """
    events = validate_events(events)
    modset = set(modalities) if modalities is not None else set(SIGNAL_MODALITIES)
    ev = events[events["modality"].isin(modset)]
    chans = {m: sorted(ev[ev["modality"] == m]["channel"].unique())
             for m in sorted(ev["modality"].unique())}
    mods = [m for m in chans if chans[m]]
    t0 = ev["timestamp_utc"].min()

    # Pre-extract each (subject, session) → {(modality, channel): (t_seconds, values)} once.
    series: dict[tuple, dict[tuple, tuple[np.ndarray, np.ndarray]]] = {}
    for (sid, ssid), g in ev.groupby(["subject_id", "session_id"], sort=False):
        d: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for (m, ch), gg in g.groupby(["modality", "channel"], sort=False):
            gg = gg.sort_values("timestamp_utc")
            t = (gg["timestamp_utc"] - t0).dt.total_seconds().to_numpy(dtype=float)
            d[(m, ch)] = (t, gg["value"].to_numpy(dtype=float))
        series[(sid, ssid)] = d

    n = len(windows)
    raw = {m: np.zeros((n, len(chans[m]) * samples), dtype=np.float32) for m in mods}
    for i, w in enumerate(windows.itertuples(index=False)):
        d = series.get((w.subject_id, w.session_id), {})
        ws = (w.window_start - t0).total_seconds()
        we = (w.window_end - t0).total_seconds()
        grid = np.linspace(ws, we, samples)
        for m in mods:
            for ci, ch in enumerate(chans[m]):
                td = d.get((m, ch))
                if td is None:
                    continue
                t, v = td
                lo = int(np.searchsorted(t, ws, "left"))
                hi = int(np.searchsorted(t, we, "right"))
                seg = slice(ci * samples, (ci + 1) * samples)
                if hi - lo >= 2:
                    raw[m][i, seg] = np.interp(grid, t[lo:hi], v[lo:hi])
                elif hi - lo == 1:
                    raw[m][i, seg] = v[lo]
    return raw, {m: len(chans[m]) for m in mods}


def build_glucose_forecast_table(
    events: pd.DataFrame,
    history_minutes: int = 30,
    horizon_minutes: int = 30,
) -> pd.DataFrame:
    """Create glucose forecast rows from CGM events."""
    events = validate_events(events)
    cgm = events[(events["modality"] == "cgm") & (events["channel"] == "glucose")].copy()
    if cgm.empty:
        raise ValueError("No CGM glucose rows found")

    rows: list[dict] = []
    for (subject_id, session_id), group in cgm.groupby(["subject_id", "session_id"], sort=False):
        group = group.sort_values("timestamp_utc")
        for _, current in group.iterrows():
            t0 = current["timestamp_utc"]
            history_start = t0 - pd.Timedelta(minutes=history_minutes)
            future_time = t0 + pd.Timedelta(minutes=horizon_minutes)
            history = group[(group["timestamp_utc"] >= history_start) & (group["timestamp_utc"] <= t0)]
            future = group[group["timestamp_utc"] >= future_time].head(1)
            if len(history) < 3 or future.empty:
                continue

            values = history["value"].to_numpy(dtype=float)
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "timestamp_utc": t0,
                    "glucose_now": float(values[-1]),
                    "glucose_mean": float(np.mean(values)),
                    "glucose_std": float(np.std(values)),
                    "glucose_min": float(np.min(values)),
                    "glucose_max": float(np.max(values)),
                    "glucose_slope": _safe_slope(values),
                    "time_in_range_fraction": float(np.mean((values >= 70) & (values <= 180))),
                    "target_glucose": float(future.iloc[0]["value"]),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No glucose forecast rows were created")
    return frame


def latest_stress_feature_row(events: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    events = validate_events(events)
    end = events["timestamp_utc"].max()
    start = end - pd.Timedelta(seconds=window_seconds)
    window = events[(events["timestamp_utc"] >= start) & (events["timestamp_utc"] <= end)]
    row = {
        "subject_id": "stream",
        "session_id": "stream",
        "window_start": start,
        "window_end": end,
    }
    row.update(_window_features(window))
    return pd.DataFrame([row]).fillna(0.0)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"subject_id", "session_id", "window_start", "window_end", "timestamp_utc", "target", "target_glucose"}
    return [col for col in frame.columns if col not in blocked and pd.api.types.is_numeric_dtype(frame[col])]


def _window_features(window: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    for (modality, channel), group in window.groupby(["modality", "channel"], sort=True):
        values = group["value"].to_numpy(dtype=float)
        prefix = f"{modality}_{channel}".replace(" ", "_").replace("/", "_")
        features.update(_basic_stats(prefix, values))
        if modality == "eeg":
            rate = float(group["sampling_rate_hz"].median())
            features.update(_eeg_bandpower(prefix, values, rate))
    return features


def _basic_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(stats.iqr(values)),
        f"{prefix}_slope": _safe_slope(values),
        f"{prefix}_energy": float(np.mean(values**2)),
    }


def _eeg_bandpower(prefix: str, values: np.ndarray, rate: float) -> dict[str, float]:
    if len(values) < max(16, int(rate)) or rate <= 0:
        return {}
    freqs, psd = signal.welch(values, fs=rate, nperseg=min(len(values), int(rate * 2)))
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    out = {}
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        out[f"{prefix}_{band}_power"] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return out


def _window_label(window: pd.DataFrame, label_name: str) -> str:
    labels = window[window["label_name"] == label_name]["label_value"]
    if labels.empty:
        labels = window["label_value"]
    labels = labels[labels != ""]
    if labels.empty:
        return ""
    return labels.mode().iloc[0]


def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])
