from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import validate_events


def generate_public_like_events(
    output_csv: str | Path,
    subjects: int = 6,
    minutes: int = 18,
    eeg_channels: int = 6,
    eeg_rate_hz: float = 16.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Create small public-data-shaped fixtures for WESAD/DEAP/CGM/EHR-style testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    start = pd.Timestamp("2026-06-01T13:00:00Z")

    for subject_idx in range(subjects):
        subject_id = f"S{subject_idx + 1:02d}"
        stress_bias = 0.12 * subject_idx
        session_start = start + pd.Timedelta(days=subject_idx)

        _add_wearable_rows(rows, rng, subject_id, session_start, minutes, stress_bias)
        _add_eeg_rows(rows, rng, subject_id, session_start, minutes, stress_bias, eeg_channels, eeg_rate_hz)
        _add_cgm_rows(rows, rng, subject_id, session_start, minutes, stress_bias)
        _add_ehr_rows(rows, rng, subject_id, session_start, stress_bias)

    events = validate_events(pd.DataFrame(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_csv, index=False)
    return events


def _stress_state(minute: float, stress_bias: float) -> str:
    if 6 <= minute < 12:
        return "stress"
    if minute >= 15 and stress_bias > 0.35:
        return "stress"
    return "non_stress"


def _add_wearable_rows(rows, rng, subject_id, start, minutes, stress_bias) -> None:
    wearable_channels = {
        "eda": ("eda", "uS", 4.0),
        "heart_rate": ("ppg", "bpm", 1.0),
        "respiration": ("resp", "a.u.", 4.0),
        "temperature": ("temp", "C", 1.0),
        "accel_mag": ("motion", "g", 4.0),
    }

    for channel, (modality, unit, rate) in wearable_channels.items():
        samples = int(minutes * 60 * rate)
        for idx in range(samples):
            t = start + pd.Timedelta(seconds=idx / rate)
            minute = idx / rate / 60
            label = _stress_state(minute, stress_bias)
            stress = 1.0 if label == "stress" else 0.0

            if channel == "eda":
                value = 1.2 + 1.0 * stress + stress_bias + rng.normal(0, 0.08)
            elif channel == "heart_rate":
                value = 72 + 14 * stress + 4 * stress_bias + rng.normal(0, 2.5)
            elif channel == "respiration":
                value = 0.4 * np.sin(idx / rate * 0.35) + 0.35 * stress + rng.normal(0, 0.05)
            elif channel == "temperature":
                value = 33.2 + 0.3 * stress + rng.normal(0, 0.08)
            else:
                value = 0.95 + 0.18 * stress + rng.normal(0, 0.04)

            rows.append(_row(subject_id, "demo_public_like", t, "fixture", "wesad_like", modality, channel, value, unit, rate, label))


def _add_eeg_rows(rows, rng, subject_id, start, minutes, stress_bias, channel_count, rate) -> None:
    eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    eeg_channels = eeg_channels[:channel_count]
    samples = int(minutes * 60 * rate)

    for channel_idx, channel in enumerate(eeg_channels):
        phase = channel_idx / len(eeg_channels)
        for idx in range(samples):
            t = start + pd.Timedelta(seconds=idx / rate)
            minute = idx / rate / 60
            label = _stress_state(minute, stress_bias)
            stress = 1.0 if label == "stress" else 0.0
            alpha = 8 + np.sin(2 * np.pi * 10 * idx / rate + phase)
            beta = stress * np.sin(2 * np.pi * 18 * idx / rate + phase)
            value = alpha + 2.2 * beta + rng.normal(0, 1.1)
            rows.append(_row(subject_id, "demo_public_like", t, "fixture", "emotiv_like", "eeg", channel, value, "uV", rate, label))


def _add_cgm_rows(rows, rng, subject_id, start, minutes, stress_bias) -> None:
    rate = 1 / 300
    samples = max(5, int(minutes / 5) + 13)
    glucose = 95 + 12 * stress_bias

    for idx in range(samples):
        t = start + pd.Timedelta(minutes=idx * 5)
        minute = idx * 5
        label = _stress_state(minute, stress_bias)
        stress = 1.0 if label == "stress" else 0.0
        meal_effect = 28 * np.exp(-((minute - 8) ** 2) / 80)
        glucose += 0.12 * (100 - glucose) + meal_effect / 20 + 3.5 * stress + rng.normal(0, 2.0)
        rows.append(_row(subject_id, "demo_public_like", t, "fixture", "cgm_like", "cgm", "glucose", glucose, "mg/dL", rate, label))


def _add_ehr_rows(rows, rng, subject_id, start, stress_bias) -> None:
    concepts = {
        "age": 25 + 8 * stress_bias + rng.integers(0, 20),
        "bmi": 23 + 6 * stress_bias + rng.normal(0, 1.0),
        "a1c": 5.2 + 0.9 * stress_bias + rng.normal(0, 0.15),
        "diabetes_family_history": 1.0 if stress_bias > 0.35 else 0.0,
    }
    for concept, value in concepts.items():
        rows.append(
            _row(
                subject_id,
                "demo_public_like",
                start - pd.Timedelta(days=7),
                "fixture",
                "ehr_like",
                "ehr",
                concept,
                value,
                "value",
                0.0,
                "",
            )
        )


def _row(subject_id, session_id, timestamp, source, device, modality, channel, value, unit, rate, label):
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "timestamp_utc": timestamp,
        "source_system": source,
        "device": device,
        "modality": modality,
        "channel": channel,
        "value": float(value),
        "unit": unit,
        "sampling_rate_hz": float(rate),
        "quality_flag": "ok",
        "label_name": "stress_state" if label else "",
        "label_value": label,
    }
