from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_EVENT_COLUMNS = [
    "subject_id",
    "session_id",
    "timestamp_utc",
    "source_system",
    "device",
    "modality",
    "channel",
    "value",
    "unit",
    "sampling_rate_hz",
    "quality_flag",
    "label_name",
    "label_value",
]


@dataclass(frozen=True)
class DataSummary:
    rows: int
    subjects: int
    sessions: int
    modalities: list[str]
    devices: list[str]
    label_values: list[str]


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical event table."""
    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    clean = events.copy()
    # The 13 canonical columns are a required floor, not an exact set: a loader may carry
    # dataset-specific extra columns (e.g. glucose_source=libre|dexcom, meal_photo_path,
    # Fitbit sub-metrics). Keep the required columns first, then preserve any extras as-is.
    extra_cols = [c for c in clean.columns if c not in REQUIRED_EVENT_COLUMNS]
    clean = clean[REQUIRED_EVENT_COLUMNS + extra_cols]
    clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean["sampling_rate_hz"] = pd.to_numeric(clean["sampling_rate_hz"], errors="coerce")

    if clean["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc contains invalid timestamps")
    if clean["value"].isna().any():
        raise ValueError("value contains non-numeric entries")
    if clean["sampling_rate_hz"].isna().any():
        raise ValueError("sampling_rate_hz contains non-numeric entries")

    text_cols = [
        "subject_id",
        "session_id",
        "source_system",
        "device",
        "modality",
        "channel",
        "unit",
        "quality_flag",
        "label_name",
        "label_value",
    ]
    for col in text_cols:
        clean[col] = clean[col].fillna("").astype(str)

    clean = clean.sort_values(["subject_id", "session_id", "timestamp_utc", "modality", "channel"])
    return clean.reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> DataSummary:
    clean = validate_events(events)
    return DataSummary(
        rows=len(clean),
        subjects=clean["subject_id"].nunique(),
        sessions=clean[["subject_id", "session_id"]].drop_duplicates().shape[0],
        modalities=sorted(clean["modality"].unique().tolist()),
        devices=sorted(clean["device"].unique().tolist()),
        label_values=sorted([x for x in clean["label_value"].unique().tolist() if x]),
    )


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return frame
