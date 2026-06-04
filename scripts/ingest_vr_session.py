"""Ingest VR/AR session telemetry into the canonical event table.

Handles three signal classes from a VR/AR session CSV export:
  - Head pose (pos_x/y/z, yaw/pitch/roll) -> modality="motion"
  - Gaze + interaction events -> modality="behavior"
  - Optional heart-rate overlay -> modality="ppg"

Usage (demo):
    python scripts/ingest_vr_session.py --demo --output outputs/vr_session_demo.csv

Usage (real file):
    python scripts/ingest_vr_session.py --input data/vr_session.csv \\
        --output outputs/vr_session.csv \\
        --subject-id sub01 --session-id vr_ses01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from goal1_pipeline.schemas import validate_events

# ---- Column classification maps --------------------------------------------

# (column_name_pattern, modality, unit)
# Patterns are checked as 'startswith' on the lowercased column name.
_COLUMN_MAP: list[tuple[str, str, str]] = [
    # head pose: position (meters)
    ("pos_x", "motion", "m"),
    ("pos_y", "motion", "m"),
    ("pos_z", "motion", "m"),
    # head pose: rotation (degrees)
    ("yaw", "motion", "deg"),
    ("pitch", "motion", "deg"),
    ("roll", "motion", "deg"),
    # gaze
    ("gaze", "behavior", "a.u."),
    # interaction / controller events
    ("interact", "behavior", "a.u."),
    ("trigger", "behavior", "a.u."),
    ("grip", "behavior", "a.u."),
    ("button", "behavior", "a.u."),
    # heart rate overlay
    ("hr", "ppg", "bpm"),
    ("heart_rate", "ppg", "bpm"),
]

_MOTION_COLS = {"pos_x", "pos_y", "pos_z", "yaw", "pitch", "roll"}
_BEHAVIOR_COLS_PREFIXES = ("gaze", "interact", "trigger", "grip", "button")
_PPG_PREFIXES = ("hr", "heart_rate")


def _classify_vr_column(name: str) -> tuple[str, str]:
    """Return (modality, unit) for a VR telemetry column name."""
    n = name.lower().strip()
    for pattern, modality, unit in _COLUMN_MAP:
        if n.startswith(pattern):
            return modality, unit
    # Fallback: treat unknown numeric columns as behavior
    return "behavior", "a.u."


# ---- Demo data generator ----------------------------------------------------

def _generate_demo_dataframe(rate_hz: float = 50.0, duration_s: float = 30.0) -> pd.DataFrame:
    """Generate synthetic VR/AR session telemetry DataFrame."""
    rng = np.random.default_rng(99)
    n_samples = int(rate_hz * duration_s)
    period_ms = int(round(1000.0 / rate_hz))
    timestamps = pd.date_range(
        start="2026-01-01T00:00:00Z",
        periods=n_samples,
        freq=pd.Timedelta(milliseconds=period_ms),
        tz="UTC",
    )
    t = np.arange(n_samples) / rate_hz

    data: dict[str, object] = {"timestamp": timestamps}

    # Head pose: position (metres) — gentle drift + noise
    data["pos_x"] = 0.1 * np.sin(0.05 * t) + rng.normal(0, 0.01, n_samples)
    data["pos_y"] = 1.6 + rng.normal(0, 0.005, n_samples)   # ~1.6 m standing height
    data["pos_z"] = 0.1 * np.cos(0.05 * t) + rng.normal(0, 0.01, n_samples)

    # Head pose: rotation (degrees)
    data["yaw"] = 10.0 * np.sin(0.1 * t) + rng.normal(0, 0.5, n_samples)
    data["pitch"] = 5.0 * np.sin(0.07 * t) + rng.normal(0, 0.3, n_samples)
    data["roll"] = 2.0 * np.sin(0.03 * t) + rng.normal(0, 0.2, n_samples)

    # Gaze: normalized screen-space coordinates
    data["gaze_x"] = rng.uniform(0.0, 1.0, n_samples)
    data["gaze_y"] = rng.uniform(0.0, 1.0, n_samples)

    # Interaction event: binary trigger press (sparse)
    trigger = np.zeros(n_samples)
    trigger[rng.integers(0, n_samples, 15)] = 1.0
    data["interact_trigger"] = trigger

    # HR overlay (bpm)
    data["hr"] = 70.0 + 5.0 * np.sin(0.02 * t) + rng.normal(0, 1.0, n_samples)

    return pd.DataFrame(data)


# ---- Core conversion --------------------------------------------------------

def convert(
    input_path: Union[str, None],
    output_path: Union[str, Path],
    demo: bool = False,
    subject_id: str = "vr_sub01",
    session_id: str = "vr_ses01",
    rate_hz: float = 50.0,
    timestamp_col: str = "timestamp",
    **opts,
) -> pd.DataFrame:
    """Ingest a VR/AR session telemetry CSV into the canonical event table.

    Parameters
    ----------
    input_path:
        Path to a VR/AR session CSV export. If None or demo=True, a small
        synthetic export is generated instead.
    output_path:
        Destination path for the canonical CSV.
    demo:
        When True, generate synthetic data instead of reading a file.
    subject_id:
        Subject identifier to embed in the output.
    session_id:
        Session identifier to embed in the output.
    rate_hz:
        Nominal capture rate of the VR session data (Hz).
    timestamp_col:
        Name of the timestamp column in the input CSV.

    Returns
    -------
    pd.DataFrame
        Validated canonical event table (also written to output_path).
    """
    output_path = Path(output_path)

    if demo or input_path is None:
        raw = _generate_demo_dataframe(rate_hz=rate_hz, duration_s=30.0)
    else:
        raw = pd.read_csv(input_path)

    # Parse or synthesize timestamps
    if timestamp_col in raw.columns:
        raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], utc=True)
    else:
        period_ms = int(round(1000.0 / rate_hz))
        raw[timestamp_col] = pd.date_range(
            "2026-01-01T00:00:00Z", periods=len(raw),
            freq=pd.Timedelta(milliseconds=period_ms), tz="UTC"
        )

    signal_cols = [c for c in raw.columns if c != timestamp_col]

    rows: list[dict] = []
    for col in signal_cols:
        col_data = pd.to_numeric(raw[col], errors="coerce").dropna()
        if col_data.empty:
            continue
        modality, unit = _classify_vr_column(col)
        for idx in col_data.index:
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "timestamp_utc": raw.loc[idx, timestamp_col],
                    "source_system": "vr_session",
                    "device": "vr_ar_headset",
                    "modality": modality,
                    "channel": col,
                    "value": float(col_data.loc[idx]),
                    "unit": unit,
                    "sampling_rate_hz": rate_hz,
                    "quality_flag": "ok",
                    "label_name": "",
                    "label_value": "",
                }
            )

    events = validate_events(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False)
    print(
        f"VR session: {len(events)} rows, "
        f"modalities={sorted(events['modality'].unique().tolist())}, "
        f"saved -> {output_path}"
    )
    return events


# ---- CLI entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest VR/AR session telemetry into canonical CSV."
    )
    parser.add_argument("--input", "-i", default=None, help="Path to VR/AR session CSV export.")
    parser.add_argument("--output", "-o", required=True, help="Destination canonical CSV path.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data.")
    parser.add_argument("--subject-id", default="vr_sub01", help="Subject identifier.")
    parser.add_argument("--session-id", default="vr_ses01", help="Session identifier.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Capture rate in Hz.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        demo=args.demo,
        subject_id=args.subject_id,
        session_id=args.session_id,
        rate_hz=args.rate_hz,
        timestamp_col=args.timestamp_col,
    )


if __name__ == "__main__":
    main()
