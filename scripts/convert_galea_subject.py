"""Convert a Galea biosensing headset export (EEG + EXG peripheral) to canonical CSV.

Galea headset provides EEG channels alongside peripheral physiological signals
(EDA, PPG). This script handles both real device CSV exports and a synthetic demo
mode for testing without hardware.

Usage (demo):
    python scripts/convert_galea_subject.py --demo --output outputs/galea_demo.csv

Usage (real file):
    python scripts/convert_galea_subject.py --input data/galea_export.csv \\
        --output outputs/galea_subject.csv --subject-id sub01 --session-id ses01
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

# ---- Channel classification helpers ----------------------------------------

# EEG channel prefixes and patterns typical for Galea headset exports.
_EEG_PREFIXES = ("fp", "af", "f", "fc", "c", "cp", "p", "po", "o", "t", "eeg_")


def _classify_channel(name: str) -> tuple[str, str]:
    """Return (modality, unit) for a given channel name."""
    n = name.lower().strip()
    if n.startswith("eda") or n == "gsr":
        return "eda", "uS"
    if n.startswith("ppg") or n.startswith("bvp") or n.startswith("hr"):
        return "ppg", "a.u."
    # Check EEG patterns: starts with common EEG electrode prefixes or "eeg_"
    for prefix in _EEG_PREFIXES:
        if n.startswith(prefix):
            return "eeg", "uV"
    return "physiology", "a.u."


# ---- Demo data generator ----------------------------------------------------

def _generate_demo_dataframe(rate_hz: float = 128.0, duration_s: float = 10.0) -> pd.DataFrame:
    """Generate a small synthetic Galea-like export DataFrame."""
    rng = np.random.default_rng(42)
    n_samples = int(rate_hz * duration_s)
    period_ms = int(round(1000.0 / rate_hz))
    timestamps = pd.date_range(
        start="2026-01-01T00:00:00Z",
        periods=n_samples,
        freq=pd.Timedelta(milliseconds=period_ms),
        tz="UTC",
    )

    eeg_channels = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    data: dict[str, np.ndarray] = {"timestamp": timestamps}

    for ch in eeg_channels:
        # Synthetic EEG: low-amplitude noise around 0 uV
        data[ch] = rng.normal(0.0, 10.0, n_samples)

    # EDA: slow-varying signal ~1-5 uS
    data["EDA"] = rng.uniform(1.0, 5.0, n_samples)
    # PPG: sinusoidal ~60 bpm
    t = np.arange(n_samples) / rate_hz
    data["PPG"] = 0.5 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.02, n_samples)

    return pd.DataFrame(data)


# ---- Core conversion --------------------------------------------------------

def convert(
    input_path: Union[str, None],
    output_path: Union[str, Path],
    demo: bool = False,
    subject_id: str = "galea_sub01",
    session_id: str = "ses01",
    rate_hz: float = 128.0,
    timestamp_col: str = "timestamp",
    **opts,
) -> pd.DataFrame:
    """Convert a Galea headset CSV export to the canonical event table.

    Parameters
    ----------
    input_path:
        Path to a Galea device CSV export. If None or demo=True, a small
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
        Nominal sampling rate of the device export (Hz).
    timestamp_col:
        Name of the timestamp column in the input CSV.

    Returns
    -------
    pd.DataFrame
        Validated canonical event table (also written to output_path).
    """
    output_path = Path(output_path)

    if demo or input_path is None:
        raw = _generate_demo_dataframe(rate_hz=rate_hz, duration_s=10.0)
    else:
        raw = pd.read_csv(input_path)

    # Parse timestamps
    if timestamp_col in raw.columns:
        raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], utc=True)
    else:
        # Synthesize timestamps at the given rate
        period_ms = int(round(1000.0 / rate_hz))
        raw[timestamp_col] = pd.date_range(
            "2026-01-01T00:00:00Z", periods=len(raw),
            freq=pd.Timedelta(milliseconds=period_ms), tz="UTC"
        )

    # Identify signal columns (everything that is not the timestamp)
    signal_cols = [c for c in raw.columns if c != timestamp_col]

    rows: list[dict] = []
    for col in signal_cols:
        modality, unit = _classify_channel(col)
        col_data = pd.to_numeric(raw[col], errors="coerce").dropna()
        for idx in col_data.index:
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "timestamp_utc": raw.loc[idx, timestamp_col],
                    "source_system": "galea_headset",
                    "device": "galea",
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
        f"Galea: {len(events)} rows, "
        f"modalities={sorted(events['modality'].unique().tolist())}, "
        f"saved -> {output_path}"
    )
    return events


# ---- CLI entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Galea biosensing headset export to canonical CSV."
    )
    parser.add_argument("--input", "-i", default=None, help="Path to Galea device CSV export.")
    parser.add_argument("--output", "-o", required=True, help="Destination canonical CSV path.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data.")
    parser.add_argument("--subject-id", default="galea_sub01", help="Subject identifier.")
    parser.add_argument("--session-id", default="ses01", help="Session identifier.")
    parser.add_argument("--rate-hz", type=float, default=128.0, help="Sampling rate in Hz.")
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
