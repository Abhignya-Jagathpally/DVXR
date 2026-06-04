"""Convert an EMOTIV EPOC X / FLEX EEG export to canonical CSV.

Supports both EPOC X (14 channels) and FLEX (32 channels) device models.
Handles real device CSV exports and a synthetic demo mode.

Usage (demo, EPOC X):
    python scripts/convert_emotiv_subject.py --demo --output outputs/emotiv_demo.csv

Usage (demo, FLEX 32-ch):
    python scripts/convert_emotiv_subject.py --demo --device flex --output outputs/emotiv_flex_demo.csv

Usage (real file):
    python scripts/convert_emotiv_subject.py --input data/emotiv_export.csv \\
        --device epocx --output outputs/emotiv_subject.csv \\
        --subject-id sub01 --session-id ses01
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

# ---- Device channel definitions --------------------------------------------

# Standard EMOTIV EPOC X 14-channel names (international 10-20 system).
EPOCX_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Generic 32-channel labels for EMOTIV FLEX (subset of 10-20 system).
FLEX_CHANNELS = [
    "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
    "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6",
    "TP10", "P7", "P3", "Pz", "P4", "P8", "PO3",
    "PO4", "O1", "Oz", "O2",
]

DEVICE_CHANNELS: dict[str, list[str]] = {
    "epocx": EPOCX_CHANNELS,
    "flex": FLEX_CHANNELS,
}


# ---- Demo data generator ----------------------------------------------------

def _generate_demo_dataframe(
    device: str = "epocx",
    rate_hz: float = 128.0,
    duration_s: float = 10.0,
) -> pd.DataFrame:
    """Generate a small synthetic EMOTIV-like export DataFrame."""
    rng = np.random.default_rng(7)
    channels = DEVICE_CHANNELS.get(device, EPOCX_CHANNELS)
    n_samples = int(rate_hz * duration_s)
    period_ms = int(round(1000.0 / rate_hz))
    timestamps = pd.date_range(
        start="2026-01-01T00:00:00Z",
        periods=n_samples,
        freq=pd.Timedelta(milliseconds=period_ms),
        tz="UTC",
    )
    data: dict[str, object] = {"timestamp": timestamps}
    for ch in channels:
        data[ch] = rng.normal(0.0, 15.0, n_samples)  # synthetic EEG in uV
    return pd.DataFrame(data)


# ---- Core conversion --------------------------------------------------------

def convert(
    input_path: Union[str, None],
    output_path: Union[str, Path],
    demo: bool = False,
    device: str = "epocx",
    subject_id: str = "emotiv_sub01",
    session_id: str = "ses01",
    rate_hz: float = 128.0,
    timestamp_col: str = "timestamp",
    **opts,
) -> pd.DataFrame:
    """Convert an EMOTIV EPOC X / FLEX CSV export to the canonical event table.

    Parameters
    ----------
    input_path:
        Path to an EMOTIV device CSV export. If None or demo=True, a small
        synthetic export is generated instead.
    output_path:
        Destination path for the canonical CSV.
    demo:
        When True, generate synthetic data instead of reading a file.
    device:
        Device model: 'epocx' (14-ch) or 'flex' (32-ch).
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
    device = device.lower().strip()
    if device not in DEVICE_CHANNELS:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(DEVICE_CHANNELS)}")

    if demo or input_path is None:
        raw = _generate_demo_dataframe(device=device, rate_hz=rate_hz, duration_s=10.0)
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

    # Identify EEG channels: all numeric columns that are not the timestamp.
    eeg_candidates = [c for c in raw.columns if c != timestamp_col]

    rows: list[dict] = []
    for col in eeg_candidates:
        col_data = pd.to_numeric(raw[col], errors="coerce").dropna()
        for idx in col_data.index:
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "timestamp_utc": raw.loc[idx, timestamp_col],
                    "source_system": "emotiv",
                    "device": device,
                    "modality": "eeg",
                    "channel": col,
                    "value": float(col_data.loc[idx]),
                    "unit": "uV",
                    "sampling_rate_hz": rate_hz,
                    "quality_flag": "ok",
                    "label_name": "",
                    "label_value": "",
                }
            )

    events = validate_events(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False)
    ch_count = events["channel"].nunique()
    print(
        f"EMOTIV ({device}): {len(events)} rows, {ch_count} channels, "
        f"modalities={sorted(events['modality'].unique().tolist())}, "
        f"saved -> {output_path}"
    )
    return events


# ---- CLI entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an EMOTIV EPOC X / FLEX EEG export to canonical CSV."
    )
    parser.add_argument("--input", "-i", default=None, help="Path to EMOTIV device CSV export.")
    parser.add_argument("--output", "-o", required=True, help="Destination canonical CSV path.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data.")
    parser.add_argument(
        "--device", default="epocx", choices=["epocx", "flex"],
        help="Device model: epocx (14-ch, default) or flex (32-ch).",
    )
    parser.add_argument("--subject-id", default="emotiv_sub01", help="Subject identifier.")
    parser.add_argument("--session-id", default="ses01", help="Session identifier.")
    parser.add_argument("--rate-hz", type=float, default=128.0, help="Sampling rate in Hz.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        demo=args.demo,
        device=args.device,
        subject_id=args.subject_id,
        session_id=args.session_id,
        rate_hz=args.rate_hz,
        timestamp_col=args.timestamp_col,
    )


if __name__ == "__main__":
    main()
