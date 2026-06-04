from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.loaders import load_wesad_subject_pickle
from goal1_pipeline.schemas import summarize_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one official WESAD subject pickle to canonical CSV.")
    parser.add_argument("subject_pickle", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--max-samples-per-channel", type=int, default=5000)
    args = parser.parse_args()

    events = load_wesad_subject_pickle(args.subject_pickle, max_samples_per_channel=args.max_samples_per_channel)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_csv, index=False)

    summary = summarize_events(events)
    print(f"Converted WESAD subject: rows={summary.rows}, modalities={', '.join(summary.modalities)}")
    print(f"Saved canonical CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
