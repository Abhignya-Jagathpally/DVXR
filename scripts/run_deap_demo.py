"""Run the DEAP EEG/peripheral arousal benchmark path.

Usage options:

    python3 scripts/run_deap_demo.py
    python3 scripts/run_deap_demo.py --events-csv data/sample/deap_s01_events.csv
    python3 scripts/run_deap_demo.py --deap-pickle /path/to/data_preprocessed_python/s01.dat

Without inputs, this script generates a small DEAP-shaped fixture so the code path is
always runnable. With real DEAP data, convert/run from the official preprocessed .dat file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.encoders import FeatureEncoder
from goal1_pipeline.explain import top_linear_contributors
from goal1_pipeline.features import build_deap_arousal_windows, feature_columns
from goal1_pipeline.loaders import load_canonical_csv, load_deap_preprocessed_pickle
from goal1_pipeline.models import train_arousal_classifier
from goal1_pipeline.sample_data import generate_deap_like_events
from goal1_pipeline.schemas import summarize_events, validate_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEAP arousal classification path.")
    parser.add_argument("--events-csv", type=Path, help="Canonical DEAP events CSV.")
    parser.add_argument("--deap-pickle", type=Path, help="Official DEAP preprocessed .dat/.pkl subject file.")
    parser.add_argument("--deap-dir", type=Path, help="Directory containing official DEAP s01.dat ... subject files.")
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--step-seconds", type=int, default=30)
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.events_csv:
        events = load_canonical_csv(args.events_csv)
        source = f"canonical CSV: {args.events_csv}"
    elif args.deap_dir:
        files = sorted(args.deap_dir.glob("s*.dat")) + sorted(args.deap_dir.glob("s*.pkl"))
        if args.max_subjects is not None:
            files = files[: args.max_subjects]
        if not files:
            sys.exit(f"No DEAP subject files found in {args.deap_dir}")
        frames = [load_deap_preprocessed_pickle(path, max_trials=args.max_trials) for path in files]
        import pandas as pd

        events = validate_events(pd.concat(frames, ignore_index=True))
        source = f"DEAP directory: {args.deap_dir} ({len(files)} subjects)"
        events.to_csv(output_dir / "deap_events_from_directory.csv", index=False)
    elif args.deap_pickle:
        events = load_deap_preprocessed_pickle(args.deap_pickle, max_trials=args.max_trials)
        source = f"DEAP preprocessed file: {args.deap_pickle}"
        events.to_csv(output_dir / "deap_events_from_pickle.csv", index=False)
    else:
        events = generate_deap_like_events(ROOT / "data" / "sample" / "deap_like_events.csv")
        source = "synthetic DEAP-shaped fixture"

    summary = summarize_events(events)
    print(f"== DEAP arousal path ({source}) ==")
    print(f"Events: rows={summary.rows}, subjects={summary.subjects}, modalities={', '.join(summary.modalities)}")
    print(f"Labels: {', '.join(summary.label_values)}")

    windows = build_deap_arousal_windows(events, window_seconds=args.window_seconds, step_seconds=args.step_seconds)
    windows.to_csv(output_dir / "deap_arousal_windows.csv", index=False)
    print(f"Arousal windows: {len(windows)} rows, {len(feature_columns(windows))} features")

    encoder = FeatureEncoder(max_components=16)
    embeddings = encoder.fit_transform(windows, feature_columns(windows))
    embeddings.to_csv(output_dir / "deap_arousal_embeddings.csv", index=False)
    print(f"Generated DEAP embeddings: shape={embeddings.shape}")

    model = train_arousal_classifier(windows)
    model.predictions.to_csv(output_dir / "deap_arousal_predictions.csv", index=False)
    print(
        "Arousal model (subject-held-out where possible): "
        f"accuracy={model.metrics['accuracy']:.3f}, "
        f"f1={model.metrics['f1']:.3f}, "
        f"auroc={model.metrics['auroc']:.3f}, "
        f"brier={model.metrics['brier']:.3f}, "
        f"ece={model.metrics['ece']:.3f}, "
        f"n_test={int(model.metrics['test_windows'])}"
    )
    print("Top arousal contributors:")
    print(top_linear_contributors(model, top_n=8).to_string(index=False))
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()