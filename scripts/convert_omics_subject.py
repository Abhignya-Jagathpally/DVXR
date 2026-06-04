#!/usr/bin/env python3
"""CLI: convert a wide multi-omics CSV to canonical events, or generate synthetic data.

Usage
-----
Generate synthetic wide table then convert::

    python scripts/convert_omics_subject.py --demo --output events.csv

Convert an existing wide CSV::

    python scripts/convert_omics_subject.py --input wide.csv --output events.csv

Optional flags::

    --id-col      Column name for subject IDs (default: subject_id)
    --label-col   Column name for per-subject labels (optional)
    --subjects    Number of synthetic subjects when --demo is used (default: 8)
    --seed        RNG seed for synthetic generation (default: 23)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Add ROOT/src to sys.path so the package is importable without installation
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from goal1_pipeline.omics import build_omics_features, generate_omics_like_table, load_omics_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a wide multi-omics CSV to canonical long-format events."
    )
    parser.add_argument(
        "--input",
        metavar="wide.csv",
        help="Path to input wide-format CSV (rows = subjects, columns = features).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="events.csv",
        help="Destination path for canonical events CSV.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a synthetic wide table first, then convert it to --output.",
    )
    parser.add_argument(
        "--id-col",
        default="subject_id",
        metavar="COL",
        help="Name of the subject-ID column in the wide CSV (default: subject_id).",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        metavar="COL",
        help="Name of the label column in the wide CSV (optional).",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        default=8,
        help="Number of synthetic subjects when --demo is used (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="RNG seed for synthetic generation (default: 23).",
    )
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.error("Provide --input <wide.csv> or use --demo to generate synthetic data.")

    if args.demo:
        # Generate a synthetic wide table into a temp file, then convert it
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        print(f"[demo] Generating synthetic multi-omics table ({args.subjects} subjects) ...")
        events = generate_omics_like_table(
            output_csv=args.output,
            subjects=args.subjects,
            seed=args.seed,
        )
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    else:
        input_path = Path(args.input)
        print(f"[convert] Loading wide CSV: {input_path} ...")
        events = load_omics_table(
            path=input_path,
            id_col=args.id_col,
            label_col=args.label_col,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(output_path, index=False)

    # Build feature summary
    features = build_omics_features(events)

    n_rows = len(events)
    n_subjects = events["subject_id"].nunique()
    modalities = sorted(events["modality"].unique().tolist())

    print(f"\n--- Canonical Events Summary ---")
    print(f"  Rows       : {n_rows}")
    print(f"  Subjects   : {n_subjects}")
    print(f"  Modalities : {modalities}")
    print(f"  Output     : {args.output}")
    print(f"\n--- Feature Matrix Shape ---")
    print(f"  Rows (subjects)  : {len(features)}")
    print(f"  Columns (total)  : {len(features.columns)}")
    omic_cols = [c for c in features.columns if c not in ("subject_id", "session_id", "target")]
    print(f"  Omic features    : {len(omic_cols)}")
    if "target" in features.columns:
        print(f"  Label classes    : {sorted(features['target'].unique().tolist())}")


if __name__ == "__main__":
    main()
