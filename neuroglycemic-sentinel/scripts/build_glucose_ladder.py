"""Run the glucose model-selection ladder and commit the justification table.

Fits persistence/linear/tree/RF/GBM/MLP on the neural run's non-test patients and scores
them on the same held-out test patients, alongside the deep NeuroGlycemicNet metrics — so
"why the deep model?" has a same-split answer.

    python scripts/build_glucose_ladder.py \
        --windows ../neuroglycemic-runtime/aligned/cgmacros_patient_windows.csv.gz \
        --run-dir ../neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1 \
        --out-dir ../outputs/_r2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuroglycemic.model_ladder import run_glucose_ladder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for glucose_model_ladder.{md,csv} (committed).")
    args = parser.parse_args()

    result = run_glucose_ladder(args.windows, args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "glucose_model_ladder.md").write_text(result.to_markdown(), encoding="utf-8")
    result.table.to_csv(args.out_dir / "glucose_model_ladder.csv", index=False)
    print(result.to_markdown())
    print(f"\nWrote {args.out_dir}/glucose_model_ladder.md and .csv")


if __name__ == "__main__":
    main()
