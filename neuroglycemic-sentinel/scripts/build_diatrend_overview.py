"""Build DiaTrend-style cohort-overview figures from recorded artifacts.

Reads the aligned window table and the ingestion audit that a ``prepare-*`` command
wrote into the external runtime workspace, and renders the five DiaTrend Figure-1-style
overview panels into ``runs/<run-name>/figures/``. Nothing is trained or synthesized —
every panel is computed from the recorded CSVs. Figures are titled with the real cohort
label so a substitute cohort is never presented as DiaTrend.

Example (from the sentinel repo root):

    python scripts/build_diatrend_overview.py \
        --workspace ../neuroglycemic-runtime \
        --run-name diatrend-style-bigideas \
        --cohort-label "BIG-IDEAS cohort (DiaTrend-style analysis)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuroglycemic.diatrend_figures import (  # noqa: E402
    DEFAULT_GLUCOSE_COLUMN,
    HYPER_MG_DL,
    HYPO_MG_DL,
    build_overview_suite,
)
from src.neuroglycemic.workspace import ResearchWorkspace  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"error: required table not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="External runtime root (disjoint sibling of this repo).")
    parser.add_argument("--run-name", required=True,
                        help="Run directory under runs/ to write figures into.")
    parser.add_argument("--data", type=Path, default=None,
                        help="Aligned window table. Defaults to the BIG-IDEAS cohort.")
    parser.add_argument("--audit", type=Path, default=None,
                        help="Ingestion audit CSV. Defaults to the BIG-IDEAS audit.")
    parser.add_argument("--cohort-label",
                        default="BIG-IDEAS cohort (DiaTrend-style analysis)",
                        help="Honest cohort label printed on every figure.")
    parser.add_argument("--glucose-column", default=DEFAULT_GLUCOSE_COLUMN)
    parser.add_argument("--hypo", type=float, default=HYPO_MG_DL)
    parser.add_argument("--hyper", type=float, default=HYPER_MG_DL)
    args = parser.parse_args()

    workspace = ResearchWorkspace.create(args.workspace, repository_root=REPO_ROOT)
    data_path = args.data or workspace.aligned / "big_ideas_wearable_cgm_windows.csv.gz"
    audit_path = args.audit or workspace.canonical / "big_ideas_ingestion_audit.csv"

    windows = _read_table(Path(data_path))
    audit = _read_table(Path(audit_path))

    figure_dir = workspace.run_directory(args.run_name) / "figures"
    outputs = build_overview_suite(
        windows, audit, figure_dir,
        cohort_label=args.cohort_label,
        glucose_column=args.glucose_column,
        hypo=args.hypo, hyper=args.hyper,
    )
    print(f"Cohort label: {args.cohort_label}")
    print(f"Window table: {data_path}")
    print(f"Ingestion audit: {audit_path}")
    for name, path in outputs.items():
        print(f"  {name:>22s} -> {path}")


if __name__ == "__main__":
    main()
