from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.sota import selected_sota_table, sota_model_table, write_sota_report


def main() -> None:
    output_dir = ROOT / "outputs"
    comparison_path, selection_path = write_sota_report(output_dir)
    table = sota_model_table()
    selected = selected_sota_table()
    print(f"SOTA comparison rows: {len(table)}")
    print("Selected Goal 1 models:")
    for row in selected.itertuples(index=False):
        print(f"- {row.task}: {row.model} (score={row.total_score}/20)")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved selection report: {selection_path}")


if __name__ == "__main__":
    main()