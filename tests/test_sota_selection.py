from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.sota import selected_sota_table, sota_model_table


class SotaSelectionTest(unittest.TestCase):
    def test_selected_models_are_explicit(self):
        selected = selected_sota_table()
        self.assertGreaterEqual(len(selected), 6)
        self.assertIn("EEG-X", set(selected["model"]))
        self.assertIn("BIOT", set(selected["model"]))
        self.assertIn("GluFormer", set(selected["model"]))

    def test_each_task_has_at_least_two_models_where_possible(self):
        table = sota_model_table()
        counts = table.groupby("task")["model"].nunique()
        multi_model_tasks = counts[counts >= 2]
        self.assertGreaterEqual(len(multi_model_tasks), 5)


if __name__ == "__main__":
    unittest.main()