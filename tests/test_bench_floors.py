"""Slice 6b: stronger floor opponents (xgboost / tabpfn / ridge_history) register and
run through the bench. Import- and data-guarded so CI stays offline-safe."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.bench.baselines import baseline_configs  # noqa: E402
from dvxr.bench.tasks import cgmacros_diabetes_task, cgmacros_glucose_task  # noqa: E402

CGM = Path(__file__).resolve().parents[1] / "data" / "real" / "cgmacros"
_HAS_XGB = importlib.util.find_spec("xgboost") is not None


@unittest.skipUnless(list(CGM.glob("**/CGMacros-*.csv")), "CGMacros absent")
class BenchFloorsTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_XGB, "xgboost not installed")
    def test_xgboost_registered_and_runs(self):
        task = cgmacros_diabetes_task(subjects=6)
        cfgs = baseline_configs(task, include_sota=False)
        self.assertIn("xgboost", cfgs)
        tr = list(range(0, task.n, 2))
        te = list(range(1, task.n, 2))
        pred = cfgs["xgboost"](task, tr, te, seed=7)
        self.assertEqual(len(pred), len(te))

    def test_ridge_history_registered_for_forecast(self):
        task = cgmacros_glucose_task(subjects=4)
        cfgs = baseline_configs(task, include_sota=False)
        self.assertIn("ridge_history", cfgs)
        # forecast-only floor should not appear on a classification task
        clf = cgmacros_diabetes_task(subjects=4)
        self.assertNotIn("ridge_history", baseline_configs(clf, include_sota=False))


if __name__ == "__main__":
    unittest.main()
