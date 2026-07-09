"""Slice 4: real-label benchmark task builders for the newly wired datasets.
Skip-guarded on the presence of the real extracts (offline-safe CI)."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = Path(__file__).resolve().parents[1]
WESAD = ROOT / "data" / "real" / "WESAD"
CGM = ROOT / "data" / "real" / "cgmacros"

from dvxr.bench.tasks import (  # noqa: E402
    cgmacros_diabetes_task,
    cgmacros_glucose_task,
    wesad_stress_task,
)


@unittest.skipUnless(list(WESAD.glob("S*/S*.pkl")), "WESAD extract absent")
class WesadTaskTest(unittest.TestCase):
    def test_multimodal_stress_task(self):
        t = wesad_stress_task(subjects=2)
        self.assertGreater(t.n, 0)
        self.assertGreater(len(t.modalities), 1)  # multimodal
        self.assertEqual(set(t.y.tolist()) <= {0, 1}, True)


@unittest.skipUnless(list(CGM.glob("**/CGMacros-*.csv")), "CGMacros extract absent")
class CGMacrosTaskTest(unittest.TestCase):
    def test_glucose_forecast_task(self):
        g = cgmacros_glucose_task(subjects=2)
        self.assertEqual(g.kind, "forecast")
        self.assertGreater(g.n, 0)
        self.assertIn("cgm", g.modalities)

    def test_diabetes_classification_task(self):
        d = cgmacros_diabetes_task()
        self.assertEqual(d.kind, "classification")
        self.assertTrue({"cgm", "wearable_phys", "ehr"}.issubset(set(d.modalities)))
        self.assertEqual(set(d.y.tolist()) <= {0, 1}, True)


if __name__ == "__main__":
    unittest.main()
