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
DEAP = ROOT / "data" / "real" / "deap"

from dvxr.bench.tasks import (  # noqa: E402
    cgmacros_diabetes_task,
    cgmacros_glucose_task,
    deap_arousal_task,
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


def _deap_present() -> bool:
    # DEAP may be symlinked into data/real/deap; ** does not traverse dir symlinks,
    # so check the known subdir directly too.
    return bool(
        list((DEAP / "data_preprocessed_python").glob("s*.dat"))
        or list(DEAP.glob("**/s*.dat"))
        or list(DEAP.glob("**/*.bdf"))
    )


@unittest.skipUnless(_deap_present(), "DEAP extract absent")
class DeapTaskTest(unittest.TestCase):
    def test_arousal_task(self):
        t = deap_arousal_task(subjects=2, max_trials=4)
        self.assertEqual(t.kind, "classification")
        self.assertGreater(t.n, 0)
        self.assertIn("eeg", t.modalities)
        self.assertEqual(set(t.y.tolist()) <= {0, 1}, True)


if __name__ == "__main__":
    unittest.main()
