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
    deap_anxiety_task,
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

    def test_diabetes_task_has_no_target_leak(self):
        # The label is int(HbA1c >= 6.5); the defining glycemic labs must never appear
        # as features (regression guard for the B2a leak).
        from dvxr.bench.tasks import DIABETES_EHR_DENYLIST
        d = cgmacros_diabetes_task()
        feature_channels = {
            n.split("_", 1)[-1] for names in d.feature_names.values() for n in names
        }
        self.assertTrue(feature_channels.isdisjoint(DIABETES_EHR_DENYLIST))
        self.assertNotIn("ehr_hba1c", [n for ns in d.feature_names.values() for n in ns])


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

    def test_anxiety_task_real_label(self):
        # Real self-report label (high-arousal + low-valence quadrant), not a proxy.
        t = deap_anxiety_task(subjects=3, max_trials=None)
        self.assertEqual(t.kind, "classification")
        self.assertGreater(t.n, 0)
        self.assertIn("eeg", t.modalities)
        self.assertEqual(set(t.y.tolist()) <= {0, 1}, True)
        # both classes must be present so the task is scorable
        self.assertEqual(set(t.y.tolist()), {0, 1})

    def test_anxiety_label_derivation(self):
        from dvxr.loaders import _deap_affect_label
        # high arousal + low valence -> anxiety positive; other quadrants negative
        self.assertEqual(_deap_affect_label(3.0, 7.0, "anxiety"), ("anxiety", "high_anxiety"))
        self.assertEqual(_deap_affect_label(7.0, 7.0, "anxiety"), ("anxiety", "low_anxiety"))
        self.assertEqual(_deap_affect_label(3.0, 2.0, "anxiety"), ("anxiety", "low_anxiety"))
        # arousal scheme is unchanged
        self.assertEqual(_deap_affect_label(1.0, 8.0, "arousal"), ("arousal", "high_arousal"))


if __name__ == "__main__":
    unittest.main()
