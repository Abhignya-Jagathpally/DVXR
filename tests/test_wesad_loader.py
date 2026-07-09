"""Slice 1: WESAD dataset loader. Runs against the real Siegen extract when present
(data/real/WESAD/S*/S*.pkl); skips otherwise so CI stays offline-safe."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.loaders import WESAD_CONDITION_LABELS, load_wesad_dataset  # noqa: E402
from dvxr.schemas import REQUIRED_EVENT_COLUMNS, summarize_events  # noqa: E402

WESAD_DIR = Path(__file__).resolve().parents[1] / "data" / "real" / "WESAD"


@unittest.skipUnless(list(WESAD_DIR.glob("S*/S*.pkl")), "real WESAD extract not present")
class WesadLoaderTest(unittest.TestCase):
    def test_loads_two_subjects_with_labels(self):
        df = load_wesad_dataset(WESAD_DIR, subjects=2, max_samples_per_channel=1000)
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns)
        summary = summarize_events(df)
        self.assertEqual(summary.subjects, 2)
        # chest + wrist physiological modalities present
        self.assertTrue({"eda", "ecg", "motion"}.issubset(set(summary.modalities)))
        # real protocol conditions (baseline/stress/amusement/meditation) are represented
        codes = {int(v) for v in df["label_value"]}
        self.assertIn(2, codes)  # stress
        self.assertTrue({1, 3, 4}.intersection(codes))  # baseline/amusement/meditation
        self.assertEqual(WESAD_CONDITION_LABELS[2], "stress")


if __name__ == "__main__":
    unittest.main()
