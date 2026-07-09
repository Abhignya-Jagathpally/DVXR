"""Slice 2: DEAP loader (auto-detects preprocessed .dat vs raw .bdf). Runs against the
real extract at data/real/deap when present; skips otherwise (offline-safe)."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.loaders import load_deap_dataset  # noqa: E402
from dvxr.schemas import REQUIRED_EVENT_COLUMNS, summarize_events  # noqa: E402

DEAP = Path(__file__).resolve().parents[1] / "data" / "real" / "deap"


def _deap_present() -> bool:
    return bool(
        list((DEAP / "data_preprocessed_python").glob("s*.dat"))
        or list(DEAP.glob("**/s*.dat"))
        or list(DEAP.glob("**/*.bdf"))
    )


@unittest.skipUnless(_deap_present(), "real DEAP extract not present")
class DeapLoaderTest(unittest.TestCase):
    def test_loads_eeg_and_physiology(self):
        df = load_deap_dataset(DEAP, subjects=2, max_trials=2)
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns)
        summary = summarize_events(df)
        self.assertEqual(summary.subjects, 2)
        self.assertIn("eeg", summary.modalities)
        self.assertIn("physiology", summary.modalities)
        # arousal labels present (preprocessed carries them)
        self.assertTrue(any(v in {"high_arousal", "low_arousal"} for v in df["label_value"]))


RAW_BDF = DEAP / "raw_bdf"


@unittest.skipUnless(list(RAW_BDF.glob("*.bdf")), "raw DEAP .bdf not present")
class DeapRawBdfTest(unittest.TestCase):
    def test_raw_bdf_loads_signals(self):
        from dvxr.loaders import load_deap_raw_bdf

        bdf = sorted(RAW_BDF.glob("*.bdf"))[0]
        df = load_deap_raw_bdf(bdf, max_seconds=10)
        summary = summarize_events(df)
        self.assertIn("eeg", summary.modalities)
        self.assertIn("physiology", summary.modalities)
        # 10-20 EEG channel names surfaced from the BioSemi montage
        self.assertIn("AF3", set(df[df["modality"] == "eeg"]["channel"]))
        self.assertEqual(df["device"].iloc[0], "biosemi_activetwo")


if __name__ == "__main__":
    unittest.main()
