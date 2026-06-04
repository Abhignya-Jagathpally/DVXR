from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REAL_DIR = ROOT / "data" / "real"
NONEEG_DIR = REAL_DIR / "noneeg"
MIMIC_DIR = REAL_DIR / "mimic_demo" / "hosp"
CGM_DIR = REAL_DIR / "shanghai_cgm"


@unittest.skipUnless(
    (NONEEG_DIR / "Subject1_AccTempEDA.hea").exists(),
    "Non-EEG data not downloaded (run: python3 scripts/fetch_data.py noneeg).",
)
class NonEegRealDataTest(unittest.TestCase):
    def test_stress_pipeline_on_real_data(self):
        from goal1_pipeline.features import build_stress_windows
        from goal1_pipeline.loaders import load_noneeg_subject
        from goal1_pipeline.models import train_stress_classifier
        from goal1_pipeline.schemas import summarize_events

        events = load_noneeg_subject(NONEEG_DIR, subject=1)
        summary = summarize_events(events)
        self.assertGreater(summary.rows, 0)
        self.assertIn("eda", summary.modalities)
        self.assertEqual(set(summary.label_values), {"non_stress", "stress"})

        windows = build_stress_windows(events, window_seconds=30, step_seconds=30)
        self.assertGreater(len(windows), 0)
        # Single-subject row-level fallback split still trains and scores.
        model = train_stress_classifier(windows)
        self.assertIn("auroc", model.metrics)


@unittest.skipUnless(
    (MIMIC_DIR / "labevents.csv.gz").exists(),
    "MIMIC-IV demo not downloaded (run: python3 scripts/fetch_data.py mimic-demo).",
)
class MimicRealDataTest(unittest.TestCase):
    def test_ehr_ingestion(self):
        from goal1_pipeline.loaders import load_mimic_demo_ehr
        from goal1_pipeline.schemas import summarize_events

        ehr = load_mimic_demo_ehr(MIMIC_DIR, max_lab_rows=2000)
        summary = summarize_events(ehr)
        self.assertGreater(summary.subjects, 0)
        self.assertEqual(summary.modalities, ["ehr"])


@unittest.skipUnless(
    CGM_DIR.exists() and any(CGM_DIR.glob("*.xlsx")),
    "Shanghai CGM not downloaded (run: python3 scripts/fetch_data.py shanghai-cgm).",
)
class ShanghaiCgmRealDataTest(unittest.TestCase):
    def test_glucose_forecast_on_real_data(self):
        from goal1_pipeline.features import build_glucose_forecast_table
        from goal1_pipeline.loaders import load_shanghai_cgm_dataset
        from goal1_pipeline.models import train_glucose_forecaster
        from goal1_pipeline.schemas import summarize_events

        events = load_shanghai_cgm_dataset(CGM_DIR, max_patients=4)
        summary = summarize_events(events)
        self.assertEqual(summary.modalities, ["cgm"])
        self.assertGreater(summary.rows, 0)

        table = build_glucose_forecast_table(events, history_minutes=120, horizon_minutes=30)
        model = train_glucose_forecaster(table)
        self.assertIn("mae_mg_dl", model.metrics)
        self.assertIn("lower_90", model.predictions.columns)


if __name__ == "__main__":
    unittest.main()
