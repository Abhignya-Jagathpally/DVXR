from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.features import build_glucose_forecast_table, build_stress_windows
from goal1_pipeline.models import train_glucose_forecaster, train_stress_classifier
from goal1_pipeline.sample_data import generate_public_like_events
from goal1_pipeline.schemas import summarize_events


class PipelineSmokeTest(unittest.TestCase):
    def test_sample_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events = generate_public_like_events(
                Path(temp_dir) / "events.csv",
                subjects=3,
                minutes=8,
                eeg_channels=4,
                eeg_rate_hz=16.0,
            )
            summary = summarize_events(events)
            self.assertEqual(summary.subjects, 3)
            self.assertIn("eeg", summary.modalities)
            self.assertIn("cgm", summary.modalities)

            windows = build_stress_windows(events, window_seconds=30, step_seconds=30)
            stress_model = train_stress_classifier(windows)
            self.assertIn("accuracy", stress_model.metrics)
            self.assertIn("brier", stress_model.metrics)
            self.assertIn("risk_band", stress_model.predictions.columns)

            glucose = build_glucose_forecast_table(events)
            glucose_model = train_glucose_forecaster(glucose)
            self.assertIn("mae_mg_dl", glucose_model.metrics)
            self.assertIn("interval_coverage", glucose_model.metrics)
            self.assertIn("lower_90", glucose_model.predictions.columns)


if __name__ == "__main__":
    unittest.main()