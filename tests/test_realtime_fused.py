from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.realtime.intervention import evaluate_interventions  # noqa: E402
from dvxr.realtime.monitor import (  # noqa: E402
    FusedRealtimeMonitor,
    stream_fused_predictions,
)
from dvxr.schemas import REQUIRED_EVENT_COLUMNS  # noqa: E402

BASE = pd.Timestamp("2026-01-01T00:00:00Z")


def _row(sec, modality, channel, value):
    return {
        "subject_id": "s1", "session_id": "sess1",
        "timestamp_utc": BASE + pd.Timedelta(seconds=sec),
        "source_system": "test", "device": "test", "modality": modality,
        "channel": channel, "value": float(value), "unit": "u",
        "sampling_rate_hz": 1.0, "quality_flag": "ok",
        "label_name": "", "label_value": "",
    }


def _events(with_cgm=True, span=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sec in range(0, span, 5):
        rows.append(_row(sec, "eda", "eda", 2.0 + rng.normal(0, 0.2)))
        rows.append(_row(sec, "hr", "hr", 70 + rng.normal(0, 3)))
    if with_cgm:
        for sec in range(0, span, 60):
            rows.append(_row(sec, "cgm", "glucose", 120 + rng.normal(0, 5)))
    df = pd.DataFrame(rows)[REQUIRED_EVENT_COLUMNS]
    return df


class InterventionRuleTest(unittest.TestCase):
    def test_hypoglycemia_fires(self):
        recs = evaluate_interventions({"glucose_now": 60})
        self.assertTrue(any(r.rule == "hypoglycemia_risk" for r in recs))

    def test_high_stress_fires(self):
        recs = evaluate_interventions({"stress_band": "high"})
        self.assertTrue(any(r.rule == "high_stress" for r in recs))

    def test_silent_when_normal(self):
        state = {"glucose_now": 105, "glucose_forecast": 108, "glucose_trend": 0.1,
                 "stress_probability": 0.2, "stress_band": "low",
                 "cognitive_workload_risk": 0.3}
        self.assertEqual(evaluate_interventions(state), [])

    def test_priority_ordering(self):
        recs = evaluate_interventions({"glucose_now": 60, "stress_band": "high"})
        self.assertEqual(recs[0].rule, "hypoglycemia_risk")  # priority 100 > 70


class FusedStreamTest(unittest.TestCase):
    def test_one_row_per_nonempty_step(self):
        df = stream_fused_predictions(_events(), step_seconds=30, window_seconds=30)
        self.assertGreater(len(df), 0)
        self.assertIn("present_modalities", df.columns)
        self.assertIn("stress_probability", df.columns)
        # every row saw wearable + cgm
        self.assertTrue((df["present_modalities"].str.contains("wearable_phys")).all())

    def test_missing_modality_still_outputs(self):
        df = stream_fused_predictions(_events(with_cgm=False), step_seconds=30)
        self.assertGreater(len(df), 0)
        self.assertTrue((df["present_modalities"] == "wearable_phys").all())
        # no CGM -> glucose forecast absent but rows still produced
        if "glucose_now" in df.columns:
            self.assertTrue(df["glucose_now"].isna().all())

    def test_update_returns_expected_fields(self):
        mon = FusedRealtimeMonitor()
        res = mon.update(_events(span=120))
        for key in ("timestamp", "present_modalities", "stress_probability",
                    "stress_band", "glucose_now", "interventions"):
            self.assertIn(key, res)
        self.assertIn("cgm", res["present_modalities"])
        self.assertIn("glucose_forecast", res)

    def test_determinism(self):
        a = stream_fused_predictions(_events(seed=1), step_seconds=30)
        b = stream_fused_predictions(_events(seed=1), step_seconds=30)
        pd.testing.assert_frame_equal(a, b)

    def test_intervention_fires_in_stream_on_low_glucose(self):
        rows = []
        for sec in range(0, 200, 5):
            rows.append(_row(sec, "eda", "eda", 2.0))
        for sec in range(0, 200, 30):
            rows.append(_row(sec, "cgm", "glucose", 55))   # hypo
        df = pd.DataFrame(rows)[REQUIRED_EVENT_COLUMNS]
        out = stream_fused_predictions(df, step_seconds=30)
        self.assertTrue(out["interventions"].str.contains("Glucose is low").any())


if __name__ == "__main__":
    unittest.main()
