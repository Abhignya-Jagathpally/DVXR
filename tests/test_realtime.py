"""
Tests for goal1_pipeline.realtime
"""
from __future__ import annotations

import sys
import os
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goal1_pipeline.realtime import RealtimeMonitor, stream_predictions
from goal1_pipeline.schemas import validate_events


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_canonical_events(
    n_subjects: int = 2,
    minutes: int = 10,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a small canonical events DataFrame with cgm + eda + ppg."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01T12:00:00Z")
    rows = []

    def _row(subject_id, timestamp, modality, channel, value, unit, rate, label):
        return {
            "subject_id": subject_id,
            "session_id": "test_session",
            "timestamp_utc": timestamp,
            "source_system": "fixture",
            "device": "test_device",
            "modality": modality,
            "channel": channel,
            "value": float(value),
            "unit": unit,
            "sampling_rate_hz": float(rate),
            "quality_flag": "ok",
            "label_name": "stress_state" if label else "",
            "label_value": label,
        }

    for sid_idx in range(n_subjects):
        subject_id = f"S{sid_idx + 1:02d}"
        session_start = start + pd.Timedelta(hours=sid_idx)
        total_seconds = minutes * 60

        # --- CGM (every 5 min) ---
        n_cgm = max(4, minutes // 5 + 2)
        glucose = 100.0 + 10 * sid_idx
        for i in range(n_cgm):
            t = session_start + pd.Timedelta(minutes=i * 5)
            glucose += rng.normal(0, 2.0)
            rows.append(_row(subject_id, t, "cgm", "glucose", glucose, "mg/dL", 1 / 300.0, "non_stress"))

        # --- EDA (4 Hz) ---
        n_eda = int(total_seconds * 4)
        for i in range(n_eda):
            t = session_start + pd.Timedelta(seconds=i / 4.0)
            label = "stress" if 120 <= i < 240 else "non_stress"
            val = 1.5 + 0.8 * (label == "stress") + rng.normal(0, 0.05)
            rows.append(_row(subject_id, t, "eda", "eda", val, "uS", 4.0, label))

        # --- PPG / heart rate (1 Hz) ---
        for i in range(total_seconds):
            t = session_start + pd.Timedelta(seconds=i)
            label = "stress" if 120 <= i < 240 else "non_stress"
            val = 72 + 12 * (label == "stress") + rng.normal(0, 2.0)
            rows.append(_row(subject_id, t, "ppg", "heart_rate", val, "bpm", 1.0, label))

    return validate_events(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# RealtimeMonitor tests
# ---------------------------------------------------------------------------

class TestRealtimeMonitorNoModel(unittest.TestCase):
    """RealtimeMonitor with trained_stress_model=None (glucose-only mode)."""

    def setUp(self):
        self.events = _make_canonical_events(n_subjects=1, minutes=10)
        self.monitor = RealtimeMonitor(trained_stress_model=None, window_seconds=30)

    def test_update_returns_dict(self):
        result = self.monitor.update(self.events)
        self.assertIsInstance(result, dict)

    def test_glucose_now_present(self):
        result = self.monitor.update(self.events)
        self.assertIn("glucose_now", result)
        self.assertIsNotNone(result["glucose_now"])

    def test_glucose_now_numeric(self):
        result = self.monitor.update(self.events)
        self.assertIsInstance(result["glucose_now"], float)
        self.assertGreater(result["glucose_now"], 0)

    def test_glucose_trend_present(self):
        result = self.monitor.update(self.events)
        self.assertIn("glucose_trend", result)
        # trend can be any float (positive/negative slope)
        self.assertIsNotNone(result["glucose_trend"])

    def test_timestamp_present(self):
        result = self.monitor.update(self.events)
        self.assertIn("timestamp", result)
        self.assertIsInstance(result["timestamp"], str)

    def test_no_stress_fields_without_model(self):
        result = self.monitor.update(self.events)
        self.assertNotIn("stress_probability", result)
        self.assertNotIn("stress_label", result)

    def test_incremental_updates(self):
        """Buffer should accumulate across multiple update calls."""
        half = len(self.events) // 2
        first_half = self.events.iloc[:half]
        second_half = self.events.iloc[half:]

        monitor = RealtimeMonitor(trained_stress_model=None, window_seconds=30)
        r1 = monitor.update(first_half)
        r2 = monitor.update(second_half)
        # Both should succeed
        self.assertIsInstance(r1, dict)
        self.assertIsInstance(r2, dict)

    def test_reset_clears_buffer(self):
        self.monitor.update(self.events)
        self.monitor.reset()
        # After reset, update with events again should still work
        result = self.monitor.update(self.events)
        self.assertIsNotNone(result["glucose_now"])

    def test_no_cgm_glucose_is_none(self):
        """Events without cgm should return None glucose fields."""
        events_no_cgm = self.events[self.events["modality"] != "cgm"].copy()
        if events_no_cgm.empty:
            self.skipTest("All events are CGM; cannot test no-CGM case")
        events_no_cgm = validate_events(events_no_cgm)
        monitor = RealtimeMonitor(trained_stress_model=None)
        result = monitor.update(events_no_cgm)
        self.assertIsNone(result["glucose_now"])
        self.assertIsNone(result["glucose_trend"])


# ---------------------------------------------------------------------------
# stream_predictions tests
# ---------------------------------------------------------------------------

class TestStreamPredictions(unittest.TestCase):

    def setUp(self):
        self.events = _make_canonical_events(n_subjects=1, minutes=10)

    def test_returns_dataframe(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        self.assertIsInstance(df, pd.DataFrame)

    def test_multiple_time_steps(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        self.assertGreater(len(df), 1, "Expected multiple time steps")

    def test_contains_glucose_columns(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        self.assertIn("glucose_now", df.columns)
        self.assertIn("glucose_trend", df.columns)

    def test_timestamp_column_present(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        self.assertIn("timestamp", df.columns)

    def test_no_stress_cols_without_model(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        # stress_probability & stress_label should not appear
        self.assertNotIn("stress_probability", df.columns)
        self.assertNotIn("stress_label", df.columns)

    def test_glucose_now_has_nonnan_values(self):
        df = stream_predictions(self.events, trained_stress_model=None, step_seconds=60, window_seconds=60)
        # At least some rows should have glucose data
        non_null = df["glucose_now"].dropna()
        self.assertGreater(len(non_null), 0)

    def test_deterministic(self):
        df1 = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        df2 = stream_predictions(self.events, trained_stress_model=None, step_seconds=30, window_seconds=30)
        pd.testing.assert_frame_equal(df1, df2)

    def test_smaller_step_gives_more_rows(self):
        df_fine = stream_predictions(self.events, trained_stress_model=None, step_seconds=15, window_seconds=30)
        df_coarse = stream_predictions(self.events, trained_stress_model=None, step_seconds=60, window_seconds=30)
        self.assertGreaterEqual(len(df_fine), len(df_coarse))

    def test_empty_returns_dataframe_with_columns(self):
        """Very short event span should return empty or minimal DataFrame, not crash."""
        single_row = self.events.head(1)
        df = stream_predictions(single_row, trained_stress_model=None, step_seconds=30, window_seconds=30)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
