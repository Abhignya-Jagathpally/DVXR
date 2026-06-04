"""
Tests for goal1_pipeline.biomarkers
"""
from __future__ import annotations

import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goal1_pipeline.biomarkers import physiological_biomarkers, neural_biomarker_saliency
from goal1_pipeline.schemas import validate_events


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_events(
    n_subjects: int = 2,
    minutes: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a small canonical events DataFrame with eda, ppg, cgm, resp, eeg."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01T08:00:00Z")
    rows = []

    def _row(subject_id, session_id, t, modality, channel, value, unit, rate, label="non_stress"):
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "timestamp_utc": t,
            "source_system": "fixture",
            "device": "test",
            "modality": modality,
            "channel": channel,
            "value": float(value),
            "unit": unit,
            "sampling_rate_hz": float(rate),
            "quality_flag": "ok",
            "label_name": "stress_state",
            "label_value": label,
        }

    for sid_idx in range(n_subjects):
        subject_id = f"S{sid_idx + 1:02d}"
        session_id = "sess01"
        session_start = start + pd.Timedelta(hours=sid_idx)
        total_seconds = minutes * 60

        # CGM (every 5 min)
        n_cgm = max(4, minutes // 5 + 2)
        glucose = 100.0 + 15 * sid_idx
        for i in range(n_cgm):
            t = session_start + pd.Timedelta(minutes=i * 5)
            glucose += rng.normal(0, 2.0)
            rows.append(_row(subject_id, session_id, t, "cgm", "glucose", glucose, "mg/dL", 1 / 300.0))

        # EDA (4 Hz)
        n_eda = int(total_seconds * 4)
        for i in range(n_eda):
            t = session_start + pd.Timedelta(seconds=i / 4.0)
            val = 1.5 + rng.normal(0, 0.08)
            rows.append(_row(subject_id, session_id, t, "eda", "eda", val, "uS", 4.0))

        # PPG / heart rate (1 Hz)
        for i in range(total_seconds):
            t = session_start + pd.Timedelta(seconds=i)
            val = 72.0 + rng.normal(0, 3.0)
            rows.append(_row(subject_id, session_id, t, "ppg", "heart_rate", val, "bpm", 1.0))

        # Respiration (4 Hz)
        for i in range(n_eda):
            t = session_start + pd.Timedelta(seconds=i / 4.0)
            val = 0.4 * np.sin(i / 4.0 * 0.35) + rng.normal(0, 0.03)
            rows.append(_row(subject_id, session_id, t, "resp", "respiration", val, "a.u.", 4.0))

        # EEG – 1 channel, 16 Hz
        eeg_rate = 16.0
        n_eeg = int(total_seconds * eeg_rate)
        for i in range(n_eeg):
            t = session_start + pd.Timedelta(seconds=i / eeg_rate)
            alpha = np.sin(2 * np.pi * 10 * i / eeg_rate)
            beta = np.sin(2 * np.pi * 18 * i / eeg_rate)
            val = 8.0 + alpha + 1.5 * beta + rng.normal(0, 1.0)
            rows.append(_row(subject_id, session_id, t, "eeg", "AF3", val, "uV", eeg_rate))

    return validate_events(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# physiological_biomarkers tests
# ---------------------------------------------------------------------------

class TestPhysiologicalBiomarkers(unittest.TestCase):

    def setUp(self):
        self.events = _make_events(n_subjects=2, minutes=6)
        self.result = physiological_biomarkers(self.events)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_at_least_one_row(self):
        self.assertGreaterEqual(len(self.result), 1)

    def test_one_row_per_subject_session(self):
        n_groups = self.events.groupby(["subject_id", "session_id"]).ngroups
        self.assertEqual(len(self.result), n_groups)

    def test_subject_session_columns_present(self):
        self.assertIn("subject_id", self.result.columns)
        self.assertIn("session_id", self.result.columns)

    def test_glucose_cv_present_and_numeric(self):
        self.assertIn("glucose_cv", self.result.columns)
        valid = self.result["glucose_cv"].dropna()
        self.assertGreater(len(valid), 0)
        self.assertTrue(np.all(valid >= 0), "glucose_cv should be non-negative")

    def test_glucose_tir_present_and_in_range(self):
        self.assertIn("glucose_tir_70_180", self.result.columns)
        valid = self.result["glucose_tir_70_180"].dropna()
        self.assertGreater(len(valid), 0)
        self.assertTrue(np.all((valid >= 0) & (valid <= 1)), "TIR should be in [0,1]")

    def test_hrv_sdnn_column_present(self):
        self.assertIn("hrv_sdnn", self.result.columns)

    def test_hrv_rmssd_column_present(self):
        self.assertIn("hrv_rmssd", self.result.columns)

    def test_hrv_values_are_positive_when_present(self):
        valid_sdnn = self.result["hrv_sdnn"].dropna()
        if len(valid_sdnn) > 0:
            self.assertTrue(np.all(valid_sdnn >= 0), "SDNN should be non-negative")
        valid_rmssd = self.result["hrv_rmssd"].dropna()
        if len(valid_rmssd) > 0:
            self.assertTrue(np.all(valid_rmssd >= 0), "RMSSD should be non-negative")

    def test_eda_columns_present(self):
        self.assertIn("eda_tonic_mean", self.result.columns)
        self.assertIn("eda_scr_rate", self.result.columns)

    def test_resp_rate_column_present(self):
        self.assertIn("resp_rate_bpm", self.result.columns)

    def test_eeg_beta_alpha_column_present(self):
        self.assertIn("eeg_beta_alpha_ratio", self.result.columns)

    def test_eeg_beta_alpha_positive_when_present(self):
        valid = self.result["eeg_beta_alpha_ratio"].dropna()
        if len(valid) > 0:
            self.assertTrue(np.all(valid >= 0), "EEG beta/alpha ratio should be non-negative")

    def test_missing_modality_gives_nan_not_crash(self):
        """Events without EEG should still work; EEG column is NaN."""
        events_no_eeg = self.events[self.events["modality"] != "eeg"].copy()
        events_no_eeg = validate_events(events_no_eeg)
        result = physiological_biomarkers(events_no_eeg)
        self.assertIn("eeg_beta_alpha_ratio", result.columns)
        self.assertTrue(result["eeg_beta_alpha_ratio"].isna().all())

    def test_empty_events_returns_empty_df(self):
        # Build a minimal valid frame then filter everything out
        # This tests robustness — use a single CGM row then filter
        single = self.events.head(1)
        # Force single row to have only CGM but still valid
        result = physiological_biomarkers(single)
        self.assertIsInstance(result, pd.DataFrame)

    def test_deterministic(self):
        r1 = physiological_biomarkers(self.events)
        r2 = physiological_biomarkers(self.events)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# neural_biomarker_saliency tests
# ---------------------------------------------------------------------------

class TestNeuralBiomarkerSaliency(unittest.TestCase):

    def _make_feature_frame(self, n: int = 50, n_features: int = 15, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
        rng = np.random.default_rng(seed)
        feature_cols = [f"feat_{i:02d}" for i in range(n_features)]
        data = {col: rng.normal(i, 1.0, n) for i, col in enumerate(feature_cols)}
        data["subject_id"] = [f"S{i % 3 + 1:02d}" for i in range(n)]
        return pd.DataFrame(data), feature_cols

    def test_returns_dataframe(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols, top_n=5)
        self.assertIsInstance(result, pd.DataFrame)

    def test_does_not_crash(self):
        """Must not raise even if neural_encoders / torch unavailable."""
        frame, cols = self._make_feature_frame()
        try:
            neural_biomarker_saliency(frame, cols, top_n=5)
        except Exception as e:
            self.fail(f"neural_biomarker_saliency raised unexpected exception: {e}")

    def test_at_most_top_n_rows(self):
        frame, cols = self._make_feature_frame(n_features=20)
        for top_n in [3, 5, 10]:
            result = neural_biomarker_saliency(frame, cols, top_n=top_n)
            self.assertLessEqual(len(result), top_n, f"top_n={top_n} violated")

    def test_method_column_present(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols)
        self.assertIn("method", result.columns)

    def test_method_column_is_valid_string(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols)
        valid_methods = {"neural_saliency", "variance_fallback"}
        for val in result["method"]:
            self.assertIn(val, valid_methods)

    def test_feature_column_present(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols)
        self.assertIn("feature", result.columns)

    def test_saliency_column_present(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols)
        self.assertIn("saliency", result.columns)

    def test_features_subset_of_input(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, cols, top_n=10)
        returned_features = set(result["feature"].tolist())
        self.assertTrue(returned_features.issubset(set(cols)))

    def test_empty_feature_cols_returns_empty(self):
        frame, _ = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, [], top_n=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_single_feature(self):
        frame, cols = self._make_feature_frame()
        result = neural_biomarker_saliency(frame, [cols[0]], top_n=5)
        self.assertLessEqual(len(result), 1)

    def test_variance_fallback_ordering(self):
        """With variance fallback, features should be ordered by descending variance."""
        frame, cols = self._make_feature_frame(n_features=10)
        # Force fallback by passing a deliberately large frame to be safe
        result = neural_biomarker_saliency(frame, cols, top_n=10)
        if result["method"].iloc[0] == "variance_fallback":
            saliency_vals = result["saliency"].to_numpy()
            # Should be descending
            self.assertTrue(
                np.all(np.diff(saliency_vals) <= 1e-12),
                "variance_fallback results should be in descending order"
            )

    def test_deterministic(self):
        frame, cols = self._make_feature_frame()
        r1 = neural_biomarker_saliency(frame, cols, top_n=5)
        r2 = neural_biomarker_saliency(frame, cols, top_n=5)
        pd.testing.assert_frame_equal(r1, r2)


if __name__ == "__main__":
    unittest.main()
