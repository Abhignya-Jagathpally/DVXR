"""
Tests for goal1_pipeline.personalization
"""
from __future__ import annotations

import sys
import os
import unittest

import numpy as np
import pandas as pd

# Ensure src is on the path when running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goal1_pipeline.personalization import per_subject_normalize, PersonalizedCalibrator


def _make_multi_subject_frame(n_subjects: int = 4, n_rows_each: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a small multi-subject feature frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid_idx in range(n_subjects):
        subject_id = f"S{sid_idx + 1:02d}"
        # Each subject has a different mean/std to create distribution shift
        base_mean = 10.0 * sid_idx
        base_std = 1.0 + 0.5 * sid_idx
        for _ in range(n_rows_each):
            rows.append({
                "subject_id": subject_id,
                "feature_a": rng.normal(base_mean, base_std),
                "feature_b": rng.normal(-base_mean, base_std + 0.5),
                "feature_c": rng.normal(5.0, 0.1),  # nearly constant within subject
            })
    return pd.DataFrame(rows)


class TestPerSubjectNormalize(unittest.TestCase):

    def setUp(self):
        self.frame = _make_multi_subject_frame()
        self.feature_cols = ["feature_a", "feature_b", "feature_c"]

    def test_returns_copy_not_inplace(self):
        original_a = self.frame["feature_a"].copy()
        result = per_subject_normalize(self.frame, self.feature_cols)
        pd.testing.assert_series_equal(self.frame["feature_a"], original_a)
        self.assertFalse(result["feature_a"].equals(original_a))

    def test_per_subject_mean_near_zero(self):
        result = per_subject_normalize(self.frame, self.feature_cols)
        for sid, grp in result.groupby("subject_id"):
            for col in ["feature_a", "feature_b"]:
                mean_val = grp[col].mean()
                self.assertAlmostEqual(
                    mean_val, 0.0, places=10,
                    msg=f"Subject {sid}, feature {col}: mean={mean_val} != 0"
                )

    def test_constant_feature_becomes_zero(self):
        # feature_c has very low variance; after normalization constant rows → 0
        frame = self.frame.copy()
        frame["feature_const"] = 7.0  # perfectly constant per subject
        result = per_subject_normalize(frame, ["feature_const"])
        self.assertTrue((result["feature_const"] == 0.0).all())

    def test_non_feature_columns_preserved(self):
        result = per_subject_normalize(self.frame, self.feature_cols)
        pd.testing.assert_series_equal(result["subject_id"], self.frame["subject_id"])

    def test_shape_preserved(self):
        result = per_subject_normalize(self.frame, self.feature_cols)
        self.assertEqual(result.shape, self.frame.shape)

    def test_missing_feature_col_raises(self):
        with self.assertRaises(ValueError):
            per_subject_normalize(self.frame, ["nonexistent_col"])

    def test_missing_subject_col_raises(self):
        with self.assertRaises(ValueError):
            per_subject_normalize(self.frame, self.feature_cols, subject_col="no_such_col")

    def test_single_subject_normalizes(self):
        single = self.frame[self.frame["subject_id"] == "S01"].copy()
        result = per_subject_normalize(single, ["feature_a", "feature_b"])
        self.assertAlmostEqual(result["feature_a"].mean(), 0.0, places=10)


class TestPersonalizedCalibrator(unittest.TestCase):

    def _make_data(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        subjects = [f"S{i % 4 + 1:02d}" for i in range(n)]
        raw_probs = np.clip(rng.normal(0.5, 0.2, size=n), 0.01, 0.99)
        truths = (raw_probs + rng.normal(0, 0.1, size=n) > 0.5).astype(int)
        return subjects, raw_probs, truths

    def test_fit_predict_basic(self):
        cal = PersonalizedCalibrator()
        subjects, probs, truths = self._make_data()
        cal.fit(subjects, probs, truths)
        out = cal.predict(subjects, probs)
        self.assertEqual(len(out), len(probs))

    def test_output_in_unit_interval(self):
        cal = PersonalizedCalibrator()
        subjects, probs, truths = self._make_data()
        cal.fit(subjects, probs, truths)
        out = cal.predict(subjects, probs)
        self.assertTrue(np.all(out >= 0.0), "Some values below 0")
        self.assertTrue(np.all(out <= 1.0), "Some values above 1")

    def test_unseen_subject_falls_back_to_global(self):
        cal = PersonalizedCalibrator()
        subjects, probs, truths = self._make_data()
        cal.fit(subjects, probs, truths)
        # predict with a subject not in training data
        unseen = ["UNKNOWN_SUBJ"] * 10
        new_probs = np.full(10, 0.6)
        out = cal.predict(unseen, new_probs)
        self.assertEqual(len(out), 10)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_predict_before_fit_raises(self):
        cal = PersonalizedCalibrator()
        with self.assertRaises(RuntimeError):
            cal.predict(["S01"], [0.5])

    def test_deterministic(self):
        cal1, cal2 = PersonalizedCalibrator(), PersonalizedCalibrator()
        subjects, probs, truths = self._make_data()
        cal1.fit(subjects, probs, truths)
        cal2.fit(subjects, probs, truths)
        out1 = cal1.predict(subjects, probs)
        out2 = cal2.predict(subjects, probs)
        np.testing.assert_array_equal(out1, out2)

    def test_single_subject_no_crash(self):
        """When only one subject is present, global calibrator handles it."""
        cal = PersonalizedCalibrator()
        subjects = ["S01"] * 100
        rng = np.random.default_rng(1)
        probs = np.clip(rng.normal(0.4, 0.2, 100), 0.01, 0.99)
        truths = (probs > 0.5).astype(int)
        cal.fit(subjects, probs, truths)
        out = cal.predict(subjects, probs)
        self.assertEqual(len(out), 100)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
