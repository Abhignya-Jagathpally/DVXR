"""PR33: CGM-only continuous glucose forecast with a split-conformal calibrated interval.

This is an honest, single-cohort, CGM-ONLY baseline: it predicts the future glucose *level* at a
horizon and wraps it in a distribution-free prediction interval whose coverage is calibrated on a
held-out subject fold (never on the test rows). The tests pin the three properties that make it honest:

  1. the conformal quantile uses the finite-sample (n+1) correction, so calibration-set coverage is
     guaranteed >= 1 - alpha;
  2. the target is strictly future (the label at t+h never enters the feature window);
  3. on held-out participants the empirical interval coverage lands near the requested level and the
     report carries point-error metrics (RMSE/MAE) alongside interval width.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.eval.clinical_metrics import interval_coverage, mean_interval_width  # noqa: E402
from dvxr.eval.glucose_forecast import (  # noqa: E402
    _conformal_quantile,
    _grouped_conformal_quantile,
    build_forecast_examples,
    run_glucose_forecast,
)
from dvxr.targets import ExcursionThresholds  # noqa: E402


def _cohort(n_subjects=14, n=240):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        base = 150.0 if i % 2 == 0 else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        # a smooth, autocorrelated glucose curve so a forecaster has real signal to learn
        glu = np.clip(base + 30 * np.sin(np.arange(n) / 8.0 + i) + rs.normal(0, 6, n), 55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": glu}))
    return pd.concat(frames, ignore_index=True)


class ConformalQuantileTest(unittest.TestCase):
    def test_finite_sample_correction_picks_the_np1_index(self):
        # residuals 1..10, alpha=0.1 -> k = ceil((10+1)*0.9) = 10 -> the 10th (largest) = 10.0
        r = np.arange(1, 11, dtype=float)
        self.assertEqual(_conformal_quantile(r, alpha=0.1), 10.0)
        # alpha=0.2 -> k = ceil(11*0.8) = 9 -> the 9th smallest = 9.0
        self.assertEqual(_conformal_quantile(r, alpha=0.2), 9.0)

    def test_calibration_coverage_meets_target(self):
        # empirical: intervals sized by this quantile cover >= 1-alpha of the SAME residual sample
        rs = np.random.RandomState(1)
        r = np.abs(rs.normal(0, 10, 500))
        q = _conformal_quantile(r, alpha=0.1)
        self.assertGreaterEqual(np.mean(r <= q), 0.9)

    def test_empty_residuals_is_infinite(self):
        self.assertEqual(_conformal_quantile(np.array([]), alpha=0.1), float("inf"))


class GroupedConformalTest(unittest.TestCase):
    def test_participant_blocked_uses_participant_count_not_row_count(self):
        # 3 participants, many correlated rows each. Ordinary split conformal sees ~n=big and returns a
        # finite radius; grouped sees m=3 participants and (with the (m+1) correction) cannot certify a
        # finite radius at 0.1 -> inf. This is the honest "too few participants" signal.
        rs = np.random.RandomState(0)
        groups = np.repeat(["p0", "p1", "p2"], 50)
        resid = np.abs(rs.normal(0, 5, 150))
        self.assertTrue(np.isfinite(_conformal_quantile(resid, alpha=0.1)))
        self.assertEqual(_grouped_conformal_quantile(resid, groups, alpha=0.1), float("inf"))

    def test_grouped_is_finite_with_enough_participants(self):
        rs = np.random.RandomState(1)
        groups = np.repeat([f"p{i}" for i in range(30)], 10)
        resid = np.abs(rs.normal(0, 5, 300))
        q = _grouped_conformal_quantile(resid, groups, alpha=0.1)
        self.assertTrue(np.isfinite(q))
        self.assertGreater(q, 0.0)


class ForecastTargetIsFutureTest(unittest.TestCase):
    def test_label_is_strictly_after_anchor(self):
        cgm = _cohort(n_subjects=2, n=120)
        ex = build_forecast_examples(cgm, thresholds=ExcursionThresholds(history_minutes=120),
                                     subject_col="subject_id")
        rep = ex[ex["censored"] == False]  # noqa: E712
        self.assertGreater(len(rep), 0)
        # the label-source time is strictly after the anchor, within tolerance of t+h
        self.assertTrue((pd.to_datetime(rep["label_time"]) > pd.to_datetime(rep["anchor_time"])).all())


class IntervalMetricsTest(unittest.TestCase):
    def test_coverage_and_width(self):
        y = np.array([100.0, 110.0, 120.0])
        lo = np.array([90.0, 130.0, 118.0])   # 2nd point falls outside its interval
        hi = np.array([105.0, 140.0, 125.0])
        self.assertAlmostEqual(interval_coverage(y, lo, hi), 2 / 3)
        self.assertAlmostEqual(mean_interval_width(lo, hi), np.mean(hi - lo))


class EndToEndForecastTest(unittest.TestCase):
    def test_report_is_calibrated_and_carries_point_error(self):
        # split-conformal path (grouped=False) exercises the coverage mechanics on this small cohort
        rep = run_glucose_forecast(_cohort(), thresholds=ExcursionThresholds(history_minutes=120),
                                   seed=1, alpha=0.1, n_folds=5, anchor_stride=6,
                                   max_anchors_per_subject=30, grouped=False)
        self.assertTrue(rep.per_horizon)
        for h, res in rep.per_horizon.items():
            if res.get("status") == "insufficient_data":
                continue
            for field in ("rmse", "mae", "bias", "coverage", "target_coverage",
                          "mean_interval_width", "n_subjects_test", "fraction_infinite_intervals"):
                self.assertIn(field, res)
            self.assertGreaterEqual(res["coverage"], 0.80)
            self.assertLessEqual(res["coverage"], 1.0)
            self.assertGreater(res["mean_interval_width"], 0.0)
            self.assertEqual(res["target_coverage"], 0.90)

    def test_grouped_conformal_is_default_and_reports_scorability(self):
        # the honest default is participant-blocked; on a small cohort the per-fold participant count can
        # be too low to certify a finite radius, which is REPORTED (fraction_infinite_intervals), not hidden
        rep = run_glucose_forecast(_cohort(), thresholds=ExcursionThresholds(history_minutes=120),
                                   seed=1, alpha=0.1, n_folds=5, anchor_stride=6,
                                   max_anchors_per_subject=30)
        self.assertEqual(rep.method, "participant-blocked conformal (CGM-only)")
        for h, res in rep.per_horizon.items():
            if res.get("status") == "insufficient_data":
                continue
            self.assertEqual(res["conformal"], "participant-blocked")
            self.assertIn("fraction_infinite_intervals", res)
            self.assertIn("fraction_scorable", res)

    def test_forecast_makes_no_eeg_or_fused_claim(self):
        rep = run_glucose_forecast(_cohort(), thresholds=ExcursionThresholds(history_minutes=120),
                                   seed=1, alpha=0.1, n_folds=5, anchor_stride=6,
                                   max_anchors_per_subject=30)
        self.assertEqual(rep.modality_scope, "cgm_only")


if __name__ == "__main__":
    unittest.main()
