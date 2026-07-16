"""PR9: clinically-relevant evaluation metrics (spec §9)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.eval.clinical_metrics import (  # noqa: E402
    bias,
    brier_score,
    chronological_personalization_split,
    false_alerts_per_participant_day,
    fraction_events_detected_early,
    mae,
    median_event_lead_time,
    rmse,
    sensitivity_at_fixed_false_alert_rate,
)


class ClinicalMetricsTest(unittest.TestCase):
    def test_brier_perfect_is_zero(self):
        self.assertAlmostEqual(brier_score([1, 0, 1], [1.0, 0.0, 1.0]), 0.0)

    def test_rmse_mae_bias(self):
        self.assertAlmostEqual(rmse([100, 110], [102, 108]), 2.0)
        self.assertAlmostEqual(mae([100, 110], [102, 108]), 2.0)
        self.assertAlmostEqual(bias([100, 110], [102, 112]), 2.0)

    def test_sensitivity_at_fixed_far(self):
        # positives score high, negatives low → at FAR 0 we still catch the positives
        y = [1, 1, 1, 0, 0, 0]
        p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        out = sensitivity_at_fixed_false_alert_rate(y, p, target_far=0.0)
        self.assertLessEqual(out["false_alert_rate"], 0.0 + 1e-9)
        self.assertAlmostEqual(out["sensitivity"], 1.0)

    def test_sensitivity_respects_far_budget(self):
        y = [1, 1, 0, 0, 0, 0]
        p = [0.9, 0.6, 0.55, 0.4, 0.3, 0.2]     # one negative (0.55) sits above a positive (0.6? no)
        out = sensitivity_at_fixed_false_alert_rate(y, p, target_far=0.25)
        self.assertLessEqual(out["false_alert_rate"], 0.25 + 1e-9)

    def test_false_alerts_per_participant_day(self):
        y = [0, 0, 1, 0]
        p = [0.9, 0.8, 0.9, 0.1]                 # 2 false positives (the two negatives >= 0.5)
        self.assertAlmostEqual(false_alerts_per_participant_day(y, p, 0.5, participant_days=4), 0.5)

    def test_lead_time_and_early_detection(self):
        # event at t=100; alarms at 70 and 92 → earliest within 60-min horizon is 70 → lead 30
        lead = median_event_lead_time([100.0], [70.0, 92.0, 130.0], horizon=60.0)
        self.assertAlmostEqual(lead, 30.0)
        frac = fraction_events_detected_early([100.0], [70.0], horizon=60.0, min_lead=15.0)
        self.assertAlmostEqual(frac, 1.0)

    def test_missed_event_has_no_lead(self):
        # only alarm is AFTER the event → miss → no lead, early fraction 0
        self.assertTrue(np.isnan(median_event_lead_time([100.0], [130.0], horizon=60.0)))
        self.assertEqual(fraction_events_detected_early([100.0], [130.0], 60.0), 0.0)

    def test_chronological_split_never_leaks_future_into_baseline(self):
        sids = ["a", "a", "a", "a", "b", "b"]
        ts = [1, 2, 3, 4, 1, 2]
        base, evl = chronological_personalization_split(sids, ts, baseline_frac=0.5)
        # subject a: earliest 2 rows (t=1,2) are baseline; later (t=3,4) are eval
        self.assertEqual(set(base.tolist()), {0, 1, 4})
        self.assertEqual(set(evl.tolist()), {2, 3, 5})
        # no overlap
        self.assertEqual(set(base.tolist()) & set(evl.tolist()), set())


if __name__ == "__main__":
    unittest.main()
