"""Regression: the CGM service request-time gates must not crash when the windowed history is
tz-naive (as `_cgm_history_from_events` localizes it) while the resolved cutoff is tz-aware
(as a "Generate now" request produces). Before the fix these raised "Cannot compare/subtract
tz-naive and tz-aware", crashing every non-abstaining forecast request.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.prediction.service import staleness_minutes, window_to_anchor  # noqa: E402


class ServiceTzMixTest(unittest.TestCase):
    def setUp(self):
        # tz-NAIVE history (what the orchestrator hands the service) ...
        self.hist = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2026-07-16 12:00", "2026-07-16 12:30", "2026-07-16 13:00"]),
            "glucose": [110.0, 120.0, 130.0],
        })
        # ... vs a tz-AWARE cutoff (a resolved "Generate now")
        self.cutoff = "2026-07-16T13:05:00+00:00"

    def test_staleness_minutes_mixed_tz(self):
        stale = staleness_minutes(self.hist, self.cutoff, "timestamp")
        self.assertIsNotNone(stale)
        self.assertAlmostEqual(stale, 5.0, places=3)   # 13:05 − 13:00

    def test_window_to_anchor_mixed_tz(self):
        win = window_to_anchor(self.hist, self.cutoff, time_col="timestamp", history_minutes=45)
        # anchor 13:05, 45-min window keeps 12:30 and 13:00 (12:00 is outside)
        self.assertEqual(len(win), 2)

    def test_both_naive_still_work(self):
        self.assertAlmostEqual(
            staleness_minutes(self.hist, "2026-07-16 13:00", "timestamp"), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
