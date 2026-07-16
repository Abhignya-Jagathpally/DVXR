"""PR4: the causal cutoff excludes future events, and leak-safe personalization fits baseline stats
on past-only data (spec §7)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.features import causal_cutoff_filter  # noqa: E402
from dvxr.personalization import SubjectBaselineNormalizer  # noqa: E402


class CausalCutoffTest(unittest.TestCase):
    def _events(self):
        return pd.DataFrame({
            "timestamp_utc": pd.to_datetime([
                "2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z",
                "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
            "subject_id": ["s", "s", "s", "s"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

    def test_future_events_are_excluded(self):
        kept = causal_cutoff_filter(self._events(), "2024-01-01T01:00:00Z")
        self.assertEqual(kept["value"].tolist(), [1.0, 2.0, 3.0])   # 02:00 excluded

    def test_cutoff_is_inclusive_of_the_boundary(self):
        kept = causal_cutoff_filter(self._events(), "2024-01-01T00:30:00Z")
        self.assertEqual(len(kept), 2)


class BaselineNormalizerTest(unittest.TestCase):
    def test_fit_uses_baseline_slice_only(self):
        # subject's baseline (t<=cutoff) has mean 0; a later spike must NOT shift the baseline stats
        frame = pd.DataFrame({
            "subject_id": ["s"] * 5,
            "t": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z",
                                 "2024-01-01T00:20:00Z", "2024-01-01T05:00:00Z",
                                 "2024-01-01T06:00:00Z"], utc=True),
            "hr": [60.0, 60.0, 60.0, 200.0, 220.0],
        })
        norm = SubjectBaselineNormalizer().fit(
            frame, ["hr"], time_col="t", baseline_cutoff=pd.Timestamp("2024-01-01T00:30:00Z"))
        # baseline mean is 60 (std 0) -> the constant baseline maps to 0; the future spikes are large
        out = norm.transform(frame)
        self.assertEqual(out["hr"].iloc[0], 0.0)          # baseline value normalizes to 0
        self.assertGreater(out["hr"].iloc[3], 5.0)        # future spike is far above baseline
        self.assertEqual(str(norm.baseline_cutoff), "2024-01-01 00:30:00+00:00")

    def test_unseen_subject_falls_back_to_pooled(self):
        frame = pd.DataFrame({"subject_id": ["a", "a", "b"], "hr": [10.0, 20.0, 15.0]})
        norm = SubjectBaselineNormalizer().fit(frame, ["hr"])
        out = norm.transform(pd.DataFrame({"subject_id": ["c"], "hr": [15.0]}))
        self.assertTrue(np.isfinite(out["hr"].iloc[0]))   # unknown subject handled via pooled stats


if __name__ == "__main__":
    unittest.main()
