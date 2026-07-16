"""Regression: the default 'Generate now' cutoff is tz-aware (datetime.now(timezone.utc)), but stored
CGM events are commonly tz-naive. Filtering history must NOT raise 'Cannot compare tz-naive and tz-aware'
— it did, 500ing every cutoff-less request that had stored CGM history (all report types)."""
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.serve.orchestrate import _cgm_history_from_events  # noqa: E402


def _events(tz_suffix):
    # tz_suffix "" -> tz-naive stored timestamps; "+00:00" -> tz-aware
    return [{"modality": "cgm", "tenant_id": "t1", "patient_id": "P1",
             "observed_at_utc": f"2020-01-01T00:{m:02d}:00{tz_suffix}", "value": 120.0 + m}
            for m in range(0, 40, 15)]


class CutoffTzMismatchTest(unittest.TestCase):
    def test_tznaive_events_vs_tzaware_cutoff(self):
        # a resolved "now" cutoff carries tz; the stored events do not
        cutoff = datetime(2020, 1, 2, tzinfo=timezone.utc).isoformat()
        df = _cgm_history_from_events(_events(""), tenant_id="t1", patient_id="P1", cutoff=cutoff)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)                       # all three kept, no raise

    def test_tzaware_events_vs_tznaive_cutoff(self):
        # the mirror: tz-aware stored events, tz-naive cutoff string
        df = _cgm_history_from_events(_events("+00:00"), tenant_id="t1", patient_id="P1",
                                      cutoff="2020-01-02T00:00:00")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)

    def test_cutoff_actually_filters(self):
        # cutoff between samples drops the later ones (correctness, not just no-raise)
        df = _cgm_history_from_events(_events(""), tenant_id="t1", patient_id="P1",
                                      cutoff=datetime(2020, 1, 1, 0, 20, tzinfo=timezone.utc).isoformat())
        self.assertEqual(len(df), 2)                       # 00:00 and 00:15 kept, 00:30 dropped


if __name__ == "__main__":
    unittest.main()
