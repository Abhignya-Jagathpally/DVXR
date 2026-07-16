"""PR10 / Gate 1: the prospective glucose-excursion target is causal, deterministic, censored, and
carries threshold-version provenance (spec §3, §5, §6)."""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.targets import (  # noqa: E402
    ExcursionExample,
    ExcursionThresholds,
    build_excursion_labels,
    history_slice,
)


def _timeline(values, start="2020-01-01 00:00:00", step_min=5, subject="s1"):
    ts = pd.date_range(start=start, periods=len(values), freq=f"{step_min}min")
    return pd.DataFrame({"subject_id": subject, "timestamp": ts, "glucose": values})


class ExcursionTargetTest(unittest.TestCase):
    def test_future_hyperglycaemia_is_labelled_positive(self):
        # flat 120 for the history, then a spike to 200 within 30 min of the anchor
        vals = [120] * 12 + [130, 150, 200, 160, 140, 120]
        df = _timeline(vals)
        anchor = df["timestamp"].iloc[11]        # cutoff right before the spike
        thr = ExcursionThresholds(horizons_minutes=(30,), history_minutes=60)
        out = build_excursion_labels(df, thresholds=thr, anchors=[anchor],
                                     subject_col="subject_id")
        row = out.iloc[0]
        self.assertEqual(row["label"], 1)
        self.assertFalse(row["censored"])
        self.assertIsNotNone(row["first_excursion_time"])

    def test_flat_future_is_labelled_negative(self):
        df = _timeline([120] * 30)
        anchor = df["timestamp"].iloc[12]
        thr = ExcursionThresholds(horizons_minutes=(30,), history_minutes=60)
        out = build_excursion_labels(df, thresholds=thr, anchors=[anchor], subject_col="subject_id")
        self.assertEqual(out.iloc[0]["label"], 0)
        self.assertFalse(out.iloc[0]["censored"])

    def test_missing_future_is_censored_not_zero(self):
        # history present, but the future window has NO samples near t+30 (a CGM gap) → censored
        df = _timeline([120] * 13)              # ends right at the anchor; nothing after
        anchor = df["timestamp"].iloc[12]
        thr = ExcursionThresholds(horizons_minutes=(30,), history_minutes=60,
                                  target_tolerance_minutes=5)
        out = build_excursion_labels(df, thresholds=thr, anchors=[anchor], subject_col="subject_id")
        row = out.iloc[0]
        self.assertTrue(row["censored"])
        self.assertTrue(pd.isna(row["label"]))
        self.assertEqual(row["censor_reason"], "no_future_samples")

    def test_feature_window_never_overlaps_the_future(self):
        # the causal guarantee: feature_window_end == anchor == target_window_start (exclusive future)
        df = _timeline([120] * 40)
        out = build_excursion_labels(df, thresholds=ExcursionThresholds(horizons_minutes=(30, 60)),
                                     subject_col="subject_id")
        self.assertTrue((out["feature_window_end"] == out["target_window_start"]).all())
        self.assertTrue((out["feature_window_start"] < out["feature_window_end"]).all()
                        | (out["n_history_samples"] >= 1).all())

    def test_history_slice_excludes_future_samples(self):
        df = _timeline([100 + i for i in range(40)])
        anchor = df["timestamp"].iloc[20]
        sl = history_slice(df, anchor, thresholds=ExcursionThresholds(history_minutes=60),
                           subject_col="subject_id", subject_id="s1")
        self.assertTrue((sl["timestamp"] <= anchor).all())
        self.assertGreater(len(sl), 0)

    def test_deterministic_and_carries_threshold_version(self):
        df = _timeline([80, 90, 200, 120, 60, 121, 122, 123, 124, 125, 130, 140, 150, 160, 170])
        thr = ExcursionThresholds(version="pilot-v9", horizons_minutes=(30,), history_minutes=30)
        a = build_excursion_labels(df, thresholds=thr, subject_col="subject_id")
        b = build_excursion_labels(df, thresholds=thr, subject_col="subject_id")
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a["threshold_version"] == "pilot-v9").all())

    def test_hypoglycaemia_counts_as_excursion(self):
        vals = [120] * 12 + [110, 90, 65, 100, 110, 120]   # dips below 70, future covers t+30
        df = _timeline(vals)
        anchor = df["timestamp"].iloc[11]
        thr = ExcursionThresholds(horizons_minutes=(30,), history_minutes=60)
        out = build_excursion_labels(df, thresholds=thr, anchors=[anchor], subject_col="subject_id")
        self.assertEqual(out.iloc[0]["label"], 1)

    def test_multi_subject_labels_are_isolated(self):
        s1 = _timeline([120] * 20, subject="s1")
        s2 = _timeline([120] * 12 + [200] * 8, subject="s2")
        df = pd.concat([s1, s2], ignore_index=True)
        out = build_excursion_labels(df, thresholds=ExcursionThresholds(horizons_minutes=(30,),
                                     history_minutes=60), subject_col="subject_id")
        self.assertEqual(set(out["subject_id"]), {"s1", "s2"})
        # s1 never excurses; s2 does
        self.assertTrue((out[out.subject_id == "s1"]["label"].fillna(0) == 0).all())
        self.assertTrue((out[out.subject_id == "s2"]["label"] == 1).any())


if __name__ == "__main__":
    unittest.main()
