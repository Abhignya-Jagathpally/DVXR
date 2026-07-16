"""PR13 / Gate 4: on the reportable path the missingness mask actually REACHES the model matrix, a
stale CGM feed forces abstention, and personalization requires an explicit cutoff and reports its
status honestly (spec §5, §7, §8)."""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.features import (  # noqa: E402
    MISSING_MASK_SUFFIX,
    build_reportable_features,
    feature_columns,
)
from dvxr.personalization import SubjectBaselineNormalizer, per_subject_normalize  # noqa: E402
from dvxr.prediction import CgmOnlyExcursionService, PredictionInputs  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels, history_slice  # noqa: E402


class MaskReachesModelTest(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "subject_id": ["s", "s", "s"],
            "cgm_mean": [0.0, np.nan, 120.0],       # genuine 0, missing, normal
            "target": ["high", "low", "high"],
        })

    def test_reportable_features_feed_the_mask_to_the_model(self):
        rep = build_reportable_features(self._frame(), ["cgm_mean"])
        cols = feature_columns(rep, include_masks=True)
        self.assertIn("cgm_mean", cols)
        self.assertIn("cgm_mean" + MISSING_MASK_SUFFIX, cols)   # mask IS a model feature here
        # the missing row's value is filled to 0.0 but its mask is 0.0 — distinct from the genuine 0
        self.assertEqual(rep["cgm_mean"].tolist(), [0.0, 0.0, 120.0])
        self.assertEqual(rep["cgm_mean" + MISSING_MASK_SUFFIX].tolist(), [1.0, 0.0, 1.0])

    def test_default_feature_columns_still_exclude_masks(self):
        rep = build_reportable_features(self._frame(), ["cgm_mean"])
        self.assertNotIn("cgm_mean" + MISSING_MASK_SUFFIX, feature_columns(rep))  # legacy unchanged


def _synthetic_cgm(n=180):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(10):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        vals = np.clip(base + rs.normal(0, 12, n) + (25 * np.sin(np.arange(n) / 6.0) if high else 0),
                       55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": vals}))
    return pd.concat(frames, ignore_index=True)


class StaleCgmForcesAbstentionTest(unittest.TestCase):
    def test_stale_feed_abstains(self):
        cgm = _synthetic_cgm()
        thr = ExcursionThresholds(history_minutes=120)
        anchors = sorted({t for _, g in cgm.groupby("subject_id")
                          for t in pd.to_datetime(g["timestamp"]).iloc[16::6]})
        ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
        svc = CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr, max_staleness_minutes=30)
        anchor = pd.to_datetime(cgm[cgm.subject_id == "s0"]["timestamp"]).iloc[120]
        hist = history_slice(cgm, anchor, thresholds=thr, subject_col="subject_id", subject_id="s0")
        # cutoff is 5 HOURS after the last history sample → stale
        stale_cutoff = (anchor + pd.Timedelta(hours=5)).isoformat()
        b = svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=hist,
                                         requested_modalities=["cgm"], cutoff=stale_cutoff))
        self.assertTrue(b.abstained)
        self.assertIn("stale", b.abstain_reason.lower())
        # a fresh cutoff (at the anchor) predicts
        fresh = svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=hist,
                                             requested_modalities=["cgm"], cutoff=anchor.isoformat()))
        self.assertFalse(fresh.abstained)


class PersonalizationCutoffTest(unittest.TestCase):
    def _frame(self):
        t = pd.date_range("2020-01-01", periods=8, freq="h")
        return pd.DataFrame({"subject_id": ["a"] * 4 + ["b"] * 4, "timestamp": list(t[:4]) * 2,
                             "x": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]})

    def test_strict_requires_cutoff(self):
        norm = SubjectBaselineNormalizer()
        with self.assertRaises(ValueError):
            norm.fit(self._frame(), ["x"], time_col="timestamp", baseline_cutoff=None, strict=True)

    def test_status_reports_personalized_vs_fallback(self):
        f = self._frame()
        cutoff = f["timestamp"].iloc[1]                 # only 2 baseline rows per subject
        norm = SubjectBaselineNormalizer().fit(f, ["x"], time_col="timestamp",
                                               baseline_cutoff=cutoff, strict=True,
                                               min_baseline_samples=2)
        st = norm.personalization_status("a")
        self.assertEqual(st["personalization_status"], "subject_specific")
        self.assertFalse(st["fallback_used"])
        # a subject below the baseline floor falls back and is flagged
        norm2 = SubjectBaselineNormalizer().fit(f, ["x"], time_col="timestamp",
                                                baseline_cutoff=cutoff, strict=True,
                                                min_baseline_samples=5)
        self.assertTrue(norm2.personalization_status("a")["fallback_used"])

    def test_legacy_whole_subject_normalizer_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            per_subject_normalize(self._frame(), ["x"])
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))


if __name__ == "__main__":
    unittest.main()
