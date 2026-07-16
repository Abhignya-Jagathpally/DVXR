"""PR20 / Gate B (spec §3, §8, §9): SCIENTIFIC VALIDITY of the CGM-only baseline.

Three defects the review flagged, each pinned by a test:
  1. same-data calibration  — the Platt layer must NEVER be fit on the training rows. If a horizon
     has no valid held-out calibration slice, its head is SKIPPED, not silently self-calibrated.
  2. single-observation prediction — a number requires ADEQUATE history (span, point count, cadence),
     not one CGM reading.
  3. decision-margin ≠ reliability — distance from the 0.5 threshold is NOT trustworthiness; an
     out-of-distribution or thinly-sampled input must lower reliability even when the probability is
     far from 0.5, and reliability (not the margin) is what gates escalation.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.prediction import (  # noqa: E402
    AdequacyConfig, CgmOnlyExcursionService, PredictionInputs)
from dvxr.safety.policy import select_action  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels, history_slice  # noqa: E402


def _synthetic_cgm(n_subjects=12, n=200, freq="15min"):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq=freq)
        noise = rs.normal(0, 12 if high else 8, n)
        vals = np.clip(base + noise + (25 * np.sin(np.arange(n) / 6.0) if high else 0), 55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": vals}))
    return pd.concat(frames, ignore_index=True)


def _fit(**kw):
    cgm = _synthetic_cgm()
    thr = ExcursionThresholds(history_minutes=120)
    anchors = sorted({t for _, g in cgm.groupby("subject_id")
                      for t in pd.to_datetime(g["timestamp"]).iloc[16::6]})
    ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
    return CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr, **kw), cgm, thr


class NoSameDataCalibrationTest(unittest.TestCase):
    def test_horizon_with_no_valid_calibration_slice_is_skipped_not_self_calibrated(self):
        # a single-subject cohort cannot yield a subject-held-out calibration slice ⇒ the head must be
        # SKIPPED entirely, never calibrated on its own training rows (which would be optimistic).
        cgm = _synthetic_cgm(n_subjects=1, n=200)
        thr = ExcursionThresholds(history_minutes=120)
        anchors = sorted(pd.to_datetime(cgm["timestamp"]).iloc[16::6])
        ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
        svc = CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr)
        self.assertEqual(svc._models, {})                 # no head fit without honest calibration
        self.assertTrue(set(svc.skipped_horizons) >= {30, 60})


class AdequateHistoryRequiredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, cls.cgm, cls.thr = _fit()

    def _hist_at(self, sid, idx):
        anchor = pd.to_datetime(self.cgm[self.cgm.subject_id == sid]["timestamp"]).iloc[idx]
        return history_slice(self.cgm, anchor, thresholds=self.thr, subject_col="subject_id",
                             subject_id=sid)

    def test_single_observation_abstains(self):
        one = self._hist_at("s0", 120).tail(1)
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=one,
                                              requested_modalities=["cgm"]))
        self.assertTrue(b.abstained)
        self.assertIn("history", (b.abstain_reason or "").lower())

    def test_short_span_abstains(self):
        # two readings 15 min apart: enough points count is low AND span far below the minimum
        short = self._hist_at("s0", 120).tail(2)
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=short,
                                              requested_modalities=["cgm"]))
        self.assertTrue(b.abstained)

    def test_adequate_history_predicts(self):
        full = self._hist_at("s0", 120)
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30, 60], cgm_history=full,
                                              requested_modalities=["cgm"]))
        self.assertFalse(b.abstained)
        self.assertIsNotNone(b.risk)


class OutOfDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, cls.cgm, cls.thr = _fit()

    def test_wildly_ood_history_abstains(self):
        # a physiologically-implausible sustained ramp (~1.7 mg/dL/min for 2.75h): far outside the
        # near-flat slope distribution the model was trained on ⇒ extrapolation, not a prediction.
        ts = pd.date_range("2020-06-01", periods=12, freq="15min")
        ood = pd.DataFrame({"timestamp": ts, "glucose": np.linspace(40.0, 320.0, 12)})
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=ood,
                                              requested_modalities=["cgm"]))
        self.assertTrue(b.abstained)
        self.assertIn("distribution", (b.abstain_reason or "").lower())


class DecisionMarginVsReliabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, cls.cgm, cls.thr = _fit()

    def test_bundle_reports_margin_and_reliability_separately(self):
        anchor = pd.to_datetime(self.cgm[self.cgm.subject_id == "s0"]["timestamp"]).iloc[120]
        hist = history_slice(self.cgm, anchor, thresholds=self.thr, subject_col="subject_id",
                             subject_id="s0")
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30, 60], cgm_history=hist,
                                              requested_modalities=["cgm"]))
        self.assertIsNotNone(b.decision_margin)
        self.assertIsNotNone(b.reliability)
        self.assertIsNotNone(b.ood_score)
        # decision margin is exactly the distance-from-0.5 signal; reliability is NOT that number
        pmax = max(b.risk.values())
        self.assertAlmostEqual(b.decision_margin, abs(pmax - 0.5) * 2, places=5)
        # confidence surfaced to the policy is reliability, not the raw margin
        self.assertAlmostEqual(b.confidence, b.reliability, places=6)

    def test_low_reliability_does_not_escalate_even_at_high_risk(self):
        # a high-risk prediction with LOW reliability must not trigger escalation (which requires
        # confidence>=0.7). Distance-from-0.5 alone can no longer force an escalation on untrustworthy data.
        d = select_action(abstained=False, risk_category="high", confidence=0.2,
                          data_quality="acceptable", role="clinician")
        self.assertNotEqual(d.action_id, "ESCALATE_PER_APPROVED_PROTOCOL")

    def test_high_reliability_high_risk_escalates(self):
        d = select_action(abstained=False, risk_category="high", confidence=0.9,
                          data_quality="good", role="clinician")
        self.assertEqual(d.action_id, "ESCALATE_PER_APPROVED_PROTOCOL")


if __name__ == "__main__":
    unittest.main()
