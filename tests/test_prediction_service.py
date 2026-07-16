"""PR12 / Gate 3: the RiskPredictionService boundary. The fused product abstains by default; the
CGM-only single-modality baseline predicts calibrated 30/60 probabilities but abstains for any request
that needs a modality it does not cover (so it can never masquerade as the fused headline)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.prediction import (  # noqa: E402
    AbstainingPredictionService,
    CgmOnlyExcursionService,
    PredictionInputs,
    RiskPredictionService,
)
from dvxr.serve.orchestrate import generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels, history_slice  # noqa: E402


def _synthetic_cgm():
    """Deterministic multi-subject CGM: 'high' subjects excurse (>180), 'normal' stay in range."""
    rs = np.random.RandomState(0)
    frames = []
    for i in range(12):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        n = 180
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        noise = rs.normal(0, 12 if high else 8, n)
        vals = np.clip(base + noise + (25 * np.sin(np.arange(n) / 6.0) if high else 0), 55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": vals}))
    return pd.concat(frames, ignore_index=True)


def _fit_service():
    cgm = _synthetic_cgm()
    thr = ExcursionThresholds(history_minutes=120)
    anchors = sorted({t for _, g in cgm.groupby("subject_id")
                      for t in pd.to_datetime(g["timestamp"]).iloc[16::6]})
    ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
    return CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr), cgm, thr


class AbstainingServiceTest(unittest.TestCase):
    def test_conforms_to_protocol_and_always_abstains(self):
        svc = AbstainingPredictionService()
        self.assertIsInstance(svc, RiskPredictionService)
        b = svc.predict(PredictionInputs("stress_glucose_risk", [30, 60]))
        self.assertTrue(b.abstained)
        self.assertIsNone(b.risk)
        self.assertEqual(b.modality_scope, "fused_gated")


class CgmOnlyServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc, cls.cgm, cls.thr = _fit_service()

    def test_fits_calibrated_heads_for_both_horizons(self):
        self.assertEqual(sorted(self.svc._models.keys()), [30, 60])
        self.assertTrue(self.svc.calibration_version.startswith("platt/"))

    def test_predicts_calibrated_probabilities_for_cgm_only(self):
        anchor = pd.to_datetime(self.cgm[self.cgm.subject_id == "s0"]["timestamp"]).iloc[120]
        hist = history_slice(self.cgm, anchor, thresholds=self.thr, subject_col="subject_id",
                             subject_id="s0")
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30, 60], cgm_history=hist,
                                              requested_modalities=["cgm"]))
        self.assertFalse(b.abstained)
        self.assertEqual(b.modality_scope, "cgm_only")
        for k in ("excursion_30m", "excursion_60m"):
            self.assertIn(k, b.risk)
            self.assertTrue(0.0 <= b.risk[k] <= 1.0)

    def test_abstains_when_request_needs_more_than_cgm(self):
        anchor = pd.to_datetime(self.cgm[self.cgm.subject_id == "s0"]["timestamp"]).iloc[120]
        hist = history_slice(self.cgm, anchor, thresholds=self.thr, subject_col="subject_id",
                             subject_id="s0")
        b = self.svc.predict(PredictionInputs("stress_glucose_risk", [30, 60], cgm_history=hist,
                                              requested_modalities=["cgm", "eeg", "wearable_phys"]))
        self.assertTrue(b.abstained)                      # never masquerades as the fused claim
        self.assertEqual(b.modality_scope, "fused_gated")

    def test_abstains_on_empty_history(self):
        b = self.svc.predict(PredictionInputs("cgm_glucose_risk", [30], cgm_history=None,
                                              requested_modalities=["cgm"]))
        self.assertTrue(b.abstained)


class OrchestratorPredictorTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def test_default_generate_abstains(self):
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent)
        self.assertEqual(out["status"], "abstained")
        self.assertIsNone(out["prediction"]["risk"])

    def test_injected_cgm_service_produces_a_number_for_cgm_request(self):
        svc, cgm, thr = _fit_service()
        # feed causal CGM events for one subject as the request's event stream
        sub = cgm[cgm.subject_id == "s0"].copy()
        cutoff = pd.to_datetime(sub["timestamp"]).iloc[120].isoformat()
        events = [{"event_id": f"e{i}", "patient_id": "P1", "modality": "cgm",
                   "observed_at_utc": r["timestamp"].isoformat(), "value": float(r["glucose"]),
                   "quality_score": 0.9}
                  for i, r in sub.iloc[:130].iterrows()]
        req = GenerateRequest(patient_id="P1", report_type="cgm_glucose_risk",
                              user_role="researcher", data_cutoff_at=cutoff,
                              prediction_horizons_minutes=[30, 60])
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent, events=events, predictor=svc)
        self.assertEqual(out["status"], "completed")
        self.assertIsNotNone(out["prediction"]["risk"])
        self.assertEqual(out["prediction"]["model_version"], "cgm-only/pilot-v1")
        self.assertTrue(out["prediction"]["snapshot_id"])

    def test_injected_cgm_service_still_abstains_for_fused_report(self):
        svc, cgm, thr = _fit_service()
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent, predictor=svc)
        self.assertEqual(out["status"], "abstained")     # fused headline stays gated


if __name__ == "__main__":
    unittest.main()
