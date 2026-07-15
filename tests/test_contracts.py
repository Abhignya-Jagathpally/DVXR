"""PR2: the Generate-lifecycle contracts (spec §2, §6) round-trip and carry reproducible ids/versions."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import (  # noqa: E402
    ActionDecision,
    GenerateRequest,
    ModelEvidence,
    PatientSnapshot,
    RiskPrediction,
)


class ContractsTest(unittest.TestCase):
    def test_request_id_is_deterministic(self):
        r = GenerateRequest(patient_id="P1", idempotency_key="k1").with_request_id()
        r2 = GenerateRequest(patient_id="P1", idempotency_key="k1").with_request_id()
        self.assertTrue(r.request_id)
        self.assertEqual(r.request_id, r2.request_id)

    def test_request_round_trips(self):
        r = GenerateRequest(patient_id="P1", question="why?").with_request_id()
        self.assertEqual(GenerateRequest.from_dict(r.to_dict()), r)

    def test_prediction_carries_versions_and_id(self):
        p = RiskPrediction(request_id="req_1", patient_id="P1", report_type="stress_glucose_risk",
                           risk={"excursion_30m": 0.5}, model_version="m1",
                           feature_version="f1", data_cutoff_at="2024-01-01T00:00:00Z").with_prediction_id()
        self.assertTrue(p.prediction_id)
        self.assertEqual(p.model_version, "m1")
        self.assertEqual(RiskPrediction.from_dict(p.to_dict()), p)

    def test_abstained_prediction_has_no_risk_number(self):
        p = RiskPrediction(request_id="r", patient_id="P1", report_type="stress_glucose_risk",
                           abstained=True, abstain_reason="no synchronized data").with_prediction_id()
        self.assertIsNone(p.risk)
        self.assertTrue(p.abstained)

    def test_prediction_is_immutable(self):
        p = RiskPrediction(request_id="r", patient_id="P1", report_type="t")
        with self.assertRaises(Exception):
            p.risk = {"x": 1.0}          # frozen dataclass — the LLM can never mutate the number

    def test_evidence_and_action_round_trip(self):
        e = ModelEvidence(prediction_id="pred_1", contributions={"cgm": 0.6, "eeg": 0.1})
        self.assertEqual(ModelEvidence.from_dict(e.to_dict()), e)
        a = ActionDecision(action_id="VERIFY_SENSOR_AND_CGM", policy_id="P", policy_version="1.0",
                           reason_codes=["elevated_predicted_risk"])
        self.assertEqual(ActionDecision.from_dict(a.to_dict()), a)

    def test_snapshot_id_reproducible_from_events(self):
        s = PatientSnapshot(patient_id="P1", data_cutoff_at="2024-01-01T00:00:00Z",
                            event_ids=["b", "a"]).with_snapshot_id()
        s2 = PatientSnapshot(patient_id="P1", data_cutoff_at="2024-01-01T00:00:00Z",
                             event_ids=["a", "b"]).with_snapshot_id()
        self.assertEqual(s.snapshot_id, s2.snapshot_id)   # order-independent event set


if __name__ == "__main__":
    unittest.main()
