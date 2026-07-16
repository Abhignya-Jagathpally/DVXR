"""PR23 / P0-2 (spec §8.6): every grounded claim resolves to a source, and a grounding failure degrades
to a safe deterministic explanation — never ungrounded prose, never a 500."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest, PatientSnapshot, RiskPrediction  # noqa: E402
from dvxr.prediction import CgmOnlyExcursionService, build_model_evidence  # noqa: E402
from dvxr.serve.orchestrate import _assemble_report, generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


class EvidenceRecordsTest(unittest.TestCase):
    def test_every_contribution_gets_an_evidence_id(self):
        pred = RiskPrediction(request_id="r", patient_id="P1", report_type="cgm_glucose_risk",
                              abstained=False, risk={"excursion_30m": 0.6}, snapshot_id="snap_x",
                              model_version="cgm-only/pilot-v1", prediction_id="pred_1")
        snap = PatientSnapshot(patient_id="P1", data_cutoff_at="", quality_by_modality={"cgm": 0.9})

        class _B:  # minimal bundle stand-in
            ood_score = 0.3
            reliability = 0.7
            decision_margin = 0.44
            modality_scope = "cgm_only"
        ev = build_model_evidence(pred, snap, _B())
        self.assertEqual(len(ev.evidence_records), 1)
        rec = ev.evidence_records[0]
        self.assertEqual(rec["feature"], "cgm")
        self.assertTrue(rec["evidence_id"].startswith("ev_"))
        self.assertEqual(rec["snapshot_id"], "snap_x")


class GroundingFallbackTest(unittest.TestCase):
    def test_grounding_failure_degrades_to_safe_explanation(self):
        # a prediction whose evidence has a contribution with NO evidence record forces a GroundingError
        # inside grounded_explanation; _assemble_report must catch it and return a safe explanation.
        pred = RiskPrediction(request_id="r", patient_id="P1", report_type="cgm_glucose_risk",
                              abstained=False, risk={"excursion_30m": 0.6},
                              prediction_id="pred_2", model_version="cgm-only/pilot-v1")
        req = GenerateRequest(patient_id="P1", report_type="cgm_glucose_risk", user_role="researcher")
        bad_evidence = {"contributions": {"cgm": 0.5}, "evidence_records": []}   # source-free ⇒ fails
        report = _assemble_report(req, pred, "pred_2", reused=False, evidence=bad_evidence)
        self.assertFalse(report["grounding_complete"])
        self.assertEqual(report["explanation"]["supporting_factors"], [])
        self.assertIn("could not be produced", report["explanation"]["risk_summary"])

    def test_normal_report_is_grounding_complete(self):
        pred, audit, consent, _m = open_local_stores(":memory:")
        consent.set_scope("P1", {"purposes": ["research"]})
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk", user_role="researcher")
        out = generate_risk_report(req, prediction_store=pred, audit_store=audit, consent_store=consent)
        self.assertTrue(out["grounding_complete"])       # abstention path grounds cleanly


if __name__ == "__main__":
    unittest.main()
