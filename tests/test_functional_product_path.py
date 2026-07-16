"""PR21 / Gate C (spec §2): the Generate path is FUNCTIONAL end to end, not scaffolding.

  * an EVENT REPOSITORY populates the snapshot (the API no longer always builds an empty one), and it
    is tenant+patient scoped (never reads another identity's events);
  * MODEL EVIDENCE is model-derived, persisted with the prediction, and surfaced in the report;
  * GET /v1/predictions/{id} rebuilds the FULL report (prediction + evidence + action + explanation),
    the same shape POST produced — not a bare prediction row.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.orchestrate import generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


def _cgm_event(patient, tenant, eid, ts, val):
    return {"event_id": eid, "patient_id": patient, "tenant_id": tenant, "modality": "cgm",
            "observed_at_utc": ts, "value": val, "quality_score": 0.9}


class EventRepositoryScopingTest(unittest.TestCase):
    def setUp(self):
        self.stores = open_local_stores(":memory:")
        self.pred, self.audit, self.consent, _m = self.stores
        self.events = self.stores.events
        self.consent.set_scope("P1", {"purposes": ["research"]}, tenant_id="t1")

    def test_window_is_tenant_and_patient_scoped(self):
        self.events.append_events([
            _cgm_event("P1", "t1", "e1", "2026-01-01T00:00:00Z", 120),
            _cgm_event("P1", "t1", "e2", "2026-01-01T00:15:00Z", 130),
            _cgm_event("P2", "t1", "e3", "2026-01-01T00:15:00Z", 999),   # other patient
            _cgm_event("P1", "t2", "e4", "2026-01-01T00:15:00Z", 888),   # other tenant
        ])
        got = self.events.window("P1", None, None, tenant_id="t1")
        self.assertEqual({e["event_id"] for e in got}, {"e1", "e2"})

    def test_generate_builds_a_populated_snapshot_from_the_repository(self):
        self.events.append_events([
            _cgm_event("P1", "t1", "e1", "2026-01-01T00:00:00Z", 120),
            _cgm_event("P1", "t1", "e2", "2026-01-01T00:15:00Z", 130),
        ])
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              tenant_id="t1", user_role="researcher",
                              data_cutoff_at="2026-01-01T01:00:00Z")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent, event_repository=self.events)
        snap = [e for e in self.audit.for_request(out["request_id"])
                if e["event"] == "snapshot.created"][0]["snapshot"]
        self.assertEqual(sorted(snap["event_ids"]), ["e1", "e2"])   # snapshot is REAL, not empty
        self.assertIn("cgm", snap["modalities_present"])

    def test_generate_does_not_read_another_patients_events(self):
        self.events.append_events([_cgm_event("P2", "t1", "e9", "2026-01-01T00:00:00Z", 500)])
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              tenant_id="t1", user_role="researcher",
                              data_cutoff_at="2026-01-01T01:00:00Z")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent, event_repository=self.events)
        snap = [e for e in self.audit.for_request(out["request_id"])
                if e["event"] == "snapshot.created"][0]["snapshot"]
        self.assertEqual(snap["event_ids"], [])                    # P2's event never admitted


class ModelEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def test_report_carries_model_derived_evidence(self):
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent)
        self.assertIn("evidence", out)
        ev = out["evidence"]
        self.assertEqual(ev["prediction_id"], out["prediction_id"])
        # the fused abstention has no attributable contributions, and names its missing modalities
        self.assertEqual(ev["contributions"], {})
        self.assertTrue(ev["missing_data_effects"])

    def test_evidence_is_persisted_with_the_prediction(self):
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent)
        stored = self.pred.get(out["prediction_id"])
        self.assertIn("evidence", stored)
        self.assertEqual(stored["evidence"]["prediction_id"], out["prediction_id"])


def _have_starlette():
    try:
        import starlette.testclient  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_starlette(), "starlette not installed (api extra)")
class GetReturnsFullReportTest(unittest.TestCase):
    def _client(self):
        from starlette.testclient import TestClient
        from dvxr.serve.api import create_app
        return TestClient(create_app(require_consent=False, unsafe_dev=True))

    def test_get_returns_same_shape_as_post(self):
        c = self._client()
        post = c.post("/v1/risk-reports", json={"patient_id": "P1"}).json()
        got = c.get(f"/v1/predictions/{post['prediction_id']}").json()
        for key in ("prediction", "evidence", "action", "explanation", "status", "model_version"):
            self.assertIn(key, got, f"GET must return {key} (full report, not a bare prediction row)")
        self.assertEqual(got["prediction"]["prediction_id"], post["prediction"]["prediction_id"])
        self.assertEqual(got["action"]["action_id"], post["action"]["action_id"])


if __name__ == "__main__":
    unittest.main()
