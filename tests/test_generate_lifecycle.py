"""PR6: the Generate lifecycle (spec §2) never trains during a request, is idempotent, persists an
audited reproducible request, and abstains for the research-stage glucose product."""
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.orchestrate import ConsentError, generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


class GenerateLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def _req(self, **ov):
        base = dict(patient_id="P1", report_type="stress_glucose_risk", user_role="researcher")
        base.update(ov)
        return GenerateRequest(**base)

    def test_model_is_never_trained_during_request(self):
        import dvxr.serve.screener as screener
        calls = {"n": 0}
        orig = screener.fit_screener

        def _spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        screener.fit_screener = _spy
        try:
            out = generate_risk_report(self._req(), prediction_store=self.pred,
                                       audit_store=self.audit, consent_store=self.consent)
        finally:
            screener.fit_screener = orig
        self.assertEqual(calls["n"], 0, "Generate must not train a model during the request")
        self.assertEqual(out["status"], "abstained")     # research-stage glucose product abstains

    def test_generate_request_is_idempotent(self):
        r1 = generate_risk_report(self._req(idempotency_key="k1"), prediction_store=self.pred,
                                  audit_store=self.audit, consent_store=self.consent)
        r2 = generate_risk_report(self._req(idempotency_key="k1"), prediction_store=self.pred,
                                  audit_store=self.audit, consent_store=self.consent)
        self.assertEqual(r1["prediction"]["prediction_id"], r2["prediction"]["prediction_id"])
        self.assertTrue(r2["reused"])

    def test_reused_report_has_the_same_shape_as_a_fresh_one(self):
        # Regression: the idempotent-reuse path must return the SAME keys as the fresh path — a
        # downstream consumer (e.g. build_report_panels) reads report["action"]/["explanation"]
        # and must not KeyError just because the prediction was served from the idempotency cache.
        fresh = generate_risk_report(self._req(idempotency_key="same"), prediction_store=self.pred,
                                     audit_store=self.audit, consent_store=self.consent)
        reused = generate_risk_report(self._req(idempotency_key="same"), prediction_store=self.pred,
                                      audit_store=self.audit, consent_store=self.consent)
        self.assertTrue(reused["reused"])
        self.assertEqual(set(fresh.keys()), set(reused.keys()))
        for key in ("action", "explanation", "prediction", "model_version", "status"):
            self.assertIn(key, reused)
        self.assertEqual(fresh["action"]["action_id"], reused["action"]["action_id"])

    def test_panels_render_on_a_reused_idempotent_request(self):
        from dvxr.serve.panels import build_report_panels
        req = self._req(idempotency_key="dup")
        generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                             consent_store=self.consent)              # prime the idempotency cache
        out = build_report_panels(req, prediction_store=self.pred, audit_store=self.audit,
                                  consent_store=self.consent, require_consent=True)
        self.assertEqual(out["status"], "abstained")
        self.assertEqual(out["panels"]["next_action"]["action_id"], "INSUFFICIENT_DATA")

    def test_abstention_carries_no_risk_number(self):
        out = generate_risk_report(self._req(), prediction_store=self.pred,
                                   audit_store=self.audit, consent_store=self.consent)
        self.assertIsNone(out["prediction"]["risk"])
        self.assertEqual(out["action"]["action_id"], "INSUFFICIENT_DATA")

    def test_request_is_audited(self):
        out = generate_risk_report(self._req(), prediction_store=self.pred,
                                   audit_store=self.audit, consent_store=self.consent)
        events = [e["event"] for e in self.audit.for_request(out["request_id"])]
        self.assertIn("generate.requested", events)
        self.assertIn("generate.completed", events)

    def test_consent_is_fail_closed(self):
        # a no-key request fingerprints on the RESOLVED cutoff, so inject a fixed clock to know its id
        clock = lambda: datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
        with self.assertRaises(ConsentError):
            generate_risk_report(self._req(patient_id="P-unknown"), prediction_store=self.pred,
                                 audit_store=self.audit, consent_store=self.consent, clock=clock)
        # the denial is audited under the request_id derived from the resolved cutoff
        cutoff = clock().astimezone(timezone.utc).isoformat()
        rid = GenerateRequest(patient_id="P-unknown", report_type="stress_glucose_risk",
                              user_role="researcher", data_cutoff_at=cutoff).with_request_id().request_id
        denied = self.audit.for_request(rid)
        self.assertTrue(any(e["event"] == "generate.denied.consent" for e in denied))

    def test_prediction_is_retrievable(self):
        out = generate_risk_report(self._req(), prediction_store=self.pred,
                                   audit_store=self.audit, consent_store=self.consent)
        self.assertEqual(self.pred.get(out["prediction_id"])["patient_id"], "P1")


def _have_starlette():
    try:
        import starlette.testclient  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_starlette(), "starlette not installed (api extra)")
class RiskReportsEndpointTest(unittest.TestCase):
    def _client(self):
        from starlette.testclient import TestClient
        from dvxr.serve.api import create_app
        # local research deployment: no auth registry ⇒ explicit unsafe_dev, consent off
        return TestClient(create_app(require_consent=False, unsafe_dev=True))

    def test_post_returns_persisted_abstention(self):
        c = self._client()
        r = c.post("/v1/risk-reports", json={"patient_id": "P1", "report_type": "stress_glucose_risk"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "abstained")
        self.assertIsNone(body["prediction"]["risk"])
        self.assertIn("not a diagnosis", body["disclaimer"].lower())
        # retrievable by id
        got = c.get(f"/v1/predictions/{body['prediction_id']}")
        self.assertEqual(got.status_code, 200)

    def test_post_is_idempotent_over_http(self):
        c = self._client()
        payload = {"patient_id": "P1", "report_type": "stress_glucose_risk", "idempotency_key": "kk"}
        a = c.post("/v1/risk-reports", json=payload).json()
        b = c.post("/v1/risk-reports", json=payload).json()
        self.assertEqual(a["prediction"]["prediction_id"], b["prediction"]["prediction_id"])
        self.assertTrue(b["reused"])

    def test_missing_patient_id_400s(self):
        self.assertEqual(self._client().post("/v1/risk-reports", json={}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
