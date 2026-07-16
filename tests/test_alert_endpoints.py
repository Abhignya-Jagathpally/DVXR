"""PR39: the alert lifecycle the UI advertises (acknowledge / dismiss / escalate) has real endpoints.

An alert is keyed by its prediction_id, tenant+patient scoped, and every state change is audited with an
append-only history. Escalation is never blocked and pins requires_clinician_review; dismiss is a
clinical judgement (clinician/admin only). Cross-tenant / unknown ids are 404 (non-disclosure).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.storage import open_local_stores  # noqa: E402


class AlertStoreTest(unittest.TestCase):
    def setUp(self):
        self.stores = open_local_stores(":memory:")
        self.alerts = self.stores.alerts

    def test_ensure_is_idempotent_open(self):
        a = self.alerts.ensure(alert_id="pred1", tenant_id="t1", patient_id="P1", prediction_id="pred1")
        self.assertEqual(a["state"], "open")
        again = self.alerts.ensure(alert_id="pred1", tenant_id="t1", patient_id="P1",
                                   prediction_id="pred1")
        self.assertEqual(again["history"], a["history"])            # not recreated

    def test_transition_appends_history_and_escalate_pins_review(self):
        self.alerts.ensure(alert_id="p", tenant_id="t1", patient_id="P1", prediction_id="p")
        ack = self.alerts.transition("p", tenant_id="t1", op="acknowledge", actor_id="u1")
        self.assertEqual(ack["state"], "acknowledged")
        esc = self.alerts.transition("p", tenant_id="t1", op="escalate", actor_id="u2", note="urgent")
        self.assertEqual(esc["state"], "escalated")
        self.assertTrue(esc["requires_clinician_review"])
        self.assertEqual([h["op"] for h in esc["history"]], ["acknowledge", "escalate"])

    def test_transition_is_tenant_scoped(self):
        self.alerts.ensure(alert_id="p", tenant_id="t1", patient_id="P1", prediction_id="p")
        self.assertIsNone(self.alerts.transition("p", tenant_id="OTHER", op="acknowledge", actor_id="x"))

    def test_unknown_op_raises(self):
        self.alerts.ensure(alert_id="p", tenant_id="t1", patient_id="P1", prediction_id="p")
        with self.assertRaises(ValueError):
            self.alerts.transition("p", tenant_id="t1", op="bogus", actor_id="x")


def _have_starlette():
    try:
        import starlette.testclient  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_starlette(), "starlette not installed (api extra)")
class AlertEndpointTest(unittest.TestCase):
    def _client(self):
        from starlette.testclient import TestClient
        from dvxr.serve.api import create_app
        return TestClient(create_app(require_consent=False, unsafe_dev=True))

    def _prediction_id(self, c):
        r = c.post("/v1/risk-reports", json={"patient_id": "P1", "report_type": "stress_glucose_risk"})
        return r.json()["prediction_id"]

    def test_alert_lifecycle_over_http(self):
        c = self._client()
        pid = self._prediction_id(c)
        # GET materializes an open alert
        got = c.get(f"/v1/alerts/{pid}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["alert"]["state"], "open")
        # acknowledge (dev principal is a researcher — allowed)
        ack = c.post(f"/v1/alerts/{pid}/acknowledge", json={"note": "seen"})
        self.assertEqual(ack.status_code, 200)
        self.assertEqual(ack.json()["alert"]["state"], "acknowledged")
        # escalate — never blocked, pins clinician review
        esc = c.post(f"/v1/alerts/{pid}/escalate", json={})
        self.assertEqual(esc.status_code, 200)
        self.assertEqual(esc.json()["alert"]["state"], "escalated")
        self.assertTrue(esc.json()["alert"]["requires_clinician_review"])

    def test_dismiss_forbidden_for_researcher(self):
        c = self._client()
        pid = self._prediction_id(c)
        r = c.post(f"/v1/alerts/{pid}/dismiss", json={})
        self.assertEqual(r.status_code, 403)                       # dismiss is clinician/admin only

    def test_unknown_alert_is_404(self):
        self.assertEqual(self._client().get("/v1/alerts/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
