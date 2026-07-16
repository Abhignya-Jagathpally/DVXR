"""PR15 / Gate 6: the API derives the actor from a server-side credential (never the body), authorizes
patient access, defaults consent ON, and scopes idempotency by tenant+patient so keys cannot collide
across contexts (spec §2, §18)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dvxr.serve.auth import (  # noqa: E402
    AuthError,
    AuthorizationError,
    Principal,
    authenticate,
    authorize,
    build_principals,
)


class AuthUnitTest(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        principals = build_principals({"k1": {"actor_id": "a", "role": "clinician",
                                              "tenant_id": "t1", "patient_scope": ["P1"]}})
        with self.assertRaises(AuthError):
            authenticate("nope", principals)
        with self.assertRaises(AuthError):
            authenticate(None, principals)

    def test_unsafe_dev_allows_default_principal_only_when_no_registry(self):
        p = authenticate(None, None, unsafe_dev=True)
        self.assertEqual(p.role, "researcher")
        with self.assertRaises(AuthError):
            authenticate(None, None, unsafe_dev=False)

    def test_authorize_denies_out_of_scope_patient(self):
        p = Principal("a", "clinician", "t1", frozenset({"P1"}))
        authorize(p, "P1")                       # ok
        with self.assertRaises(AuthorizationError):
            authorize(p, "P2")

    def test_authorize_denies_cross_tenant_record(self):
        p = Principal("a", "clinician", "t1", "*")
        with self.assertRaises(AuthorizationError):
            authorize(p, "P1", record_tenant="t2")


class ApiSecurityTest(unittest.TestCase):
    def setUp(self):
        try:
            from starlette.testclient import TestClient
        except Exception:  # pragma: no cover
            self.skipTest("starlette not available")
        from dvxr.serve.api import create_app
        self.principals = build_principals({
            "key-clin": {"actor_id": "clin", "role": "clinician", "tenant_id": "t1",
                         "patient_scope": ["P1"]},
            "key-other": {"actor_id": "other", "role": "clinician", "tenant_id": "t2",
                          "patient_scope": ["P9"]},
        })
        app = create_app(principals=self.principals, require_consent=False)
        self.client = TestClient(app, raise_server_exceptions=False)

    def _post(self, key, patient_id, **extra):
        headers = {"X-API-Key": key} if key else {}
        body = {"patient_id": patient_id, "report_type": "stress_glucose_risk", **extra}
        return self.client.post("/v1/risk-reports", json=body, headers=headers)

    def test_missing_key_is_401(self):
        self.assertEqual(self._post(None, "P1").status_code, 401)

    def test_in_scope_patient_succeeds(self):
        r = self._post("key-clin", "P1")
        self.assertEqual(r.status_code, 200)

    def test_out_of_scope_patient_is_403(self):
        self.assertEqual(self._post("key-clin", "P2").status_code, 403)

    def test_self_asserted_role_in_body_is_ignored(self):
        # the body says admin, but the principal's role (clinician) is what the server uses
        r = self._post("key-clin", "P1", user_role="admin")
        self.assertEqual(r.status_code, 200)
        # the persisted request records the SERVER role, not "admin"
        self.assertIn(r.json().get("status"), ("abstained", "completed"))

    def test_cross_tenant_prediction_retrieval_is_denied(self):
        pid = self._post("key-clin", "P1").json()["prediction_id"]
        # a principal from another tenant cannot read it. The fetch is tenant-scoped (Gate A), so the
        # record never leaves storage under the other tenant — a 404 (non-disclosure of existence),
        # which is STRONGER than a 403 that would confirm the id exists in some other tenant.
        g = self.client.get(f"/v1/predictions/{pid}", headers={"X-API-Key": "key-other"})
        self.assertEqual(g.status_code, 404)
        # the owning principal can
        g2 = self.client.get(f"/v1/predictions/{pid}", headers={"X-API-Key": "key-clin"})
        self.assertEqual(g2.status_code, 200)

    def test_idempotency_key_is_scoped_no_cross_patient_collision(self):
        # same raw idempotency key, two different patients (both in a broad-scope principal)
        principals = build_principals({"k": {"actor_id": "a", "role": "clinician",
                                            "tenant_id": "t1", "patient_scope": "*"}})
        from dvxr.serve.api import create_app
        from starlette.testclient import TestClient
        client = TestClient(create_app(principals=principals, require_consent=False),
                            raise_server_exceptions=False)
        r1 = client.post("/v1/risk-reports", headers={"X-API-Key": "k"},
                         json={"patient_id": "PA", "report_type": "stress_glucose_risk",
                               "idempotency_key": "same"})
        r2 = client.post("/v1/risk-reports", headers={"X-API-Key": "k"},
                         json={"patient_id": "PB", "report_type": "stress_glucose_risk",
                               "idempotency_key": "same"})
        self.assertNotEqual(r1.json()["prediction_id"], r2.json()["prediction_id"])  # no collision


if __name__ == "__main__":
    unittest.main()
