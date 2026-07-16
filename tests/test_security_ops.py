"""PR22 / Gate D (spec §2, §18, §23): security & ops hardening.

  * ACTION-level RBAC — authorize() checks the requested action against a role→actions matrix, not just
    patient scope; a read-only role cannot generate. Unknown roles are rejected at construction.
  * AUDIT actor — every generate audit entry records WHO acted (actor_id).
  * canonical-fingerprint idempotency — reusing a key with a DIFFERENT semantic request is a 409
    conflict, never a silently-mismatched cached result.
  * SEPARATE product surface — the Sentinel product API exposes only the /v1 lifecycle routes, never
    the benchmark/screener endpoints.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.auth import (  # noqa: E402
    AuthorizationError, Principal, Role, authorize, build_principals)
from dvxr.serve.orchestrate import (  # noqa: E402
    IdempotencyConflict, generate_risk_report)
from dvxr.storage import open_local_stores  # noqa: E402


class ActionRbacTest(unittest.TestCase):
    def test_participant_may_read_but_not_generate(self):
        p = Principal("a", Role.PARTICIPANT, "t1", "*")
        authorize(p, "P1", "read_prediction")                 # ok
        with self.assertRaises(AuthorizationError):
            authorize(p, "P1", "generate_risk_report")        # read-only role cannot generate

    def test_clinician_may_generate(self):
        authorize(Principal("a", Role.CLINICIAN, "t1", "*"), "P1", "generate_risk_report")

    def test_unknown_action_is_denied(self):
        with self.assertRaises(AuthorizationError):
            authorize(Principal("a", Role.CLINICIAN, "t1", "*"), "P1", "delete_everything")

    def test_unknown_role_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            build_principals({"k": {"actor_id": "a", "role": "wizard", "tenant_id": "t1"}})


class AuditActorTest(unittest.TestCase):
    def test_generate_records_the_actor(self):
        pred, audit, consent, _m = open_local_stores(":memory:")
        consent.set_scope("P1", {"purposes": ["research"]})
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher", actor_id="actor-42")
        out = generate_risk_report(req, prediction_store=pred, audit_store=audit,
                                   consent_store=consent)
        actors = {e.get("actor_id") for e in audit.for_request(out["request_id"])}
        self.assertIn("actor-42", actors)


class IdempotencyConflictTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def _gen(self, **ov):
        base = dict(patient_id="P1", report_type="cgm_glucose_risk", user_role="researcher",
                    idempotency_key="k1", prediction_horizons_minutes=[30])
        base.update(ov)
        return generate_risk_report(GenerateRequest(**base), prediction_store=self.pred,
                                    audit_store=self.audit, consent_store=self.consent)

    def test_same_key_same_request_reuses(self):
        a = self._gen()
        b = self._gen()
        self.assertTrue(b["reused"])
        self.assertEqual(a["prediction_id"], b["prediction_id"])

    def test_same_key_different_request_conflicts(self):
        self._gen(prediction_horizons_minutes=[30])
        with self.assertRaises(IdempotencyConflict):
            self._gen(prediction_horizons_minutes=[30, 60])    # same key, different semantic request


def _have_starlette():
    try:
        import starlette.testclient  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_starlette(), "starlette not installed (api extra)")
class ProductSurfaceTest(unittest.TestCase):
    def test_product_api_exposes_only_v1_routes(self):
        from starlette.testclient import TestClient
        from dvxr.sentinel import create_product_api
        c = TestClient(create_product_api(require_consent=False, unsafe_dev=True),
                       raise_server_exceptions=False)
        # product lifecycle routes are present
        self.assertEqual(c.post("/v1/risk-reports", json={"patient_id": "P1"}).status_code, 200)
        # benchmark/screener endpoints are NOT part of the product surface
        for bench in ("/tasks", "/evidence", "/triage/depression"):
            self.assertEqual(c.get(bench).status_code, 404, f"{bench} must not be a product route")

    def test_idempotency_conflict_is_409_over_http(self):
        from starlette.testclient import TestClient
        from dvxr.sentinel import create_product_api
        c = TestClient(create_product_api(require_consent=False, unsafe_dev=True),
                       raise_server_exceptions=False)
        base = {"patient_id": "P1", "report_type": "cgm_glucose_risk", "idempotency_key": "kk"}
        self.assertEqual(c.post("/v1/risk-reports", json={**base, "prediction_horizons_minutes": [30]}
                                ).status_code, 200)
        r = c.post("/v1/risk-reports", json={**base, "prediction_horizons_minutes": [30, 60]})
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()
