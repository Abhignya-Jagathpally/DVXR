"""PR30 / Gate A: audit storage + retrieval are tenant-scoped. The audit table carries a tenant_id
column; for_request(tenant_id=...) never returns another tenant's audit trail for the same request id."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.orchestrate import generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


class AuditTenantScopeTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")

    def test_for_request_is_tenant_scoped(self):
        self.audit.append({"tenant_id": "t1", "request_id": "R", "event": "e1"})
        self.audit.append({"tenant_id": "t2", "request_id": "R", "event": "e2"})
        t1 = self.audit.for_request("R", tenant_id="t1")
        self.assertEqual([e["event"] for e in t1], ["e1"])       # only t1's entry
        t2 = self.audit.for_request("R", tenant_id="t2")
        self.assertEqual([e["event"] for e in t2], ["e2"])
        # unscoped read remains backward-compatible (returns both)
        self.assertEqual(len(self.audit.for_request("R")), 2)

    def test_generate_audit_trail_is_scoped_to_its_tenant(self):
        self.consent.set_scope("P1", {"purposes": ["research"]}, tenant_id="t1")
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              tenant_id="t1", user_role="researcher")
        out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent)
        # every entry of this request is retrievable under its own tenant, and none under another
        own = self.audit.for_request(out["request_id"], tenant_id="t1")
        self.assertTrue(any(e["event"] == "generate.completed" for e in own))
        other = self.audit.for_request(out["request_id"], tenant_id="t2")
        self.assertEqual(other, [])


if __name__ == "__main__":
    unittest.main()
