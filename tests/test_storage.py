"""PR2: local sqlite-backed stores satisfy the Protocols, are idempotent, and fail-closed on consent."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.storage import (  # noqa: E402
    AuditStore,
    ConsentStore,
    ModelRegistry,
    PredictionStore,
    open_local_stores,
)


class LocalStoresTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, self.models = open_local_stores(":memory:")

    def test_protocol_conformance(self):
        self.assertIsInstance(self.pred, PredictionStore)
        self.assertIsInstance(self.audit, AuditStore)
        self.assertIsInstance(self.consent, ConsentStore)
        self.assertIsInstance(self.models, ModelRegistry)

    def test_prediction_put_get_roundtrip(self):
        pid = self.pred.put({"request_id": "r1", "patient_id": "P1", "risk": {"excursion_30m": 0.5}})
        got = self.pred.get(pid)
        self.assertEqual(got["patient_id"], "P1")
        self.assertEqual(self.pred.latest_for_patient("P1")["prediction_id"], pid)

    def test_prediction_is_idempotent_by_key(self):
        p = {"request_id": "r1", "patient_id": "P1", "risk": {"excursion_30m": 0.5}}
        a = self.pred.put(p, idempotency_key="idem-1")
        b = self.pred.put({**p, "risk": {"excursion_30m": 0.99}}, idempotency_key="idem-1")
        self.assertEqual(a, b)                                  # same key ⇒ same prediction id
        self.assertEqual(self.pred.get(a)["risk"]["excursion_30m"], 0.5)  # first write wins

    def test_audit_is_appended_and_queryable(self):
        self.audit.append({"request_id": "r1", "event": "generate.requested"})
        self.audit.append({"request_id": "r1", "event": "generate.completed"})
        rows = self.audit.for_request("r1")
        self.assertEqual([r["event"] for r in rows],
                         ["generate.requested", "generate.completed"])

    def test_consent_fails_closed(self):
        self.assertFalse(self.consent.check("P-unknown", "research"))   # no record ⇒ deny
        self.consent.set_scope("P1", {"purposes": ["research"]})
        self.assertTrue(self.consent.check("P1", "research"))
        self.assertFalse(self.consent.check("P1", "clinical"))

    def test_model_registry_tracks_active_version(self):
        self.models.register("neuroglycemic-fusion", "1.0.0", {"note": "baseline"}, active=True)
        self.models.register("neuroglycemic-fusion", "1.1.0", {"note": "gated"}, active=True)
        self.assertEqual(self.models.active("neuroglycemic-fusion")["version"], "1.1.0")
        self.assertEqual(self.models.get("neuroglycemic-fusion", "1.0.0")["meta"]["note"], "baseline")


if __name__ == "__main__":
    unittest.main()
