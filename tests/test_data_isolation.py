"""PR19 / Gate A (spec §7): DATA ISOLATION — a prediction for one (tenant, patient) can never read,
reuse, or surface another (tenant, patient)'s data. These are adversarial tests: every fixture mixes
identities on purpose and asserts the cross-identity data is refused, never inferred.

Covered surfaces:
  * snapshot builder        — only own tenant+patient+id-bearing events admitted (no fail-open)
  * CGM history assembly     — same, for the numeric history the predictor consumes
  * prediction store         — rows + idempotency keyed by (tenant, prediction_id)/(tenant, key)
  * consent store            — consent is tenant-scoped (tenant B's consent ≠ tenant A's)
  * scoped idempotency        — the same raw key across patients/tenants never collides
  * GET /v1/predictions/{id}  — a cross-tenant id is a 404 (record never leaves storage)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.orchestrate import (  # noqa: E402
    _cgm_history_from_events, _scoped_idempotency_key)
from dvxr.serve.snapshot import build_patient_snapshot  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


def _ev(patient, tenant, eid, modality="cgm", value=120.0, ts="2026-01-01T00:00:00Z", **extra):
    ev = {"patient_id": patient, "tenant_id": tenant, "event_id": eid,
          "modality": modality, "value": value, "observed_at_utc": ts, "quality_score": 0.9}
    ev.update(extra)
    return ev


class SnapshotIsolationTest(unittest.TestCase):
    def test_snapshot_admits_only_own_tenant_and_patient(self):
        events = [
            _ev("A", "t1", "e-A1"),                       # ours
            _ev("A", "t1", "e-A2", modality="wearable_phys"),  # ours, another modality
            _ev("B", "t1", "e-B1"),                       # other patient, same tenant — REJECT
            _ev("A", "t2", "e-A-t2"),                     # our patient id, other tenant — REJECT
        ]
        snap = build_patient_snapshot(events, patient_id="A", data_cutoff_at="",
                                      tenant_id="t1", expected_modalities=("cgm", "wearable_phys"))
        self.assertEqual(snap.event_ids, ["e-A1", "e-A2"])
        self.assertEqual(snap.tenant_id, "t1")
        self.assertNotIn("e-B1", snap.event_ids)
        self.assertNotIn("e-A-t2", snap.event_ids)

    def test_snapshot_rejects_missing_identity_never_infers(self):
        # a fail-OPEN builder would treat a missing patient_id as "matches the requested patient".
        events = [
            {"event_id": "e-noid-patient", "tenant_id": "t1", "modality": "cgm", "value": 1},
            {"event_id": "e-noid-tenant", "patient_id": "A", "modality": "cgm", "value": 1},
            _ev("A", "t1", "", modality="cgm"),           # blank event_id — quarantined
            _ev("A", "t1", "e-ok"),
        ]
        snap = build_patient_snapshot(events, patient_id="A", data_cutoff_at="", tenant_id="t1")
        self.assertEqual(snap.event_ids, ["e-ok"])

    def test_snapshot_id_is_tenant_bound(self):
        # identical events + patient + cutoff but different tenant ⇒ different snapshot id
        ev_a = [_ev("A", "t1", "e1")]
        ev_b = [_ev("A", "t2", "e1")]
        a = build_patient_snapshot(ev_a, patient_id="A", data_cutoff_at="", tenant_id="t1")
        b = build_patient_snapshot(ev_b, patient_id="A", data_cutoff_at="", tenant_id="t2")
        self.assertNotEqual(a.snapshot_id, b.snapshot_id)


class CgmHistoryIsolationTest(unittest.TestCase):
    def test_history_admits_only_own_tenant_and_patient(self):
        events = [
            _ev("A", "t1", "e1", value=100, ts="2026-01-01T00:00:00Z"),
            _ev("A", "t1", "e2", value=110, ts="2026-01-01T00:05:00Z"),
            _ev("B", "t1", "e3", value=999, ts="2026-01-01T00:06:00Z"),   # other patient
            _ev("A", "t2", "e4", value=888, ts="2026-01-01T00:07:00Z"),   # other tenant
        ]
        hist = _cgm_history_from_events(events, tenant_id="t1", patient_id="A", cutoff="")
        self.assertIsNotNone(hist)
        self.assertEqual(list(hist["glucose"]), [100, 110])
        self.assertNotIn(999, list(hist["glucose"]))
        self.assertNotIn(888, list(hist["glucose"]))

    def test_history_rejects_missing_identity(self):
        events = [
            {"modality": "cgm", "value": 500, "observed_at_utc": "2026-01-01T00:00:00Z"},  # no ids
            {"modality": "cgm", "value": 501, "tenant_id": "t1",
             "observed_at_utc": "2026-01-01T00:01:00Z"},                                    # no patient
        ]
        self.assertIsNone(_cgm_history_from_events(events, tenant_id="t1", patient_id="A", cutoff=""))


class ScopedIdempotencyTest(unittest.TestCase):
    def _key(self, patient, tenant):
        req = GenerateRequest(patient_id=patient, tenant_id=tenant,
                              report_type="stress_glucose_risk", idempotency_key="RAW").with_request_id()
        return _scoped_idempotency_key(req)

    def test_same_raw_key_differs_across_patients(self):
        self.assertNotEqual(self._key("A", "t1"), self._key("B", "t1"))

    def test_same_raw_key_differs_across_tenants(self):
        self.assertNotEqual(self._key("A", "t1"), self._key("A", "t2"))


class StorageIsolationTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")

    def test_same_prediction_id_isolated_by_tenant(self):
        rec_a = {"prediction_id": "pred_shared", "patient_id": "A", "tenant_id": "t1", "risk": None}
        rec_b = {"prediction_id": "pred_shared", "patient_id": "A", "tenant_id": "t2", "risk": None}
        self.pred.put(rec_a)
        self.pred.put(rec_b)
        self.assertEqual(self.pred.get("pred_shared", tenant_id="t1")["tenant_id"], "t1")
        self.assertEqual(self.pred.get("pred_shared", tenant_id="t2")["tenant_id"], "t2")

    def test_idempotency_key_isolated_by_tenant(self):
        self.pred.put({"prediction_id": "p1", "patient_id": "A", "tenant_id": "t1", "risk": None},
                      idempotency_key="t1|A|r|k")
        self.pred.put({"prediction_id": "p2", "patient_id": "A", "tenant_id": "t2", "risk": None},
                      idempotency_key="t2|A|r|k")
        # each tenant's scoped key resolves only its own row
        self.assertEqual(
            self.pred.get_by_idempotency_key("t1|A|r|k", tenant_id="t1")["prediction_id"], "p1")
        self.assertEqual(
            self.pred.get_by_idempotency_key("t2|A|r|k", tenant_id="t2")["prediction_id"], "p2")
        # tenant t1 asking with t2's key gets nothing
        self.assertIsNone(self.pred.get_by_idempotency_key("t2|A|r|k", tenant_id="t1"))

    def test_latest_for_patient_is_tenant_scoped(self):
        self.pred.put({"prediction_id": "pa", "patient_id": "A", "tenant_id": "t1", "risk": None})
        self.pred.put({"prediction_id": "pb", "patient_id": "A", "tenant_id": "t2", "risk": None})
        self.assertEqual(self.pred.latest_for_patient("A", tenant_id="t1")["prediction_id"], "pa")
        self.assertEqual(self.pred.latest_for_patient("A", tenant_id="t2")["prediction_id"], "pb")

    def test_consent_is_tenant_scoped(self):
        self.consent.set_scope("A", {"purposes": ["research"]}, tenant_id="t1")
        self.assertTrue(self.consent.check("A", "research", tenant_id="t1"))
        # tenant t2 never recorded consent for patient A ⇒ fail-closed deny
        self.assertFalse(self.consent.check("A", "research", tenant_id="t2"))


def _have_starlette():
    try:
        import starlette.testclient  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_starlette(), "starlette not installed (api extra)")
class CrossTenantRetrievalTest(unittest.TestCase):
    def _client(self, principals):
        from starlette.testclient import TestClient
        from dvxr.serve.api import create_app
        return TestClient(create_app(require_consent=False, principals=principals))

    def test_cross_tenant_prediction_id_is_not_retrievable(self):
        from dvxr.serve.auth import build_principals
        principals = build_principals({
            "key-t1": {"actor_id": "u1", "role": "researcher", "tenant_id": "t1"},
            "key-t2": {"actor_id": "u2", "role": "researcher", "tenant_id": "t2"},
        })
        c = self._client(principals)
        made = c.post("/v1/risk-reports", json={"patient_id": "A"},
                      headers={"X-API-Key": "key-t1"}).json()
        pid = made["prediction_id"]
        # tenant t1 (owner) can read it
        self.assertEqual(
            c.get(f"/v1/predictions/{pid}", headers={"X-API-Key": "key-t1"}).status_code, 200)
        # tenant t2 cannot — the record never leaves storage under t2's scope
        self.assertEqual(
            c.get(f"/v1/predictions/{pid}", headers={"X-API-Key": "key-t2"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
