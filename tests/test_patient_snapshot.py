"""PR11 / Gate 2: the PatientSnapshot is reproducible and causal — only events at/before the cutoff
enter it, its id is deterministic, and the generated prediction links back to it (spec §2 step 5)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.orchestrate import generate_risk_report  # noqa: E402
from dvxr.serve.snapshot import build_patient_snapshot  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


def _ev(eid, modality, ts, patient="P1", q=0.9):
    return {"event_id": eid, "patient_id": patient, "modality": modality,
            "observed_at_utc": ts, "quality_score": q}


class PatientSnapshotTest(unittest.TestCase):
    def test_future_events_are_excluded(self):
        events = [_ev("e1", "cgm", "2026-07-15T15:30:00"),
                  _ev("e2", "cgm", "2026-07-15T15:59:00"),
                  _ev("e3", "cgm", "2026-07-15T16:30:00")]      # AFTER cutoff
        snap = build_patient_snapshot(events, patient_id="P1",
                                      data_cutoff_at="2026-07-15T16:00:00")
        self.assertIn("e1", snap.event_ids)
        self.assertIn("e2", snap.event_ids)
        self.assertNotIn("e3", snap.event_ids)      # strictly-future excluded

    def test_snapshot_id_is_deterministic(self):
        events = [_ev("e2", "cgm", "2026-07-15T15:59:00"),
                  _ev("e1", "cgm", "2026-07-15T15:30:00")]
        a = build_patient_snapshot(events, patient_id="P1", data_cutoff_at="2026-07-15T16:00:00")
        b = build_patient_snapshot(list(reversed(events)), patient_id="P1",
                                   data_cutoff_at="2026-07-15T16:00:00")
        self.assertTrue(a.snapshot_id)
        self.assertEqual(a.snapshot_id, b.snapshot_id)    # order-independent (event_ids sorted)

    def test_missing_and_present_modalities(self):
        events = [_ev("e1", "cgm", "2026-07-15T15:30:00")]
        snap = build_patient_snapshot(events, patient_id="P1", data_cutoff_at="2026-07-15T16:00:00",
                                      expected_modalities=("eeg", "cgm", "wearable_phys"))
        self.assertEqual(snap.modalities_present, ["cgm"])
        self.assertEqual(snap.missing_modalities, ["eeg", "wearable_phys"])
        self.assertIn("cgm", snap.quality_by_modality)

    def test_other_patients_events_are_not_admitted(self):
        events = [_ev("e1", "cgm", "2026-07-15T15:30:00", patient="P1"),
                  _ev("e9", "cgm", "2026-07-15T15:30:00", patient="P2")]
        snap = build_patient_snapshot(events, patient_id="P1", data_cutoff_at="2026-07-15T16:00:00")
        self.assertEqual(snap.event_ids, ["e1"])

    def test_prediction_links_to_a_snapshot(self):
        pred, audit, consent, _m = open_local_stores(":memory:")
        consent.set_scope("P1", {"purposes": ["research"]})
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher", data_cutoff_at="2026-07-15T16:00:00")
        events = [_ev("e1", "cgm", "2026-07-15T15:30:00")]
        out = generate_risk_report(req, prediction_store=pred, audit_store=audit,
                                   consent_store=consent, events=events)
        # the prediction carries a snapshot_id, and the snapshot is in the audit trail
        self.assertTrue(out["prediction"]["snapshot_id"])
        snap_events = [e for e in audit.for_request(out["request_id"])
                       if e["event"] == "snapshot.created"]
        self.assertEqual(len(snap_events), 1)
        self.assertEqual(snap_events[0]["snapshot"]["snapshot_id"], out["prediction"]["snapshot_id"])
        self.assertIn("e1", snap_events[0]["snapshot"]["event_ids"])


if __name__ == "__main__":
    unittest.main()
