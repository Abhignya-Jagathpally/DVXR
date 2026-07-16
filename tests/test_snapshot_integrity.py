"""PR25 / P0-7 (spec §5): snapshot integrity. A duplicate event_id with divergent content is an
integrity conflict (not a silent dedup), and the content hash covers the provenance that changes what
the model saw, so two materially-different inputs cannot collide on one snapshot id."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.serve.snapshot import EventIntegrityConflict, build_patient_snapshot  # noqa: E402


def _ev(eid, value=120.0, **extra):
    ev = {"event_id": eid, "patient_id": "P1", "tenant_id": "default", "modality": "cgm",
          "observed_at_utc": "2026-01-01T00:00:00Z", "value": value, "quality_score": 0.9}
    ev.update(extra)
    return ev


class DuplicateIdTest(unittest.TestCase):
    def test_identical_duplicate_is_idempotent(self):
        snap = build_patient_snapshot([_ev("e1"), _ev("e1")], patient_id="P1", data_cutoff_at="",
                                      tenant_id="default")
        self.assertEqual(snap.event_ids, ["e1"])          # same event twice ⇒ counted once

    def test_divergent_duplicate_is_an_integrity_conflict(self):
        with self.assertRaises(EventIntegrityConflict):
            build_patient_snapshot([_ev("e1", value=120.0), _ev("e1", value=999.0)],
                                   patient_id="P1", data_cutoff_at="", tenant_id="default")


class RicherContentHashTest(unittest.TestCase):
    def _sid(self, ev):
        return build_patient_snapshot([ev], patient_id="P1", data_cutoff_at="",
                                      tenant_id="default").snapshot_id

    def test_unit_change_changes_snapshot_id(self):
        # two readings identical except for UNIT (mg/dL vs mmol/L) must NOT collide
        self.assertNotEqual(self._sid(_ev("e1", unit="mg/dL")),
                            self._sid(_ev("e1", unit="mmol/L")))

    def test_converter_version_change_changes_snapshot_id(self):
        self.assertNotEqual(self._sid(_ev("e1", converter_version="1.0")),
                            self._sid(_ev("e1", converter_version="2.0")))

    def test_access_scope_change_changes_snapshot_id(self):
        self.assertNotEqual(self._sid(_ev("e1", access_scope="clinician")),
                            self._sid(_ev("e1", access_scope="participant")))


if __name__ == "__main__":
    unittest.main()
