"""PR17 / Gate 8: the provenanced event contract is strict — event ids include the tenant, quality
scores must be in [0,1], duplicate events are rejected, and unspecified-consent rows are quarantined
rather than accepted into a prediction-ready table (spec §5)."""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.schemas import (  # noqa: E402
    enrich_provenance,
    quarantine_unconsented_events,
    validate_provenanced_events,
)


def _row(**ov):
    r = dict(subject_id="s1", session_id="ses1", timestamp_utc="2024-01-01T00:00:00Z",
             source_system="test", device="dev", modality="cgm", channel="glucose",
             value=100.0, unit="mg/dL", sampling_rate_hz=1.0, quality_flag="ok",
             label_name="", label_value="")
    r.update(ov)
    return r


class SchemaStrictnessTest(unittest.TestCase):
    def _events(self, rows):
        return pd.DataFrame(rows)

    def test_event_id_includes_tenant(self):
        df = self._events([_row()])
        a = enrich_provenance(df, tenant_id="t1")["event_id"].iloc[0]
        b = enrich_provenance(df, tenant_id="t2")["event_id"].iloc[0]
        self.assertNotEqual(a, b)          # same reading, different tenant ⇒ distinct id

    def test_out_of_range_quality_is_rejected(self):
        prov = enrich_provenance(self._events([_row()]), tenant_id="t1")
        prov = prov.copy()
        prov["quality_score"] = 1.5
        with self.assertRaises(ValueError):
            validate_provenanced_events(prov)

    def test_duplicate_event_id_is_rejected(self):
        # two identical rows ⇒ identical content hash ⇒ duplicate event_id
        prov = enrich_provenance(self._events([_row(), _row()]), tenant_id="t1")
        with self.assertRaises(ValueError):
            validate_provenanced_events(prov)

    def test_unspecified_consent_is_quarantined(self):
        ready_row = _row(subject_id="a")
        prov_ready = enrich_provenance(self._events([ready_row]), tenant_id="t1",
                                       consent_scope="research", access_scope="care_team")
        ready, quar = quarantine_unconsented_events(prov_ready)
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(quar), 0)

        prov_bad = enrich_provenance(self._events([_row(subject_id="b")]), tenant_id="t1")  # defaults unspecified
        ready2, quar2 = quarantine_unconsented_events(prov_bad)
        self.assertEqual(len(ready2), 0)
        self.assertEqual(len(quar2), 1)


if __name__ == "__main__":
    unittest.main()
