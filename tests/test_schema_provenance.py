"""PR2: the additive provenance layer (spec §5) must attach identity/timing/quality/consent metadata
WITHOUT changing the pure 13-column `validate_events` contract."""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.schemas import (  # noqa: E402
    PROVENANCE_COLUMNS,
    REQUIRED_EVENT_COLUMNS,
    SCHEMA_VERSION,
    enrich_provenance,
    validate_events,
    validate_provenanced_events,
)


def _row(**ov):
    r = dict(subject_id="s1", session_id="ses1", timestamp_utc="2024-01-01T00:00:00Z",
             source_system="test", device="dev", modality="cgm", channel="glucose",
             value=100.0, unit="mg/dL", sampling_rate_hz=1.0, quality_flag="ok",
             label_name="", label_value="")
    r.update(ov)
    return r


class ProvenanceLayerTest(unittest.TestCase):
    def test_validate_events_still_returns_exactly_thirteen(self):
        clean = validate_events(pd.DataFrame([_row()]))
        self.assertEqual(list(clean.columns), REQUIRED_EVENT_COLUMNS)

    def test_enrich_adds_all_provenance_columns(self):
        prov = enrich_provenance(pd.DataFrame([_row()]))
        for c in PROVENANCE_COLUMNS:
            self.assertIn(c, prov.columns)
        self.assertEqual(prov["schema_version"].iloc[0], SCHEMA_VERSION)

    def test_event_id_is_deterministic_and_idempotent(self):
        df = pd.DataFrame([_row(), _row(value=105.0)])
        a = enrich_provenance(df)["event_id"].tolist()
        b = enrich_provenance(df)["event_id"].tolist()
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 2)          # distinct rows ⇒ distinct ids

    def test_quality_flag_maps_to_score_and_status(self):
        df = pd.DataFrame([_row(quality_flag="ok"), _row(value=1.0, quality_flag="bad")])
        prov = enrich_provenance(df)
        self.assertEqual(prov.loc[prov["quality_flag"] == "ok", "quality_status"].iloc[0], "good")
        self.assertEqual(prov.loc[prov["quality_flag"] == "bad", "quality_status"].iloc[0], "unusable")
        self.assertGreater(prov.loc[prov["quality_flag"] == "ok", "quality_score"].iloc[0],
                           prov.loc[prov["quality_flag"] == "bad", "quality_score"].iloc[0])

    def test_caller_supplied_values_are_not_overwritten(self):
        df = pd.DataFrame([_row(patient_id="PSEUDO-9")])   # extra col supplied by a converter
        prov = enrich_provenance(df, tenant_id="T1")
        self.assertEqual(prov["patient_id"].iloc[0], "PSEUDO-9")
        self.assertEqual(prov["tenant_id"].iloc[0], "T1")

    def test_ingested_at_is_a_passed_value_not_wall_clock(self):
        prov = enrich_provenance(pd.DataFrame([_row()]), ingested_at_utc="2024-01-02T00:00:00Z")
        self.assertEqual(prov["ingested_at_utc"].iloc[0], "2024-01-02T00:00:00Z")

    def test_validate_provenanced_requires_the_layer(self):
        with self.assertRaises(ValueError):
            validate_provenanced_events(validate_events(pd.DataFrame([_row()])))
        # but passes once enriched
        ok = validate_provenanced_events(enrich_provenance(pd.DataFrame([_row()])))
        self.assertEqual(len(ok), 1)


if __name__ == "__main__":
    unittest.main()
