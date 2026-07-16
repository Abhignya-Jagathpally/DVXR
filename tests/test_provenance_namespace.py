"""PR31 / Gate A (spec §5): a research-participant id is never silently equated with a clinical patient
id. When patient_id is derived from subject_id it is namespaced (research:<subject>) and stamped
patient_id_namespace='research'; an explicitly-supplied patient_id wins and is marked 'explicit'."""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.schemas import enrich_provenance  # noqa: E402


def _row(subject_id="s0", **ov):
    r = {"subject_id": subject_id, "session_id": "sess", "timestamp_utc": "2026-01-01T00:00:00Z",
         "source_system": "test", "device": "dev", "modality": "cgm", "channel": "glucose",
         "value": 120.0, "unit": "mg/dL", "sampling_rate_hz": 0.0, "quality_flag": "ok",
         "label_name": "", "label_value": ""}
    r.update(ov)
    return r


class ProvenanceNamespaceTest(unittest.TestCase):
    def test_defaulted_patient_id_is_namespaced_not_bare_subject(self):
        prov = enrich_provenance(pd.DataFrame([_row("s0")]), tenant_id="t1")
        self.assertEqual(prov["patient_id"].iloc[0], "research:s0")     # NOT bare "s0"
        self.assertEqual(prov["patient_id_namespace"].iloc[0], "research")

    def test_explicit_patient_id_is_preserved_and_marked_explicit(self):
        df = pd.DataFrame([_row("s0", patient_id="MRN-42")])
        prov = enrich_provenance(df, tenant_id="t1")
        self.assertEqual(prov["patient_id"].iloc[0], "MRN-42")
        self.assertEqual(prov["patient_id_namespace"].iloc[0], "explicit")

    def test_map_hit_is_explicit_unmapped_is_research(self):
        df = pd.DataFrame([_row("s0"), _row("s1")])
        prov = enrich_provenance(df, tenant_id="t1", patient_id_map={"s0": "MRN-7"})
        by_subj = dict(zip(prov["subject_id"], prov["patient_id"]))
        by_ns = dict(zip(prov["subject_id"], prov["patient_id_namespace"]))
        self.assertEqual(by_subj["s0"], "MRN-7")
        self.assertEqual(by_ns["s0"], "explicit")
        self.assertEqual(by_subj["s1"], "research:s1")                  # unmapped ⇒ namespaced
        self.assertEqual(by_ns["s1"], "research")


if __name__ == "__main__":
    unittest.main()
