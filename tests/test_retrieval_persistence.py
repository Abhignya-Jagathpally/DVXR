"""PR27 / P0-6 (spec §4, §14): retrieval is integrated into routine Generate and its sources are
PERSISTED as a manifest, so a GET/idempotent-reuse reconstructs the identical citations — a
reproducible report never silently loses its sources. A retrieval outage is surfaced explicitly."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.retrieval import LocalKeywordTextIndex  # noqa: E402
from dvxr.serve.orchestrate import (  # noqa: E402
    assemble_persisted_report, generate_risk_report)
from dvxr.storage import open_local_stores  # noqa: E402


def _index_with_protocol():
    idx = LocalKeywordTextIndex()
    idx.index({"chunk_id": "chk_proto", "text": "stress glucose risk verify the CGM feed protocol",
               "metadata": {"document_type": "protocol", "document_id": "P-1",
                            "protocol_id": "CGM-ELEVATED", "protocol_version": 3, "active": True,
                            "section": "1"}})
    idx.index({"chunk_id": "chk_card", "text": "model card limitations research-grade not a diagnosis",
               "metadata": {"document_type": "model_card", "document_id": "MC-1", "section": "limits"}})
    return idx


class _BrokenRetrieval:
    def search(self, *a, **k):
        raise RuntimeError("retrieval backend down")
    def search_patient(self, *a, **k):
        raise RuntimeError("retrieval backend down")


class RetrievalManifestTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def _gen(self, retrieval, **ov):
        base = dict(patient_id="P1", report_type="stress_glucose_risk", user_role="researcher")
        base.update(ov)
        return generate_risk_report(GenerateRequest(**base), prediction_store=self.pred,
                                    audit_store=self.audit, consent_store=self.consent,
                                    retrieval=retrieval)

    def test_routine_generate_retrieves_and_persists_a_manifest(self):
        out = self._gen(_index_with_protocol())
        self.assertEqual(out["retrieval_status"], "complete")
        ids = {m["chunk_id"] for m in out["retrieval_manifest"]}
        self.assertIn("chk_proto", ids)                # active protocol retrieved without a question
        # the manifest is persisted with the prediction
        stored = self.pred.get(out["prediction_id"])
        self.assertEqual({m["chunk_id"] for m in stored["retrieval_manifest"]}, ids)

    def test_get_reconstructs_the_same_citations(self):
        out = self._gen(_index_with_protocol())
        post_citations = {c["source_id"] for c in out["explanation"]["citations"]}
        self.assertTrue(post_citations)                # POST produced citations
        rec = self.pred.get(out["prediction_id"])
        got = assemble_persisted_report(rec, user_role="researcher")
        get_citations = {c["source_id"] for c in got["explanation"]["citations"]}
        self.assertEqual(post_citations, get_citations)   # GET reproduces them exactly

    def test_retrieval_outage_is_surfaced_not_swallowed(self):
        out = self._gen(_BrokenRetrieval())
        self.assertEqual(out["retrieval_status"], "unavailable")
        self.assertFalse(out["protocol_grounding_complete"])   # explicit, not silently "grounded"
        self.assertEqual(out["retrieval_manifest"], [])

    def test_no_backend_is_disabled_not_incomplete(self):
        out = self._gen(None)
        self.assertEqual(out["retrieval_status"], "disabled")
        self.assertTrue(out["protocol_grounding_complete"])    # nothing to ground against in prototype


if __name__ == "__main__":
    unittest.main()
