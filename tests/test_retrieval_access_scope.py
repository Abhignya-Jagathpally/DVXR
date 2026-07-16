"""PR28 / Gate A (spec §7): a patient note is returned only if its access_scope admits the requesting
role. access_scope was required at index time but never compared to the caller's role at search time."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.retrieval import LocalKeywordTextIndex  # noqa: E402


def _note(cid, access_scope):
    return {"chunk_id": cid, "text": "glucose stress note content",
            "metadata": {"document_type": "clinical_note", "patient_id": "P1", "tenant_id": "t1",
                         "access_scope": access_scope, "section": "1"}}


class AccessScopeTest(unittest.TestCase):
    def setUp(self):
        self.idx = LocalKeywordTextIndex()
        self.idx.index(_note("clin_only", ["clinician"]))
        self.idx.index(_note("shared", ["clinician", "researcher"]))
        self.idx.index(_note("everyone", "all"))

    def _ids(self, role):
        hits = self.idx.search_patient("glucose stress", patient_id="P1", tenant_id="t1", role=role)
        return {h["chunk_id"] for h in hits}

    def test_researcher_cannot_see_clinician_only_note(self):
        got = self._ids("researcher")
        self.assertNotIn("clin_only", got)               # access_scope excludes researcher
        self.assertIn("shared", got)
        self.assertIn("everyone", got)

    def test_clinician_sees_clinician_scoped_notes(self):
        got = self._ids("clinician")
        self.assertEqual(got, {"clin_only", "shared", "everyone"})

    def test_missing_scope_admits_no_role(self):
        idx = LocalKeywordTextIndex()
        # a note whose access_scope is present-but-empty is fail-closed against any role
        idx._chunks.append({"chunk_id": "bad", "text": "glucose",
                            "metadata": {"document_type": "clinical_note", "patient_id": "P1",
                                         "tenant_id": "t1", "access_scope": ""}})
        self.assertEqual(idx.search_patient("glucose", patient_id="P1", tenant_id="t1",
                                            role="clinician"), [])

    def test_no_role_is_backward_compatible(self):
        # a caller that supplies no role keeps the pre-existing behavior (scope not enforced)
        hits = self.idx.search_patient("glucose stress", patient_id="P1", tenant_id="t1")
        self.assertEqual({h["chunk_id"] for h in hits}, {"clin_only", "shared", "everyone"})


if __name__ == "__main__":
    unittest.main()
