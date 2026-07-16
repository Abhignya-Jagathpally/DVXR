"""PR16 / Gate 7: retrieval is accurately named + patient-isolated + provenance-enforced, and the
glucose Generate path uses ONLY the validated grounded explainer (spec §4, §8, §14, §16)."""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.retrieval import (  # noqa: E402
    LocalKeywordTextIndex,
    LocalTextIndex,
    RetrievalMetadataError,
    RetrievalRepository,
)

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "dvxr")


class RetrievalNamingTest(unittest.TestCase):
    def test_alias_points_to_keyword_index(self):
        self.assertIs(LocalTextIndex, LocalKeywordTextIndex)   # honest name; alias kept
        self.assertIsInstance(LocalKeywordTextIndex(), RetrievalRepository)


class ProvenanceEnforcementTest(unittest.TestCase):
    def test_protocol_missing_version_is_rejected(self):
        idx = LocalKeywordTextIndex()
        with self.assertRaises(RetrievalMetadataError):
            idx.index({"chunk_id": "c", "text": "x",
                       "metadata": {"document_type": "protocol", "protocol_id": "P"}})  # no version/active

    def test_note_missing_tenant_or_scope_is_rejected(self):
        idx = LocalKeywordTextIndex()
        with self.assertRaises(RetrievalMetadataError):
            idx.index({"chunk_id": "c", "text": "x",
                       "metadata": {"document_type": "clinical_note", "patient_id": "P1",
                                    "tenant_id": "t1"}})           # missing access_scope


class PatientIsolationTest(unittest.TestCase):
    def _idx(self):
        idx = LocalKeywordTextIndex()
        for pid in ("P1", "P2"):
            idx.index({"chunk_id": f"n-{pid}", "text": "assessment stable",
                       "metadata": {"document_type": "clinical_note", "patient_id": pid,
                                    "tenant_id": "t1", "access_scope": "care_team"}})
        return idx

    def test_general_search_never_returns_notes(self):
        self.assertEqual(self._idx().search("assessment"), [])

    def test_search_patient_isolates_by_patient(self):
        hits = self._idx().search_patient("assessment", patient_id="P1", tenant_id="t1")
        self.assertTrue(hits and all(h["metadata"]["patient_id"] == "P1" for h in hits))

    def test_search_patient_isolates_by_tenant(self):
        self.assertEqual(self._idx().search_patient("assessment", patient_id="P1", tenant_id="other"),
                         [])


class OneExplanationPathTest(unittest.TestCase):
    """The glucose Generate lifecycle must reach ONLY the validated grounded explainer — never the
    hosted-prose insight path or an experimental LLM predictor."""

    FORBIDDEN = ("llm.insight", "llm_representation_probe", "llm.predictor")

    def _imports(self, relpath):
        with open(os.path.join(SRC, relpath), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        return mods

    def test_generate_path_does_not_import_unvalidated_llm(self):
        for relpath in ("serve/orchestrate.py", "prediction/service.py", "llm/grounded.py"):
            mods = self._imports(relpath)
            for bad in self.FORBIDDEN:
                self.assertFalse(any(bad in m for m in mods),
                                 f"{relpath} must not import {bad!r} (found in {mods})")

    def test_grounded_explanation_is_the_generate_explainer(self):
        # orchestrate imports grounded_explanation, the self-validating path
        self.assertIn("dvxr.llm.grounded", self._imports("serve/orchestrate.py"))


if __name__ == "__main__":
    unittest.main()
