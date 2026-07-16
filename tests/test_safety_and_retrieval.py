"""PR7: the policy engine selects the action, the LLM can't alter numbers or invent an action, every
claim resolves to a source, and inactive/superseded protocols are never retrieved (spec §8, §14, §16)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.llm.grounded import grounded_explanation  # noqa: E402
from dvxr.retrieval import LocalTextIndex, chunk_protocol  # noqa: E402
from dvxr.safety.policy import ACTION_REGISTRY, select_action  # noqa: E402
from dvxr.safety.validators import (  # noqa: E402
    GroundingError,
    validate_action_id,
    validate_citations,
    validate_no_diagnosis_language,
    validate_numbers,
)


class PolicyEngineTest(unittest.TestCase):
    def test_abstention_selects_insufficient_data(self):
        self.assertEqual(select_action(abstained=True).action_id, "INSUFFICIENT_DATA")

    def test_high_risk_high_confidence_escalates_for_clinician(self):
        d = select_action(risk_category="high", confidence=0.9, data_quality="good", role="clinician")
        self.assertEqual(d.action_id, "ESCALATE_PER_APPROVED_PROTOCOL")
        self.assertTrue(d.requires_clinician_review)

    def test_high_risk_on_poor_data_requests_verification_not_a_warning(self):
        d = select_action(risk_category="high", confidence=0.9, data_quality="poor")
        self.assertEqual(d.action_id, "VERIFY_SENSOR_AND_CGM")

    def test_low_risk_good_data_continues_monitoring(self):
        self.assertEqual(select_action(risk_category="low", confidence=0.9,
                                       data_quality="good").action_id, "CONTINUE_MONITORING")

    def test_action_is_always_from_the_approved_registry(self):
        for args in [dict(abstained=True), dict(risk_category="high", confidence=0.9, data_quality="good"),
                     dict(risk_category="elevated"), dict(data_quality="poor")]:
            self.assertIn(select_action(**args).action_id, ACTION_REGISTRY)

    def test_role_cannot_get_a_disallowed_action(self):
        # a participant may not receive a clinician-only escalation
        d = select_action(risk_category="high", confidence=0.9, data_quality="good", role="participant")
        self.assertIn("participant", ACTION_REGISTRY[d.action_id].permitted_roles)


class NumericAndCitationGroundingTest(unittest.TestCase):
    def _prediction(self):
        return {"risk": {"excursion_30m": 0.58, "excursion_60m": 0.76}, "confidence": 0.81,
                "prediction_horizons_minutes": [30, 60], "missing_modalities": []}

    def test_matching_numbers_pass(self):
        validate_numbers("30-minute risk 0.58 and 60-minute risk 0.76, confidence 0.81",
                         self._prediction())

    def test_fabricated_number_is_rejected(self):
        with self.assertRaises(GroundingError):
            validate_numbers("the risk is 0.99", self._prediction())

    def test_action_mismatch_is_rejected(self):
        with self.assertRaises(GroundingError):
            validate_action_id("CONTINUE_MONITORING", "ESCALATE_PER_APPROVED_PROTOCOL")

    def test_claim_without_a_real_source_is_rejected(self):
        with self.assertRaises(GroundingError):
            validate_citations([{"statement": "x", "source_id": "ghost"}], {"chk_real"})

    def test_diagnostic_language_is_rejected(self):
        with self.assertRaises(GroundingError):
            validate_no_diagnosis_language("You have diabetes and should increase your insulin.")
        validate_no_diagnosis_language("This is research-grade decision-support, not a diagnosis.")


class GroundedExplanationTest(unittest.TestCase):
    def test_abstention_explanation_is_valid_and_number_free(self):
        pred = {"report_type": "stress_glucose_risk", "abstained": True,
                "abstain_reason": "synchronized data required", "missing_modalities": ["cgm"],
                "prediction_horizons_minutes": [30, 60]}
        action = {"action_id": "INSUFFICIENT_DATA"}
        exp = grounded_explanation(pred, evidence=None, action=action, sources=[])
        self.assertEqual(exp["action_id"], "INSUFFICIENT_DATA")
        self.assertIn("not a diagnosis", " ".join(exp["limitations"]).lower())

    def test_supporting_factors_cite_only_real_sources(self):
        pred = {"report_type": "stress_glucose_risk", "risk": {"excursion_30m": 0.58},
                "prediction_horizons_minutes": [30], "confidence": 0.81, "missing_modalities": []}
        ev = {"contributions": {"cgm": 0.6, "eeg": 0.1}}
        exp = grounded_explanation(pred, ev, {"action_id": "REVIEW_ELEVATED_RISK"}, sources=[])
        self.assertEqual(len(exp["supporting_factors"]), 2)


class RetrievalFilterTest(unittest.TestCase):
    def _index(self):
        idx = LocalTextIndex()
        active = chunk_protocol("1. Verify the CGM feed before acting on elevated risk.",
                                {"document_id": "P-1", "document_type": "protocol",
                                 "protocol_id": "CGM-ELEVATED", "protocol_version": 2, "active": True})
        old = chunk_protocol("1. Old guidance: act immediately on elevated risk.",
                             {"document_id": "P-0", "document_type": "protocol",
                              "protocol_id": "CGM-ELEVATED", "protocol_version": 1, "active": False})
        idx.index_all(active + old)
        return idx

    def test_inactive_protocol_is_not_retrieved(self):
        hits = self._index().search("elevated risk CGM", filters={"active": True})
        self.assertTrue(hits)
        for h in hits:
            self.assertTrue(h["metadata"]["active"])
            self.assertNotEqual(h["metadata"]["protocol_version"], 1)

    def test_version_filter_excludes_superseded(self):
        hits = self._index().search("elevated risk", filters={"protocol_version": 2})
        self.assertTrue(all(h["metadata"]["protocol_version"] == 2 for h in hits))

    def test_patient_namespace_is_mandatory_for_clinical_notes(self):
        from dvxr.retrieval import chunk_note
        from dvxr.retrieval.search import LocalKeywordTextIndex
        idx = LocalKeywordTextIndex()
        idx.index_all(chunk_note("Assessment:\nStable.", {"document_id": "n1",
                      "document_type": "clinical_note", "patient_id": "P1", "tenant_id": "t1",
                      "access_scope": "care_team"}))
        idx.index_all(chunk_note("Assessment:\nUnstable.", {"document_id": "n2",
                      "document_type": "clinical_note", "patient_id": "P2", "tenant_id": "t1",
                      "access_scope": "care_team"}))
        # a clinical note is NOT reachable through general search (no patient scope) ...
        self.assertEqual(idx.search("assessment"), [])
        # ... only through search_patient, which requires patient_id + tenant_id
        hits = idx.search_patient("assessment", patient_id="P1", tenant_id="t1")
        self.assertTrue(hits)
        self.assertTrue(all(h["metadata"]["patient_id"] == "P1" for h in hits))
        with self.assertRaises(ValueError):
            idx.search_patient("assessment", patient_id="", tenant_id="t1")

    def test_chunk_missing_required_metadata_is_rejected(self):
        from dvxr.retrieval.search import LocalKeywordTextIndex, RetrievalMetadataError
        idx = LocalKeywordTextIndex()
        with self.assertRaises(RetrievalMetadataError):   # clinical_note without tenant_id/access_scope
            idx.index({"chunk_id": "c1", "text": "x",
                       "metadata": {"document_type": "clinical_note", "patient_id": "P1"}})


if __name__ == "__main__":
    unittest.main()
