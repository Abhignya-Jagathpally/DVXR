"""The Claude-backed explainer is grounded by construction — no hallucinated numbers.

These tests need no API key and no network: they exercise the grounding validator, the
deterministic fallback, and the retry-then-fallback guard with a fake LLM that deliberately
invents numbers. The guarantee under test: whatever the LLM does, the shipped explanation
never contains a number absent from the prediction body.
"""

from __future__ import annotations

import os
import unittest

from dvxr.serve import llm_explainer as LE

_BODY_OK = {
    "status": "ok",
    "selected_outcome": {"name": "glucose_instability", "probability": 0.87,
                         "risk_band": "high", "validated_for_clinical_use": False},
    "contributions": [{"factor": "hba1c", "direction": "raises"}],
}
_BODY_ABSTAIN = {
    "status": "abstained",
    "selected_outcome": {"name": "diabetes_status", "probability": None,
                         "validated_for_clinical_use": False},
    "missing_or_stale_data": ["metabolic inputs"],
}


class Grounding(unittest.TestCase):
    def test_allowed_includes_value_and_percent_form(self):
        allowed = LE.allowed_numbers(_BODY_OK)
        self.assertIn("0.87", allowed)
        self.assertIn("87", allowed)  # percent restatement

    def test_is_grounded_flags_invented_numbers(self):
        allowed = LE.allowed_numbers(_BODY_OK)
        ok, ung = LE.is_grounded("probability 0.87, risk 0.42", allowed)
        self.assertFalse(ok)
        self.assertIn("0.42", ung)

    def test_word_internal_digits_are_not_numbers(self):
        ok, ung = LE.is_grounded("provide HbA1c and SpO2", set())
        self.assertTrue(ok)
        self.assertEqual(ung, set())


class DeterministicFallback(unittest.TestCase):
    def test_deterministic_is_grounded_and_never_predicts(self):
        out = LE.deterministic_explanation(_BODY_OK)
        self.assertFalse(out["predicts"])
        self.assertTrue(out["grounded"])
        allowed = LE.allowed_numbers(_BODY_OK)
        ok, _ = LE.is_grounded(out["text"], allowed)
        self.assertTrue(ok)

    def test_abstention_explained_without_number(self):
        out = LE.deterministic_explanation(_BODY_ABSTAIN)
        self.assertFalse(out["predicts"])
        self.assertIn("abstain", out["text"].lower())

    def test_available_false_without_key(self):
        self.assertFalse(LE.LLMExplainer(api_key="").available())


class HallucinationGuard(unittest.TestCase):
    def test_persistently_hallucinating_llm_is_caught_and_falls_back(self):
        ex = LE.LLMExplainer(api_key="dummy")
        ex.available = lambda: True  # pretend the API is reachable
        calls = {"n": 0}

        def fake_call(facts, flagged=None):
            calls["n"] += 1
            # always sneak in an ungrounded number
            return "probability 0.87 (high) but really 0.42 danger"

        ex._call = fake_call
        prev = os.environ.get("DVXR_EXPLAINER")
        os.environ["DVXR_EXPLAINER"] = "llm"
        try:
            out = ex.explain(_BODY_OK)
        finally:
            if prev is None:
                os.environ.pop("DVXR_EXPLAINER", None)
            else:
                os.environ["DVXR_EXPLAINER"] = prev
        # it retried (initial + retries) then fell back
        self.assertGreaterEqual(calls["n"], 2)
        self.assertEqual(out["source"], "deterministic_fallback_ungrounded_llm")
        # the SHIPPED text has no ungrounded number
        ok, ung = LE.is_grounded(out["text"], LE.allowed_numbers(_BODY_OK))
        self.assertTrue(ok, f"shipped ungrounded numbers: {ung}")

    def test_valid_llm_output_is_used(self):
        ex = LE.LLMExplainer(api_key="dummy")
        ex.available = lambda: True
        ex._call = lambda facts, flagged=None: (
            "Research-stage estimate: probability 0.87, risk band high. Not validated for clinical use."
        )
        prev = os.environ.get("DVXR_EXPLAINER")
        os.environ["DVXR_EXPLAINER"] = "llm"
        try:
            out = ex.explain(_BODY_OK)
        finally:
            if prev is None:
                os.environ.pop("DVXR_EXPLAINER", None)
            else:
                os.environ["DVXR_EXPLAINER"] = prev
        self.assertEqual(out["source"], "claude")
        self.assertTrue(out["grounded"])
        self.assertFalse(out["predicts"])


if __name__ == "__main__":
    unittest.main()
