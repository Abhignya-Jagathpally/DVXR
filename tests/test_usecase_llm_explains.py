"""End-to-end use-case: prediction + grounded LLM explanation + honest abstention.

Exercises the same path scripts/usecase_demo.py prints, asserting the framework behaves
correctly: a present-input case yields a grounded explanation that never predicts; a
missing-input case abstains; and clinical-safety flags hold throughout.
"""

from __future__ import annotations

import unittest

from dvxr.serve.research_predict import run_research_prediction

try:
    from dvxr.serve.agents import agentic_available, run_agentic_prediction

    _AGENTIC = agentic_available()
except Exception:  # noqa: BLE001
    _AGENTIC = False


def _predict(payload):
    if _AGENTIC:
        return run_agentic_prediction(payload)
    return run_research_prediction(payload)


_PRESENT = {
    "selected_outcome": "glucose_instability",
    "prediction_horizons_minutes": [30, 60],
    "inputs": {"hba1c": 7.8, "fasting_glucose": 142, "bmi": 33,
               "cgm_std": 52, "time_above_range": 48, "hrv_rmssd": 24},
}
_MISSING = {
    "selected_outcome": "diabetes_status",
    "prediction_horizons_minutes": [30],
    "inputs": {"hrv_rmssd": 41},
}


class UseCaseLLMExplains(unittest.TestCase):
    def test_present_inputs_predict_and_explain(self):
        body = _predict(_PRESENT)
        sel = body["selected_outcome"]
        self.assertIsNotNone(sel.get("probability"))
        # clinical-safety flag always holds
        self.assertFalse(sel.get("validated_for_clinical_use", True))
        # there is a grounded explanation (when the agentic path is available)
        if _AGENTIC:
            self.assertIn("explanation", body)
            self.assertFalse(body["explanation"]["predicts"])
            self.assertTrue(body["explanation"]["text"])

    def test_missing_inputs_abstain(self):
        body = _predict(_MISSING)
        self.assertEqual(body.get("status"), "abstained")
        # abstention names what is missing rather than guessing
        self.assertTrue(body.get("missing_or_stale_data")
                        or body["selected_outcome"].get("missing_or_stale_data"))
        if _AGENTIC:
            self.assertFalse(body["explanation"]["predicts"])

    def test_disclaimer_present(self):
        body = _predict(_PRESENT)
        self.assertIn("disclaimer", body)
        self.assertIn("not a diagnosis", body["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
