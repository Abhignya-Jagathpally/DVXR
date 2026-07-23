"""The LangGraph orchestration path changes control flow, never numbers.

Skips cleanly when the optional ``agents`` extra (LangGraph) is not installed, so the
torch-free honesty audit and base test run are unaffected.
"""

from __future__ import annotations

import unittest

from dvxr.serve.research_predict import run_research_prediction

try:  # optional dependency
    from dvxr.serve.agents import agentic_available, run_agentic_prediction

    _AGENTIC = agentic_available()
except Exception:  # noqa: BLE001
    _AGENTIC = False


_ADDED_KEYS = ("explanation", "trace", "orchestration")


def _numeric_body(body: dict) -> dict:
    stripped = dict(body)
    for key in _ADDED_KEYS:
        stripped.pop(key, None)
    return stripped


_PREDICT_PAYLOAD = {
    "selected_outcome": "glucose_instability",
    "prediction_horizons_minutes": [30, 60],
    "inputs": {
        "hba1c": 7.2, "fasting_glucose": 130, "bmi": 31,
        "cgm_std": 45, "time_above_range": 40, "hrv_rmssd": 28, "eda_scl": 6.5,
    },
}
_ABSTAIN_PAYLOAD = {
    "selected_outcome": "diabetes_status",
    "prediction_horizons_minutes": [30],
    "inputs": {"hrv_rmssd": 28},
}


@unittest.skipUnless(_AGENTIC, "LangGraph (dvxr[agents]) not installed")
class AgenticParity(unittest.TestCase):
    def test_predicting_payload_is_byte_identical(self):
        direct = run_research_prediction(_PREDICT_PAYLOAD)
        agentic = run_agentic_prediction(_PREDICT_PAYLOAD)
        self.assertEqual(direct["status"], "ok")
        self.assertEqual(_numeric_body(agentic), direct)

    def test_abstaining_payload_is_byte_identical_and_fail_closed(self):
        direct = run_research_prediction(_ABSTAIN_PAYLOAD)
        agentic = run_agentic_prediction(_ABSTAIN_PAYLOAD)
        self.assertEqual(direct["status"], "abstained")
        self.assertEqual(_numeric_body(agentic), direct)

    def test_orchestration_additions_are_present_and_predict_free(self):
        agentic = run_agentic_prediction(_PREDICT_PAYLOAD)
        self.assertEqual(agentic["orchestration"], "langgraph-v1")
        self.assertIsInstance(agentic["trace"], list)
        self.assertGreaterEqual(len(agentic["trace"]), 6)
        # The explanation node explains; it never claims to predict.
        self.assertFalse(agentic["explanation"]["predicts"])

    def test_abstention_routes_to_abstention_explainer(self):
        agentic = run_agentic_prediction(_ABSTAIN_PAYLOAD)
        visited = [record["node"] for record in agentic["trace"]]
        self.assertIn("explain_abstention", visited)
        self.assertNotIn("explain_prediction", visited)


if __name__ == "__main__":
    unittest.main()
