"""The grounded explanation never invents a number (LLM explains, never predicts).

Every numeric token the explanation emits must already appear in the prediction body it
explains. This is the machine-checkable form of the anti-hallucination invariant.
"""

from __future__ import annotations

import re
import unittest

from dvxr.serve.research_predict import run_research_prediction

try:
    from dvxr.serve.agents import agentic_available, run_agentic_prediction

    _AGENTIC = agentic_available()
except Exception:  # noqa: BLE001
    _AGENTIC = False

# Standalone numbers only — a digit inside an identifier (HbA1c, COVID-19, STAI-6) is not
# a numeric claim, so require non-alphanumeric boundaries on both sides.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])-?\d+\.?\d*(?![A-Za-z0-9])")


def _numbers_in(text: str) -> set[str]:
    out = set()
    for token in _NUMBER.findall(text or ""):
        try:
            out.add(f"{float(token):.4g}")
        except ValueError:
            continue
    return out


def _numbers_in_body(body: dict) -> set[str]:
    """Every numeric value anywhere in the body, normalised, plus percent forms."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(f"{float(value):.4g}")
            found.add(f"{float(value) * 100:.4g}")  # percent restatement of a probability
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(body)
    return found


_PREDICT = {
    "selected_outcome": "glucose_instability",
    "prediction_horizons_minutes": [30, 60],
    "inputs": {"hba1c": 7.2, "fasting_glucose": 130, "bmi": 31,
               "cgm_std": 45, "time_above_range": 40, "hrv_rmssd": 28},
}
_ABSTAIN = {"selected_outcome": "diabetes_status",
            "prediction_horizons_minutes": [30], "inputs": {"hrv_rmssd": 28}}


@unittest.skipUnless(_AGENTIC, "LangGraph (dvxr[agents]) not installed")
class NoHallucinatedNumbers(unittest.TestCase):
    def _assert_grounded(self, payload):
        body = run_agentic_prediction(payload)
        explanation = body.get("explanation") or {}
        # the explanation must declare it does not predict
        self.assertFalse(explanation.get("predicts", True))
        text_numbers = _numbers_in(explanation.get("text", ""))
        allowed = _numbers_in_body(body)
        # horizons and other integers in the prompt are also legitimate restatements
        allowed |= {f"{float(h):.4g}" for h in payload["prediction_horizons_minutes"]}
        ungrounded = {n for n in text_numbers if n not in allowed}
        self.assertEqual(
            ungrounded, set(),
            f"explanation emitted numbers absent from the body: {ungrounded}\n"
            f"text={explanation.get('text')!r}",
        )

    def test_prediction_explanation_is_grounded(self):
        self._assert_grounded(_PREDICT)

    def test_abstention_explanation_is_grounded(self):
        self._assert_grounded(_ABSTAIN)

    def test_direct_and_agentic_bodies_share_numbers(self):
        # sanity: the agentic path introduces no new numeric content in the scored body
        direct = run_research_prediction(_PREDICT)
        agentic = run_agentic_prediction(_PREDICT)
        for key in ("selected_outcome", "target_predictions", "forecast", "contributions"):
            self.assertEqual(agentic.get(key), direct.get(key))


if __name__ == "__main__":
    unittest.main()
