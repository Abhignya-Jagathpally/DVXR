from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.llm.client import LLMClient, OfflineLLM  # noqa: E402
from dvxr.llm.insight import (  # noqa: E402
    CAVEAT,
    build_grounded_facts,
    clinician_summary,
    personal_insight,
    write_insight_report,
)

BUNDLE = {
    "tasks": {"stress_detection": {"probability": 0.72, "band": "elevated"}},
    "glucose": {"now": 168.0, "forecast": 183.0, "lower": 168.0, "upper": 198.0},
    "biomarkers": {"hrv_rmssd": 28.40},
    "top_modality": "wearable_phys",
    "interventions": ["Stress is elevated — try paced breathing."],
}

_NUM = re.compile(r"\d+\.?\d*")


def _offline_client():
    # force offline explicitly so tests NEVER make a live API call
    return LLMClient(provider="offline")


class OfflineClientTest(unittest.TestCase):
    def test_offline_llm_deterministic(self):
        m = [{"role": "user", "content": "hello facts"}]
        self.assertEqual(OfflineLLM().complete(m), OfflineLLM().complete(m))

    def test_client_is_offline_without_key(self):
        c = _offline_client()
        self.assertTrue(c.is_offline)
        self.assertEqual(c.backend_name, "offline-template")


class InsightGroundingTest(unittest.TestCase):
    def test_personal_insight_contains_caveat(self):
        text = personal_insight(BUNDLE, client=_offline_client())
        self.assertIn(CAVEAT, text)

    def test_clinician_summary_contains_caveat(self):
        text = clinician_summary(BUNDLE, client=_offline_client())
        self.assertIn(CAVEAT, text)

    def test_no_ungrounded_numbers(self):
        text = personal_insight(BUNDLE, client=_offline_client())
        allowed = set(_NUM.findall(build_grounded_facts(BUNDLE)))
        found = set(_NUM.findall(text))
        # caveat/header carry no digits, so every number must come from the facts
        self.assertTrue(found.issubset(allowed),
                        f"ungrounded numbers: {found - allowed}")

    def test_determinism(self):
        a = personal_insight(BUNDLE, client=_offline_client())
        b = personal_insight(BUNDLE, client=_offline_client())
        self.assertEqual(a, b)

    def test_write_report_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "insight.md"
            res = write_insight_report(BUNDLE, out_path=str(out),
                                       client=_offline_client())
            self.assertEqual(res["backend"], "offline-template")
            self.assertTrue(out.exists())
            self.assertIn(CAVEAT, out.read_text())


if __name__ == "__main__":
    unittest.main()
