"""Tests for the research-stage glucose product surface (dvxr.serve.vision).

The default stress_glucose_risk report must ABSTAIN with no fabricated risk number while the product
is research-stage — the honesty guardrail on the glucose re-headline.
"""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GlucoseAbstentionTest(unittest.TestCase):
    def test_report_abstains_with_no_fabricated_risk(self):
        from dvxr.serve.vision import glucose_risk_report, ABSTAIN_ACTION_ID
        r = glucose_risk_report(patient_id="PSEUDO-1")
        self.assertEqual(r["status"], "abstained")
        self.assertEqual(r["action_id"], ABSTAIN_ACTION_ID)
        self.assertIsNone(r["risk"], "the research-stage glucose report must not carry a risk number")
        self.assertTrue(r["research_stage"])
        self.assertIn("synchronized", " ".join(r["missing_or_stale_data"]).lower())

    def test_render_states_research_stage_and_not_a_diagnosis(self):
        from dvxr.serve.vision import render_glucose_report
        text = render_glucose_report().lower()
        self.assertIn("research-stage", text)
        self.assertIn("not a diagnosis", text)
        self.assertIn("abstain", text)
        # no fabricated probability leaks into the render
        import re
        self.assertNotRegex(text, r"risk\s*(probability)?\s*[:=]?\s*0?\.\d")

    def test_horizons_default_to_30_60(self):
        from dvxr.serve.vision import glucose_risk_report
        self.assertEqual(glucose_risk_report()["prediction_horizons_minutes"], [30, 60])

    def test_app_panel_is_importable_without_streamlit(self):
        from scripts.screen_app import glucose_product_panel_md
        md = glucose_product_panel_md().lower()
        self.assertIn("research-stage", md)
        self.assertIn("not a diagnosis", md)
        self.assertIn("abstain", md)


if __name__ == "__main__":
    unittest.main()
