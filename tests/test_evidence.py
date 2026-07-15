"""Tests for the evidence layer (dvxr.serve.evidence) and the evidence-page generator.

These are the product's honesty tests: every headline number must still trace to its committed
scoreboard, and the excluded claims must be present so the P5 audit can enforce them.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class EvidenceTraceTest(unittest.TestCase):
    def test_all_numbers_trace_to_scoreboards(self):
        from dvxr.serve.evidence import verify_against_scoreboards
        problems = verify_against_scoreboards()
        self.assertEqual(problems, [], f"scoreboard drift: {problems}")

    def test_depression_is_the_headline(self):
        from dvxr.serve.evidence import comparative_table
        head = [r for r in comparative_table() if r["headline"]]
        self.assertEqual(len(head), 1)
        self.assertIn("Depression", head[0]["task"])
        self.assertEqual(head[0]["winner_method"], "LaBraM EEG FM")
        self.assertGreater(head[0]["auroc"], 0.95)

    def test_excluded_claims_named(self):
        from dvxr.serve.evidence import EXCLUDED_CLAIMS
        for key in ("deap_affect", "cacmf_as_win", "llm_as_predictor",
                    "mimic_mortality", "cgmacros_diabetes", "diagnosis"):
            self.assertIn(key, EXCLUDED_CLAIMS)

    def test_report_renders_and_verifies(self):
        from dvxr.serve.evidence import render_report
        r = render_report()
        self.assertIn("VERIFIED", r)
        self.assertIn("HEADLINE", r)
        self.assertIn("NOT claimed", r)


class EvidencePageTest(unittest.TestCase):
    def test_page_is_self_contained(self):
        import re
        from build_evidence_page import render_page
        h = render_page()
        self.assertIn("DVXR Screen", h)
        self.assertIn("0.961", h)
        # CSP-safe: no external resource LOADS (DOI <a href> navigation links are allowed)
        for bad in ("src=", "@import", "url(http", "<link", "<script"):
            self.assertNotIn(bad, h)
        self.assertNotIn("http", re.sub(r'href="[^"]*"', "", h))
        self.assertIn("does <em>not</em> claim", h)
        self.assertIn("subject-level", h)      # both granularities surfaced
        self.assertIn("doi.org", h)            # external SOTA cited with DOIs


if __name__ == "__main__":
    unittest.main()
