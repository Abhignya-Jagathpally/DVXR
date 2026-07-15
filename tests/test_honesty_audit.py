"""BLOCKING honesty audit for the DVXR Screen product surfaces.

This is the gate the whole project's credibility rests on. It asserts, across the *structured*
claim registry and every prose surface a user sees (CLI report, evidence page, model card), that:

  1. every headline number still resolves to a committed scoreboard file (no drift);
  2. no excluded capability is ever sold as a product claim — DEAP affect, the learned CACMF fusion
     as a win, the LLM as a predictor, MIMIC mortality, the cgmacros_diabetes leak;
  3. nothing is presented as a diagnosis — every "diagnos*" mention is negated.

If this test fails, the product is making a claim it cannot stand behind — do not ship.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EXCLUDED_TASKS = {"deap_anxiety", "deap_arousal", "deap_affect", "mortality", "cgmacros_diabetes"}
FORBIDDEN_WINNERS = {"learned CACMF fusion", "learned cross-modal fusion", "CACMF", "rep:llm", "LLM"}


class StructuredClaimAudit(unittest.TestCase):
    def test_numbers_trace_to_scoreboards(self):
        from dvxr.serve.evidence import verify_against_scoreboards
        self.assertEqual(verify_against_scoreboards(), [])

    def test_no_excluded_task_is_a_product_claim(self):
        from dvxr.serve.evidence import PRODUCT_CLAIMS
        for c in PRODUCT_CLAIMS:
            self.assertNotIn(c.task, EXCLUDED_TASKS, f"{c.task} must not be a product claim")

    def test_no_forbidden_model_is_a_winner(self):
        from dvxr.serve.evidence import comparative_table
        for row in comparative_table():
            self.assertNotIn(row["winner_method"], FORBIDDEN_WINNERS,
                             f"{row['task']}: winner is a forbidden/losing model")

    def test_every_claim_has_a_caveat_and_real_source(self):
        from dvxr.serve.evidence import PRODUCT_CLAIMS
        for c in PRODUCT_CLAIMS:
            self.assertTrue(c.caveat.strip(), f"{c.task} missing caveat")
            self.assertTrue((ROOT / c.source_file).exists(), f"{c.source_file} missing")

    def test_llm_is_not_wired_as_a_predictor(self):
        from dvxr.serve.screener import REPRESENTATION_BY_TASK
        for rep in REPRESENTATION_BY_TASK.values():
            self.assertNotIn("llm", rep.lower())

    def test_every_external_result_carries_doi_and_protocol(self):
        """External comparators must never appear without their provenance + protocol label."""
        from dvxr.serve.evidence import EXTERNAL_SOTA
        ok_protocols = {"LOSO (cross-subject)", "subject-independent",
                        "within-subject/segment", "external-validation"}
        for task, results in EXTERNAL_SOTA.items():
            for e in results:
                self.assertRegex(e.doi, r"^10\.\d{4,}/", f"{task}:{e.method} DOI")
                self.assertIn(e.protocol, ok_protocols, f"{task}:{e.method} protocol")

    def test_report_shows_both_auroc_granularities(self):
        from dvxr.serve.evidence import render_report
        r = render_report().lower()
        self.assertIn("window-level", r)
        self.assertIn("subject-level", r)


_NEGATOR = re.compile(r"not |never |rather than|decision-support|screening|isn't|is not")


def _negated(text: str) -> bool:
    """Every line that mentions 'diagnos*' must also carry a negator on that line — i.e. the mention
    is a disclaimer ("not a diagnosis", "screening, not diagnosis"), never a positive claim."""
    for line in text.lower().splitlines():
        if "diagnos" in line and not _NEGATOR.search(line):
            return False
    return True


class ProseSurfaceAudit(unittest.TestCase):
    def _surfaces(self):
        from dvxr.serve.evidence import render_report
        from build_evidence_page import render_page
        surfaces = {"report": render_report(), "evidence_page": render_page()}
        mc = ROOT / "docs" / "MODEL_CARD.md"
        if mc.exists():
            surfaces["model_card"] = mc.read_text()
        return surfaces

    def test_no_undisclosed_diagnosis_claim(self):
        for name, text in self._surfaces().items():
            self.assertTrue(_negated(text), f"{name}: an un-negated diagnosis claim slipped in")

    def test_disclaimer_present(self):
        for name, text in self._surfaces().items():
            low = text.lower()
            self.assertTrue("not a diagnos" in low or "never a diagnos" in low
                            or "not a diagnostic" in low or "never a diagnostic" in low,
                            f"{name}: missing the research-prototype / not-a-diagnosis disclaimer")

    def test_decision_curve_surface_is_attributed_and_honest(self):
        # Wherever a net-benefit / decision curve is shown, it must name its method (Vickers & Elkin)
        # and the default policies it is measured against — never presented as a bare "utility" number.
        page = self._surfaces()["evidence_page"].lower()
        if "net benefit" in page or "decision-curve" in page or "decision curve" in page:
            self.assertIn("vickers", page, "decision-curve surface missing method attribution")
            self.assertIn("treat", page, "decision-curve surface missing the treat-all/none baseline")
            self.assertIn("bootstrap", page, "decision-curve surface must disclose the useful-band gate")

    def test_evidence_page_has_no_external_resource_loads(self):
        page = self._surfaces()["evidence_page"]
        # Resource LOADS are forbidden (CSP-blocked anyway); DOI hyperlinks (<a href>) are allowed.
        for bad in ("src=", "@import", "url(http", "<link", "<script"):
            self.assertNotIn(bad, page, f"evidence page loads an external resource: {bad}")
        # after removing hyperlinks, no bare URL should remain (i.e. every URL is a navigation link)
        stripped = re.sub(r'href="[^"]*"', "", page)
        self.assertNotIn("http", stripped, "evidence page has a non-hyperlink external URL")


class UploadOutOfDistributionAudit(unittest.TestCase):
    """The live upload path must never present an upload's number as the validated cohort AUROC."""

    def test_upload_result_is_flagged_not_validated(self):
        # run the live engine on a synthetic band-power task with the upload flag
        import numpy as np
        from dvxr.serve.live import run_screening_live
        sys.path.insert(0, str(ROOT / "tests"))
        from test_live import _fake_bandpower_task, _synthetic_screener
        out = run_screening_live(_synthetic_screener(), _fake_bandpower_task(), "B",
                                 validated=False, source="upload")
        self.assertFalse(out["validated"])
        self.assertEqual(out["source"], "upload")

    def test_upload_surfaces_carry_ood_disclaimer(self):
        # the app, the CLI screen path, and the loader must all disclaim OOD uploads
        for rel in ("scripts/screen_app.py", "src/dvxr/cli.py", "src/dvxr/serve/live.py"):
            text = (ROOT / rel).read_text().lower()
            self.assertTrue("out-of-distribution" in text or "out of distribution" in text
                            or "illustrative" in text,
                            f"{rel}: upload path missing the out-of-distribution disclaimer")


if __name__ == "__main__":
    unittest.main()
