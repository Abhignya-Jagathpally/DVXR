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

    def test_every_traceability_source_is_git_tracked(self):
        """The clean-clone guarantee: every source a claim verifies against must be COMMITTED, not
        just present locally. A source that exists only as an untracked artifact makes the "audit
        green on a fresh checkout" promise false — the exact gap this test now blocks."""
        import subprocess
        from dvxr.serve.evidence import PRODUCT_CLAIMS
        paths = {c.source_file for c in PRODUCT_CLAIMS}
        paths |= {f"{c.verify_manifest}/manifest.json" for c in PRODUCT_CLAIMS if c.verify_manifest}
        for rel in sorted(paths):
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                     cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(tracked.returncode, 0,
                             f"traceability source not committed to git: {rel}")

    def test_provenance_verifier_passes(self):
        """The offline provenance tool must confirm the committed board matches the manifest — the
        network-free half of `docs/REPRODUCE.md` (the other half is the real re-run recorded there).

        Load the tool deterministically from the committed ``scripts/`` path via importlib so a
        stray same-named module elsewhere on sys.path (e.g. an untracked file at repo root) cannot
        shadow the audited one."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_dvxr_prov_build_dnh_labram_scoreboard", ROOT / "scripts" / "build_dnh_labram_scoreboard.py")
        prov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prov)
        self.assertEqual(prov.verify(), 0)

    def test_missing_source_degrades_to_problem_string_not_crash(self):
        """A missing source must surface as a traceable audit failure, never an uncaught exception —
        so this class of gap can't again masquerade as an unrelated stack trace."""
        from dvxr.serve import evidence
        self.assertIsNone(evidence._read_scoreboard("outputs/_no_such_board.csv"))
        self.assertIsNone(evidence._manifest_auroc("outputs/_no_such_screener"))

    def test_depression_headline_is_consistent_across_every_committed_artifact(self):
        """Cross-artifact traceability: the depression headline (AUROC 0.9608 / base_err 0.0392 /
        rounded 0.961) must agree across EVERY committed surface that carries it — the served screener
        manifest, the benchmark board, the evidence.py pin, and the findings doc. A number with many
        homes drifts unless they are checked against each other; this makes that drift a test failure."""
        import csv as _csv
        from dvxr.serve import evidence
        # 1. served screener manifest — the trained artifact
        manifest_auroc = evidence._manifest_auroc("outputs/product/screeners/mumtaz_depression")
        self.assertIsNotNone(manifest_auroc)
        self.assertAlmostEqual(manifest_auroc, 0.9608, places=3)
        # 2. committed benchmark board — the audited source
        board = evidence._read_scoreboard("outputs/_dnh_labram/benchmark_scoreboard.csv")
        self.assertIsNotNone(board, "committed _dnh_labram board is missing")
        board_err = float(board["mumtaz_depression"]["base_err"])
        self.assertAlmostEqual(board_err, round(1.0 - manifest_auroc, 4), places=3,
                               msg="board base_err disagrees with the manifest AUROC")
        # 3. evidence.py pin — the product claim
        claim = next(c for c in evidence.PRODUCT_CLAIMS if c.task == "mumtaz_depression")
        self.assertAlmostEqual(claim.source_err, board_err, places=3)
        self.assertAlmostEqual(claim.auroc, round(1.0 - board_err, 3), places=3)
        # 4. findings doc — the prose record
        findings = (ROOT / "BENCHMARK_FINDINGS.md").read_text()
        self.assertIn("0.0392", findings)
        self.assertIn("0.961", findings)

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


class ProductVisionAudit(unittest.TestCase):
    """The re-headlined glucose product (NeuroGlycemic Sentinel) must be presented as research-stage,
    never as a validated claim: no fabricated AUROC, synchrony-gated, clearly not-yet-validated. This
    is the honesty guardrail on the glucose re-headline — the product VISION is real, but the validated
    NUMBERS remain the components (depression / stress / workload). If this class fails, the re-headline
    has manufactured a claim the data cannot support."""

    def test_vision_is_research_stage_and_synchrony_gated(self):
        from dvxr.serve.evidence import PRODUCT_VISION
        self.assertTrue(PRODUCT_VISION.research_stage, "glucose product must be flagged research-stage")
        self.assertTrue(PRODUCT_VISION.requires_synchronized_data,
                        "glucose product must require synchronized same-subject data")

    def test_vision_carries_no_fabricated_auroc(self):
        from dvxr.serve.evidence import PRODUCT_VISION
        self.assertIsNone(PRODUCT_VISION.auroc, "the glucose product must not carry a headline AUROC")

    def test_vision_caveat_names_synchronized_data_and_not_validated(self):
        from dvxr.serve.evidence import PRODUCT_VISION
        low = PRODUCT_VISION.caveat.lower()
        self.assertIn("synchronized", low)
        self.assertIn("not yet validated", low)
        self.assertIn("not a diagnosis", low)

    def test_report_marks_glucose_product_research_stage_without_a_number(self):
        from dvxr.serve.evidence import render_report, PRODUCT_VISION
        r = render_report()
        self.assertIn("research-stage", r.lower())
        self.assertIn(PRODUCT_VISION.name, r)
        # the glucose product's own lines must never carry a fabricated AUROC
        for line in r.splitlines():
            if PRODUCT_VISION.name in line:
                self.assertNotRegex(line, r"AUROC\s*0?\.\d",
                                    "glucose product line carries a fabricated AUROC")

    def test_validated_components_still_trace(self):
        """The re-headline must not delete the component validation it stands on."""
        from dvxr.serve.evidence import PRODUCT_CLAIMS, PRODUCT_VISION
        claim_tasks = {c.task for c in PRODUCT_CLAIMS}
        for comp in PRODUCT_VISION.components:
            self.assertIn(comp, claim_tasks, f"validated component {comp} lost its scoreboard claim")

    def test_glucose_diabetes_fusion_remain_excluded_as_validated_claims(self):
        """Re-headlining to glucose must NOT relax the exclusion of the leaky/negative results."""
        from dvxr.serve.evidence import EXCLUDED_CLAIMS
        for k in ("cgmacros_diabetes", "cacmf_as_win", "llm_as_predictor"):
            self.assertIn(k, EXCLUDED_CLAIMS, f"{k} must stay excluded as a validated claim")

    def test_fused_report_type_has_no_committed_artifact_and_abstains(self):
        """PR34 guardrail: the model resolver must ALWAYS return an abstaining service for the fused
        stress_glucose_risk report type — no synchronized cohort exists, so no fused artifact can be
        committed. Even with a fully populated registry, the fused path can never serve a number."""
        from dvxr.prediction.registry import resolve_predictor
        from dvxr.prediction.service import PredictionInputs
        svc = resolve_predictor("stress_glucose_risk", model_registry=None, artifact_root=None)
        b = svc.predict(PredictionInputs("stress_glucose_risk", [30, 60],
                                         requested_modalities=["cgm", "eeg", "wearable_phys"]))
        self.assertTrue(b.abstained, "the fused stress-glucose report must abstain (no synchronized data)")
        self.assertIsNone(b.risk)
        self.assertIsNone(b.forecast, "an abstaining fused report must carry no fabricated forecast")


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

    def test_cgm_wearable_is_not_claimed_to_add_value(self):
        # PR32 / P0-4: the corrected participant-level ablation found NO wearable benefit. The model
        # card must not re-inflate the superseded "+0.042 AUROC ... adds value" claim.
        mc = ROOT / "docs" / "MODEL_CARD.md"
        if not mc.exists():
            self.skipTest("model card absent")
        text = mc.read_text()
        low = text.lower()
        # the honest negative result must be stated (whitespace/markdown-tolerant)...
        norm = " ".join(low.replace("**", "").split())
        self.assertIn("does not add value", norm,
                      "model card must state the corrected CGM+wearable negative result")
        # ...and the old positive delta may appear ONLY as a superseded/artifact reference
        for line in text.splitlines():
            if "0.042" in line:
                self.assertTrue(
                    any(w in line.lower() for w in ("supersede", "artifact", "earlier", "flawed")),
                    f"stale positive +0.042 claim not marked as superseded: {line!r}")

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


class GlassBoxSurfaceAudit(unittest.TestCase):
    """The glass-box demo shows the proposed multimodal path as-is: it must disclaim diagnosis, flag a
    sample as out-of-distribution, state the honest full-observation loss, never sell fusion/LLM as a
    full-obs win, and load no external resource (offline / CSP-safe)."""

    def _page(self, n: int = 2) -> str:
        from dvxr.serve.glassbox import _synthetic_trace
        from dvxr.serve.glassbox_render import render_glassbox
        traces = []
        for i in range(n):
            t = _synthetic_trace("wesad_stress", note="audit fixture").to_dict()
            t["subject"] = f"S{i}"
            traces.append(t)
        return render_glassbox(traces)

    def test_no_external_resource_loads(self):
        # inline <script>/<style> are allowed; loading from any external host is not.
        page = self._page().lower()
        for bad in ("src=", "@import", "url(http", "<link", "<script src", "http://", "https://"):
            self.assertNotIn(bad, page, f"glass-box loads an external resource: {bad}")

    def test_disclaims_diagnosis(self):
        page = self._page(1).lower()
        self.assertTrue("not a diagnosis" in page or "not a diagnostic" in page,
                        "glass-box missing the not-a-diagnosis disclaimer")

    def test_states_the_honest_full_obs_loss(self):
        self.assertIn("loses on full-observation accuracy", self._page(1).lower(),
                      "glass-box must state the proposed path's full-observation loss")

    def test_never_sells_fusion_or_llm_as_a_full_obs_win(self):
        page = self._page().lower()
        for bad in ("fusion outperforms", "cacmf outperforms", "cacmf wins", "learned fusion wins",
                    "outperforms the single-modality", "fusion is the win", "llm-based prediction wins"):
            self.assertNotIn(bad, page, f"glass-box frames the proposed path as a full-obs win: {bad!r}")

    def test_sample_entry_is_flagged_out_of_distribution(self):
        # the synthetic/upload trace is not validated → the OOD badge must appear
        self.assertIn("out-of-distribution", self._page(1).lower())


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


class UserFacingWebAudit(unittest.TestCase):
    """The user-facing web surface (dvxr.serve.asgi + src/dvxr/web) must never present fabricated
    numbers as real: it renders only backend-returned fields, its one canned example is visibly labeled
    'illustrative / no live model executed', and it carries the not-a-diagnosis disclaimer. It must also
    load no external resource (offline / CSP-safe), like every other DVXR surface."""

    def _web(self, rel: str) -> str:
        return (ROOT / "src" / "dvxr" / "web" / rel).read_text()

    def test_landing_page_discloses_research_stage_and_not_a_diagnosis(self):
        page = self._web("index.html").lower()
        self.assertTrue("not a diagnos" in page or "never a diagnos" in page,
                        "landing page missing the not-a-diagnosis disclaimer")

    def test_illustrative_example_is_visibly_labeled(self):
        # index.html must carry a visible illustrative banner; app.js must mark the canned sample as an
        # interface-only object with no model executed — so it can never be mistaken for a real result.
        page = self._web("index.html").lower()
        self.assertIn("illustrative", page, "landing page missing the illustrative-example labeling")
        js = self._web("assets/app.js")
        self.assertIn("illustrative-interface/no-model-executed", js,
                      "the canned sample is not marked as no-model-executed")
        self.assertIn("illustrative: true", js.replace('"true"', "true"),
                      "the canned sample is not flagged illustrative:true")

    def test_fused_report_status_is_abstains_in_the_ui_config(self):
        # the machine-readable UI config must advertise the fused report as abstaining, never as a number
        asgi = (ROOT / "src" / "dvxr" / "serve" / "asgi.py").read_text()
        self.assertIn("abstains_until_synchronized_artifact", asgi,
                      "asgi /ui/config must report the fused report as abstaining")

    def test_web_surface_loads_no_external_resource(self):
        html = self._web("index.html")
        for bad in ("http://", "https://", "//cdn", "<script src=\"http"):
            self.assertNotIn(bad, html, f"landing page loads an external resource: {bad}")
        css = self._web("assets/styles.css")
        self.assertNotIn("url(http", css, "stylesheet loads an external resource")


if __name__ == "__main__":
    unittest.main()
