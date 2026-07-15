"""Tests for the glass-box tracer (dvxr.serve.glassbox).

The synthetic path is torch-free so it runs in the base (no-torch) CI that the honesty audit uses. The
real path (fits a screener + runs LaBraM/VQ/Qwen) is slow and dependency-heavy, so it is opt-in via
DVXR_GLASSBOX_REAL=1. Both paths must uphold the honesty contract: the proposed multimodal path is shown
as-is (never framed as a win), a sample entry is out-of-distribution, and every trace disclaims diagnosis.
"""
import json
import os
import unittest

from dvxr.serve import glassbox


class SyntheticTraceStructure(unittest.TestCase):
    def setUp(self):
        # force the deterministic fixture — no torch/data needed
        self.t = glassbox._synthetic_trace("wesad_stress", note="unit-test fixture")

    def test_has_both_pipelines(self):
        self.assertTrue(self.t.winner and self.t.proposed)
        self.assertIsNotNone(self.t.winner["probability"])
        self.assertIsNotNone(self.t.proposed["probability"])

    def test_attention_is_a_distribution(self):
        att = self.t.proposed["attention"]
        self.assertGreater(len(att), 1)
        self.assertAlmostEqual(sum(att.values()), 1.0, places=2)

    def test_vq_perplexity_in_range(self):
        for m, d in self.t.proposed["vq"].items():
            n = d["n_codes"]
            self.assertGreater(d["perplexity"], 1.0, f"{m}: perplexity must exceed 1")
            self.assertLessEqual(d["perplexity"], n + 1e-6, f"{m}: perplexity <= K")
            self.assertTrue(all(0 <= c < n for c in d["codes"]), f"{m}: code out of range")

    def test_proposed_is_shown_as_is_not_a_win(self):
        note = self.t.proposed["note"].lower()
        self.assertIn("underperform", note)
        for win_word in ("beats the winner", "outperforms the winner", "proposed wins"):
            self.assertNotIn(win_word, note)

    def test_sample_entry_is_out_of_distribution(self):
        # the synthetic / upload paths are never presented as the validated cohort number
        self.assertFalse(self.t.validated)
        self.assertTrue(self.t.note)

    def test_disclaimer_present(self):
        self.assertIn("not a diagnosis", self.t.disclaimer.lower())

    def test_json_serializable(self):
        # the renderer serializes the trace into the offline HTML
        json.dumps(self.t.to_dict())


class ScoreboardPanel(unittest.TestCase):
    def test_panel_is_failsoft_and_structured(self):
        panel = glassbox.scoreboard_panel("wesad_stress")
        self.assertEqual(panel["task"], "wesad_stress")
        self.assertIn("full_observation", panel)
        self.assertIn("dropout_crossover", panel)
        # if the committed board carries WESAD, the proposed-vs-baseline verdict is present and honest
        fo = panel["full_observation"]
        if fo is not None:
            self.assertIn("best_baseline", fo)
            self.assertIn("verdict", fo)

    def test_unknown_task_does_not_crash(self):
        panel = glassbox.scoreboard_panel("no_such_task")
        self.assertIsNone(panel["full_observation"])


class TracePipelineFallback(unittest.TestCase):
    def test_bad_task_degrades_to_synthetic(self):
        # an unknown task must not raise — it degrades to the flagged synthetic fixture
        t = glassbox.trace_pipeline("definitely_not_a_task")
        self.assertTrue(t.note)
        self.assertFalse(t.validated)
        self.assertIsNotNone(t.proposed["probability"])


@unittest.skipUnless(os.environ.get("DVXR_GLASSBOX_REAL"), "set DVXR_GLASSBOX_REAL=1 for the real path")
class RealTraceSmoke(unittest.TestCase):
    def test_real_wesad_trace(self):
        t = glassbox.trace_pipeline("wesad_stress", include_llm=True)
        self.assertTrue(t.validated)
        self.assertEqual(t.source, "cohort")
        self.assertIsNotNone(t.winner["probability"])
        self.assertGreater(len(t.proposed["vq"]), 1)
        self.assertAlmostEqual(sum(t.proposed["attention"].values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
