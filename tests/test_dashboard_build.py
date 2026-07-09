"""tests/test_dashboard_build.py — dashboard replay build + intervention wiring.

Data-heavy build assertions are skip-guarded on dataset presence; the intervention
and HTML-self-containment checks run everywhere (no data needed).

Run:  venv/bin/python -m unittest tests.test_dashboard_build
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_dashboard as bd  # noqa: E402
from dvxr.realtime.intervention import evaluate_interventions  # noqa: E402

WESAD_OK = (ROOT / "data" / "real" / "WESAD").exists()

_STEP_KEYS = {
    "t", "stress_prob", "stress_band", "glucose_now", "glucose_forecast",
    "glucose_lower", "glucose_upper", "present_modalities", "attribution",
    "interventions", "narration",
}


class TestInterventions(unittest.TestCase):
    """The reused rule engine must fire on hypo / hyper / high-stress states."""

    def _names(self, state):
        return {r.rule for r in evaluate_interventions(state)}

    def test_hypoglycemia_fires(self):
        self.assertIn("hypoglycemia_risk", self._names({"glucose_now": 55.0}))

    def test_hyperglycemia_fires(self):
        self.assertIn("hyperglycemia_risk", self._names({"glucose_now": 250.0}))

    def test_high_stress_fires(self):
        self.assertIn("high_stress", self._names({"stress_band": "high",
                                                  "stress_probability": 0.92}))

    def test_calm_state_fires_nothing(self):
        self.assertEqual(self._names({"glucose_now": 105.0, "stress_band": "low"}), set())


class TestRenderHtmlSelfContained(unittest.TestCase):
    """The HTML must embed the JSON and make no external requests (offline-safe)."""

    def setUp(self):
        step = {k: (None if k not in ("present_modalities", "attribution", "interventions")
                    else ([] if k != "attribution" else {})) for k in _STEP_KEYS}
        step.update(stress_prob=0.4, stress_band="watch", present_modalities=["ECG"],
                    attribution={"ECG": 1.0}, narration="demo", t="2026-01-01T00:00:00")
        self.replays = {"demo": {
            "task": "demo", "title": "Demo", "kind": "classification",
            "test_subject": "S1", "modalities": ["ECG"], "attribution_source": "occlusion",
            "grounded_facts": "- demo", "insight": "x\n\nCaveat: research prototype.",
            "runs": {"full": {"label": "Full", "dropped": None, "drop_from_step": None,
                              "steps": [step, dict(step)]},
                     "dropout": {"label": "Drop", "dropped": "ECG", "drop_from_step": 1,
                                 "steps": [step, dict(step)]}}}}

    def test_placeholder_replaced_and_json_embedded(self):
        html = bd.render_html(self.replays)
        self.assertNotIn("__DATA_JSON__", html)
        start = html.index('<script id="data"')
        blob = html[html.index(">", start) + 1: html.index("</script>", start)]
        parsed = json.loads(blob)
        self.assertIn("demo", parsed["replays"])

    def test_no_external_requests(self):
        html = bd.render_html(self.replays)
        for needle in ("http://", "https://", "src=", "<link", "@import", "//cdn"):
            self.assertNotIn(needle, html, f"self-contained page must not contain {needle!r}")


@unittest.skipUnless(WESAD_OK, "WESAD dataset not present")
class TestBuildStressReplay(unittest.TestCase):
    """A real (fast, 1-epoch) build must produce a non-empty, well-formed replay."""

    @classmethod
    def setUpClass(cls):
        bd.STRESS_EPOCHS = 1
        bd.MODALITY_DROPOUT = 0.3
        bd._llm_attribution_or_none = staticmethod(lambda task: None)  # skip LLM load
        cls.rep = bd.build_stress_replay()

    def test_non_empty_steps(self):
        steps = self.rep["runs"]["full"]["steps"]
        self.assertGreater(len(steps), 0)

    def test_expected_keys(self):
        step = self.rep["runs"]["full"]["steps"][0]
        self.assertTrue(_STEP_KEYS.issubset(step.keys()))

    def test_probabilities_in_range(self):
        for s in self.rep["runs"]["full"]["steps"]:
            self.assertGreaterEqual(s["stress_prob"], 0.0)
            self.assertLessEqual(s["stress_prob"], 1.0)
            self.assertIn(s["stress_band"], {"low", "watch", "elevated", "high"})

    def test_dropout_shows_modality_change(self):
        run = self.rep["runs"]["dropout"]
        df = run["drop_from_step"]
        before = set(run["steps"][df - 1]["present_modalities"])
        after = set(run["steps"][df]["present_modalities"])
        self.assertTrue(before - after, "dropout run must remove a modality mid-stream")

    def test_json_round_trips(self):
        blob = json.dumps(self.rep)
        self.assertEqual(json.loads(blob)["task"], "wesad_stress")


if __name__ == "__main__":
    unittest.main()
