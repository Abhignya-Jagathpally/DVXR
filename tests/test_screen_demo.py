"""Tests for the screener-backed demo builder (scripts/build_screen_demo.py).

The renderer + subject-picker are exercised on synthetic inputs (fast, no data). The full build
(fitting real screeners) is covered by the CLI/serve integration tests and is not repeated here.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class DemoRenderTest(unittest.TestCase):
    def _panel(self):
        return {"task": "t", "title": "Depression screen from resting EEG", "icon": "🧠",
                "blurb": "b", "encoder": "real LaBraM EEG foundation model", "caveat": "not a diagnosis",
                "literature": ["LaBraM — Jiang et al., ICLR 2024"], "band_thresholds": {"low": 0.25},
                "heldout": {"auroc": 0.9608, "auroc_ci": [0.9417, 0.9756], "ece": 0.03,
                            "n_subjects": 58, "n_windows": 812, "protocol": "3x5 subject-held-out CV"},
                "cards": [{"subject": "MDD_S9", "truth": 1,
                           "result": {"probability": 0.81, "risk_band": "high",
                                      "interval": [0.6, 1.0], "n_windows": 14},
                           "drivers": [{"direction": "raises", "feature": "latent[7]",
                                        "contribution": 0.5}]}]}

    def test_render_html_is_self_contained_and_honest(self):
        from build_screen_demo import render_html
        h = render_html([self._panel()], [("Some task", "weights unavailable")])
        self.assertIn("<!DOCTYPE html>", h)
        self.assertIn("DVXR Screen", h)
        self.assertIn("0.9608", h)                 # benchmark-reproduced number is displayed
        self.assertIn("Not a diagnosis", h)        # honesty banner
        self.assertIn("Some task", h)              # skipped-panel honesty
        self.assertNotIn("http://", h)             # no external resources
        self.assertNotIn("https://", h)

    def test_gauge_svg_valid(self):
        from build_screen_demo import _gauge_svg
        svg = _gauge_svg(0.5, "watch", 0.3, 0.7)
        self.assertIn("<svg", svg)
        self.assertIn("0.50", svg)

    def test_pick_subjects_prefers_case_and_control(self):
        from build_screen_demo import _pick_subjects
        subs = np.array(["a", "a", "b", "b", "c", "c"])
        y = np.array([1, 1, 0, 0, 1, 1])
        picked = _pick_subjects(subs, y, k=2)
        self.assertEqual(len(picked), 2)
        labels = {int(round(float(y[subs == s].mean()))) for s in picked}
        self.assertIn(1, labels)
        self.assertIn(0, labels)


if __name__ == "__main__":
    unittest.main()
