"""Tests for clinical-utility / decision-curve analysis (dvxr.serve.utility).

Checks the net-benefit math against the Vickers & Elkin (2006) definition on hand-computable cases,
confirms a perfect classifier is useful over the whole band while a non-informative one is not, and
that the rendered SVG is self-contained (no external resource loads — honesty-audit invariant).
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.serve.utility import (  # noqa: E402
    decision_curve, net_benefit, render_decision_curve_svg, subject_aggregate)


class NetBenefitMathTest(unittest.TestCase):
    def test_matches_vickers_formula(self):
        # 10 subjects, 4 positive. Threshold 0.2 → odds weight 0.25.
        y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        prob = [0.9, 0.8, 0.7, 0.1, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05]
        # flagged (>=0.2): 3 TP (0.9,0.8,0.7), 1 FP (0.6). NB = 3/10 - (1/10)(0.2/0.8) = 0.3-0.025.
        self.assertAlmostEqual(net_benefit(y, prob, 0.2), 0.275, places=6)

    def test_treat_none_is_zero_and_degenerate_thresholds_safe(self):
        y = [1, 0, 1, 0]
        prob = [0.6, 0.4, 0.7, 0.2]
        self.assertEqual(net_benefit(y, prob, 0.0), 0.0)
        self.assertEqual(net_benefit(y, prob, 1.0), 0.0)


class DecisionCurveTest(unittest.TestCase):
    def test_perfect_classifier_is_useful_across_band(self):
        y = np.array([1] * 40 + [0] * 60)
        prob = np.where(y == 1, 0.99, 0.01)  # separated across the whole 0.05–0.75 band
        curve = decision_curve(y, prob)
        self.assertTrue(curve["summary"]["useful"])
        self.assertAlmostEqual(curve["prevalence"], 0.4, places=3)
        for p in curve["points"]:
            self.assertEqual(p["none"], 0.0)
            # perfect model catches every case with no false alarms → NB == prevalence throughout
            self.assertAlmostEqual(p["model"], 0.4, places=6)
            self.assertGreaterEqual(p["model"] + 1e-9, p["all"])  # never worse than treat-all

    def test_noninformative_classifier_is_not_useful(self):
        # A random score independent of the label: expected net benefit is dominated by the best
        # default, so with enough samples the bootstrap gate rejects the noise-level advantage.
        rng = np.random.default_rng(1)
        y = (rng.random(2000) < 0.3).astype(int)
        prob = rng.random(2000)
        curve = decision_curve(y, prob)
        self.assertFalse(curve["summary"]["useful"])
        self.assertLessEqual(curve["summary"]["best_gain_lo"], 0.0)

    def test_subject_aggregate_collapses_to_one_row_per_subject(self):
        subjects = np.array(["a", "a", "b", "b", "b"])
        prob = np.array([0.8, 0.6, 0.1, 0.2, 0.3])
        y = np.array([1, 1, 0, 0, 0])
        sy, sp = subject_aggregate(subjects, prob, y)
        self.assertEqual(list(sy), [1, 0])
        self.assertAlmostEqual(sp[0], 0.7, places=6)
        self.assertAlmostEqual(sp[1], 0.2, places=6)


class DecisionCurveSvgTest(unittest.TestCase):
    def test_svg_is_self_contained(self):
        curve = decision_curve([1, 1, 0, 0, 0], [0.9, 0.8, 0.2, 0.1, 0.3])
        svg = render_decision_curve_svg(curve)
        self.assertTrue(svg.startswith("<svg"))
        # no external resource loads
        self.assertNotIn("http", svg)
        self.assertFalse(re.search(r"(src=|@import|url\(http|<image)", svg))
        for legend in ("DVXR screen", "treat all", "treat none"):
            self.assertIn(legend, svg)


if __name__ == "__main__":
    unittest.main()
