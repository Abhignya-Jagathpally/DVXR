"""Tests for the per-subject report artifact (dvxr.serve.report).

Renders on a synthetic live-result dict (no data/LaBraM): asserts the HTML is self-contained, shows
both AUROC granularities, the per-window trace, drivers, the grounded note, and the not-a-diagnosis
caveat. A gated end-to-end write is covered by the CLI on real cohorts.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _Screener:
    heldout = {"auroc": 0.961, "auroc_ci": [0.942, 0.976], "auroc_subject": 0.986,
               "auroc_subject_ci": [0.966, 0.999], "ece": 0.03, "protocol": "3x5 subject-held-out CV",
               "n_subjects": 58}
    meta = {"literature": ["LaBraM — Jiang et al., ICLR 2024, arXiv:2405.18765"]}


def _report(validated=True):
    return {
        "subject": "MDD_S9",
        "result": {"label": "Depression screen (MDD vs healthy) from resting EEG",
                   "probability": 0.86, "risk_band": "high", "interval": [0.66, 1.0], "n_windows": 14},
        "window_probs": [0.7, 0.8, 0.9, 0.85, 0.88, 0.82, 0.9],
        "drivers": [{"direction": "raises", "feature": "latent[7]", "contribution": 0.5},
                    {"direction": "lowers", "feature": "latent[3]", "contribution": -0.2}],
        "narrative": {"clinician": "Clinician-facing summary (grounded): elevated depression screen. "
                                   "Caveat: research prototype, not a diagnosis."},
        "embed_meta": {"encoder": "real LaBraM EEG foundation model (frozen)"},
        "validated": validated,
        "evidence": {"window_auroc": 0.961, "window_ci": [0.942, 0.976], "subject_auroc": 0.986,
                     "subject_ci": [0.966, 0.999], "ece": 0.03, "protocol": "3x5 subject-held-out CV",
                     "n_subjects": 58, "literature": _Screener.meta["literature"],
                     "decision_curve": {
                         "prevalence": 0.5, "n": 58, "level": "subject",
                         "points": [{"threshold": t / 100, "model": 0.4, "all": 0.3, "none": 0.0}
                                    for t in (5, 25, 50, 75)],
                         "summary": {"useful": True, "useful_band": [0.05, 0.75],
                                     "best_threshold": 0.25, "best_gain": 0.1, "best_gain_lo": 0.07,
                                     "note": "Screening beats both treat-all and treat-none for "
                                             "decision thresholds 5-75%."}}},
    }


class ReportRenderTest(unittest.TestCase):
    def test_self_contained_and_complete(self):
        from dvxr.serve.report import render_report_html
        h = render_report_html(_report(), _Screener())
        self.assertIn("<!DOCTYPE html>", h)
        self.assertNotIn("http://", h)
        self.assertNotIn("https://", h)                       # self-contained
        self.assertIn("0.961", h)                              # window-level
        self.assertIn("0.986", h)                              # subject-level
        self.assertIn("<svg", h)                               # per-window trace
        self.assertIn("latent[7]", h)                          # drivers
        self.assertIn("not a diagnosis", " ".join(h.lower().split()))

    def test_decision_curve_panel_rendered_with_attribution(self):
        from dvxr.serve.report import render_report_html
        h = render_report_html(_report(), _Screener())
        low = " ".join(h.lower().split())
        self.assertIn("decision-curve analysis", low)      # clinical-utility panel present
        self.assertIn("net benefit", low)                  # SVG axis label
        self.assertIn("vickers", low)                      # method attribution
        self.assertIn("treat-all", low)                    # honest comparison policies
        self.assertNotIn("http://", h)
        self.assertNotIn("https://", h)                    # SVG self-contained (currentColor)

    def test_upload_report_shows_ood_banner(self):
        from dvxr.serve.report import render_report_html
        h = render_report_html(_report(validated=False), _Screener())
        self.assertIn("out of distribution", h.lower())

    def test_sparkline_handles_short_traces(self):
        from dvxr.serve.report import _sparkline
        self.assertIn("<svg", _sparkline([0.5]))
        self.assertIn("polyline", _sparkline([0.1, 0.9]))


if __name__ == "__main__":
    unittest.main()
