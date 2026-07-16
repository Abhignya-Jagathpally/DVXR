"""PR8: the dashboard assembles six panels that visibly separate data / prediction / explanation /
action, and surfaces abstention honestly (spec §13, §20)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.serve.panels import build_report_panels  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402


class ReportPanelsTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")

    def _panels(self, **ov):
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher", **ov)
        return build_report_panels(req, prediction_store=self.pred, audit_store=self.audit,
                                   consent_store=self.consent, require_consent=False)

    def test_all_six_panels_present(self):
        panels = self._panels()["panels"]
        for name in ("context", "data_readiness", "prediction", "why", "next_action",
                     "evidence_provenance"):
            self.assertIn(name, panels)

    def test_abstention_is_surfaced_in_data_and_prediction(self):
        out = self._panels()
        self.assertEqual(out["status"], "abstained")
        self.assertTrue(out["panels"]["data_readiness"]["abstained"])
        self.assertIsNone(out["panels"]["prediction"]["risk"])   # no fabricated number

    def test_next_action_is_policy_selected_with_controls(self):
        p = self._panels()["panels"]["next_action"]
        self.assertEqual(p["action_id"], "INSUFFICIENT_DATA")
        self.assertTrue(p["action_text"])
        self.assertIn("escalate", p["controls"])

    def test_provenance_has_ids_versions_and_disclaimer(self):
        prov = self._panels()["panels"]["evidence_provenance"]
        self.assertTrue(prov["request_id"])
        self.assertTrue(prov["prediction_id"])
        self.assertTrue(prov["model_version"])
        self.assertIn("not a diagnosis", prov["disclaimer"].lower())

    def test_data_prediction_explanation_action_are_distinct_panels(self):
        # the separation the dashboard must make visible: each concern in its own panel
        panels = self._panels()["panels"]
        self.assertIn("missing_modalities", panels["data_readiness"])
        self.assertIn("risk", panels["prediction"])
        self.assertIn("supporting_factors", panels["why"])
        self.assertIn("action_id", panels["next_action"])


class AppRenderTest(unittest.TestCase):
    def test_markdown_renderer_is_importable_and_covers_six_panels(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.screen_app import report_panels_markdown
        pred, audit, consent, _m = open_local_stores(":memory:")
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk",
                              user_role="researcher")
        out = build_report_panels(req, prediction_store=pred, audit_store=audit,
                                  consent_store=consent, require_consent=False)
        md = report_panels_markdown(out).lower()
        for heading in ("context", "data readiness", "prediction", "why", "next action",
                        "evidence & provenance"):
            self.assertIn(heading, md)
        self.assertIn("abstained", md)
        self.assertIn("not a diagnosis", md)


if __name__ == "__main__":
    unittest.main()
