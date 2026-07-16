"""PR14 / Gate 5: the realtime intervention layer routes through the ONE central policy engine (it no
longer hard-codes its own action ids), and the heuristic streaming monitor is explicitly labelled
experimental (spec §14, §17)."""
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.realtime.intervention import evaluate_interventions  # noqa: E402
from dvxr.safety.policy import ACTION_REGISTRY, select_action  # noqa: E402


class InterventionUsesPolicyEngineTest(unittest.TestCase):
    def test_action_matches_the_policy_engine_not_a_hard_coded_id(self):
        # hypoglycaemia (high risk, confident) on good data → the engine escalates
        recs = evaluate_interventions({"glucose_now": 60, "data_quality": "good"}, role="clinician")
        self.assertTrue(recs)
        top = recs[0]
        engine = select_action(risk_category="high", confidence=0.9, data_quality="good",
                               role="clinician")
        self.assertEqual(top.action_id, engine.action_id)          # single authority
        self.assertIn(top.action_id, ACTION_REGISTRY)

    def test_poor_data_downgrades_to_verification_not_a_warning(self):
        # same physiological signal but poor data → engine asks to verify, not escalate
        recs = evaluate_interventions({"glucose_now": 60, "data_quality": "poor"}, role="clinician")
        self.assertEqual(recs[0].action_id, "VERIFY_SENSOR_AND_CGM")

    def test_reason_codes_come_from_the_engine(self):
        recs = evaluate_interventions({"glucose_now": 60, "data_quality": "good"}, role="clinician")
        self.assertTrue(recs[0].reason_codes)     # populated by select_action, not the rule


class HeuristicDemoIsLabelledTest(unittest.TestCase):
    def test_renamed_module_is_flagged_experimental(self):
        from dvxr.realtime import heuristic_demo
        self.assertTrue(getattr(heuristic_demo, "EXPERIMENTAL_ONLY", False))
        self.assertTrue(getattr(heuristic_demo, "NOT_FOR_CLINICAL_INFERENCE", False))

    def test_old_monitor_path_warns_but_still_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib
            import dvxr.realtime.monitor as m
            importlib.reload(m)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertTrue(hasattr(m, "FusedRealtimeMonitor"))


if __name__ == "__main__":
    unittest.main()
