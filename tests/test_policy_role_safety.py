"""PR24 / P0-3 (spec §14): a role restriction changes who-sees-what and who-is-notified, but NEVER
silently lowers urgency. A participant's high-risk state must not collapse to "Continue monitoring"."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.safety.policy import select_action  # noqa: E402


class RoleSafetyTest(unittest.TestCase):
    def _act(self, role, **ov):
        base = dict(risk_category="high", confidence=0.9, data_quality="good", role=role)
        base.update(ov)
        return select_action(**base)

    def test_participant_high_risk_is_urgent_not_downgraded(self):
        d = self._act("participant")
        self.assertNotEqual(d.action_id, "CONTINUE_MONITORING")     # never downgraded
        self.assertEqual(d.action_id, "CONTACT_APPROVED_CARE_CHANNEL")
        self.assertTrue(d.requires_clinician_review)                # urgency + review preserved
        # the clinician escalation still fires internally
        self.assertEqual(d.system_action_id, "ESCALATE_PER_APPROVED_PROTOCOL")

    def test_participant_elevated_awaits_review_not_monitoring(self):
        d = self._act("participant", risk_category="elevated", confidence=0.5)
        self.assertNotEqual(d.action_id, "CONTINUE_MONITORING")
        self.assertEqual(d.action_id, "AWAIT_CLINICIAN_REVIEW")
        self.assertTrue(d.requires_clinician_review)
        self.assertEqual(d.system_action_id, "REVIEW_ELEVATED_RISK")

    def test_researcher_high_risk_is_not_downgraded_to_verify(self):
        # previously a researcher's high-risk escalation collapsed to VERIFY_SENSOR_AND_CGM
        d = self._act("researcher")
        self.assertEqual(d.action_id, "CONTACT_APPROVED_CARE_CHANNEL")
        self.assertTrue(d.requires_clinician_review)
        self.assertEqual(d.system_action_id, "ESCALATE_PER_APPROVED_PROTOCOL")

    def test_clinician_escalation_is_unchanged(self):
        d = self._act("clinician")
        self.assertEqual(d.action_id, "ESCALATE_PER_APPROVED_PROTOCOL")
        self.assertEqual(d.system_action_id, "ESCALATE_PER_APPROVED_PROTOCOL")
        self.assertTrue(d.requires_clinician_review)

    def test_low_risk_still_continues_monitoring_for_participant(self):
        d = self._act("participant", risk_category="low", confidence=0.9)
        self.assertEqual(d.action_id, "CONTINUE_MONITORING")        # genuinely low risk is fine


if __name__ == "__main__":
    unittest.main()
