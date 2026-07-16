"""PR29 / Gate A (spec §2, §18): consent is more than a purpose list — the check honors revocation,
effective dates / expiry (against the request's causal cutoff, no wall-clock), modality scope, and
study scope. Fail-closed on every unmet dimension; the simple {purposes:[...]} form still works."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.storage import open_local_stores  # noqa: E402


class ConsentRichnessTest(unittest.TestCase):
    def setUp(self):
        _p, _a, self.consent, _m = open_local_stores(":memory:")

    def _set(self, scope):
        self.consent.set_scope("P1", scope)

    def test_simple_purposes_form_still_works(self):
        self._set({"purposes": ["research"]})
        self.assertTrue(self.consent.check("P1", "research"))
        self.assertFalse(self.consent.check("P1", "clinical"))

    def test_revoked_consent_is_denied(self):
        self._set({"purposes": ["research"], "revoked": True})
        self.assertFalse(self.consent.check("P1", "research"))

    def test_expired_consent_is_denied_as_of_cutoff(self):
        self._set({"purposes": ["research"], "expires_at": "2026-01-01T00:00:00Z"})
        self.assertTrue(self.consent.check("P1", "research", as_of="2025-12-31T00:00:00Z"))
        self.assertFalse(self.consent.check("P1", "research", as_of="2026-02-01T00:00:00Z"))

    def test_not_yet_effective_consent_is_denied(self):
        self._set({"purposes": ["research"], "effective_from": "2026-06-01T00:00:00Z"})
        self.assertFalse(self.consent.check("P1", "research", as_of="2026-01-01T00:00:00Z"))
        self.assertTrue(self.consent.check("P1", "research", as_of="2026-07-01T00:00:00Z"))

    def test_modality_scope_is_enforced(self):
        self._set({"purposes": ["research"], "modalities": ["cgm"]})
        self.assertTrue(self.consent.check("P1", "research", modality="cgm"))
        self.assertFalse(self.consent.check("P1", "research", modality="eeg"))

    def test_study_scope_is_enforced(self):
        self._set({"purposes": ["research"], "study_id": "STUDY-A"})
        self.assertTrue(self.consent.check("P1", "research", study_id="STUDY-A"))
        self.assertFalse(self.consent.check("P1", "research", study_id="STUDY-B"))

    def test_unparseable_as_of_when_temporal_check_requested_is_denied(self):
        self._set({"purposes": ["research"], "expires_at": "2026-01-01T00:00:00Z"})
        self.assertFalse(self.consent.check("P1", "research", as_of="not-a-date"))


if __name__ == "__main__":
    unittest.main()
