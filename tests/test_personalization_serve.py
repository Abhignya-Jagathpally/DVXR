"""Tests for serve-time personalization wired into the Screener (dvxr-screener/2).

Per-subject recalibration applies only to within-subject STATE tasks (a subject carries both
classes); for subject-level-diagnosis tasks it is a correct no-op. Persistence is back-compatible:
v1 artifacts (no per-subject state) still load. The reported ECE gain is honest (per-subject split);
on tiny cohorts it may be negative — which we report, not hide.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FitPersonalizationTest(unittest.TestCase):
    def test_within_subject_task_is_applicable(self):
        from dvxr.serve.screener import _fit_personalization
        rng = np.random.default_rng(0)
        subs = np.repeat([f"s{i}" for i in range(6)], 30)
        y = np.tile([0, 1], 90)                                  # each subject has both classes
        oof = np.clip(y * 0.4 + 0.3 + rng.normal(0, 0.2, len(y)), 0, 1)
        cal, m = _fit_personalization(subs, oof, y, np.ones(len(y), bool))
        self.assertTrue(m["applicable"])
        self.assertIsNotNone(cal)
        self.assertIn("population_ece", m)
        self.assertIn("personalized_ece", m)
        self.assertEqual(m["n_personalized_subjects"], 6)

    def test_subject_level_task_is_noop(self):
        from dvxr.serve.screener import _fit_personalization
        subs = np.repeat([f"s{i}" for i in range(10)], 8)
        y = np.repeat(np.arange(10) % 2, 8)                      # one class per subject
        oof = np.clip(y * 0.5 + 0.25, 0, 1)
        cal, m = _fit_personalization(subs, oof, y, np.ones(len(y), bool))
        self.assertFalse(m["applicable"])
        self.assertIsNone(cal)


class ScreenerPersistenceV2Test(unittest.TestCase):
    def _synthetic(self, personal=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from dvxr.calibration import fit_platt_calibrator
        from dvxr.serve.screener import Screener
        rng = np.random.default_rng(0)
        n = 200
        y = (rng.random(n) < 0.5).astype(int)
        emb = rng.normal(0, 1, (n, 6)) + y[:, None] * 1.4
        sc = StandardScaler().fit(emb)
        clf = LogisticRegression(max_iter=500).fit(sc.transform(emb), y)
        p = clf.predict_proba(sc.transform(emb))[:, 1]
        return Screener(task="t", representation="bandpower_concat", scaler=sc, head=clf,
                        calibrator=fit_platt_calibrator(p, y), conformal=0.2,
                        heldout={"auroc": 0.9}, meta={"label": "t"}, personal=personal), emb, y

    def test_v1_roundtrip_has_no_personal(self):
        s, emb, _ = self._synthetic(personal=None)
        with tempfile.TemporaryDirectory() as d:
            s.save(d)
            man = json.loads((Path(d) / "manifest.json").read_text())
            self.assertEqual(man["format"], "dvxr-screener/1")
            self.assertFalse(man["personalized"])
            from dvxr.serve.screener import Screener
            s2 = Screener.load(d)
            self.assertIsNone(s2.personal)
            np.testing.assert_allclose(s.predict_windows(emb), s2.predict_windows(emb))

    def test_v2_roundtrip_and_personal_prediction_differs(self):
        from dvxr.personalization import PersonalizedCalibrator
        rng = np.random.default_rng(1)
        subs = np.repeat(["A", "B"], 40)
        probs = rng.random(80)
        y = (probs + rng.normal(0, 0.1, 80) > 0.5).astype(int)
        pc = PersonalizedCalibrator()
        pc.fit(subs, probs, y)
        s, emb, _ = self._synthetic(personal=pc)
        with tempfile.TemporaryDirectory() as d:
            s.save(d)
            man = json.loads((Path(d) / "manifest.json").read_text())
            self.assertEqual(man["format"], "dvxr-screener/2")
            from dvxr.serve.screener import Screener
            s2 = Screener.load(d)
            self.assertIsNotNone(s2.personal)
            pop = s2.predict_windows(emb)
            per = s2.predict_windows(emb, subject_id="A")
            self.assertFalse(np.allclose(pop, per))              # personalization changes output


if __name__ == "__main__":
    unittest.main()
