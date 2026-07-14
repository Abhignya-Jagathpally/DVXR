"""Tests for the product serving core (dvxr.serve.screener).

Fast path: the Screener's persistence + calibrated inference mechanics on synthetic embeddings
(no data, no LaBraM). Gated integration: fit_screener on a real cohort reproduces the benchmark
held-out AUROC (the product must report the SAME validated number), incl. the LaBraM depression
screener (~0.96) when the weights + cohort are available.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _synthetic_screener(seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from dvxr.calibration import fit_platt_calibrator
    from dvxr.serve.screener import Screener
    rng = np.random.default_rng(seed)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    emb = rng.normal(0, 1, (n, 8)) + y[:, None] * 1.5      # separable
    sc = StandardScaler().fit(emb)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(emb), y)
    p = clf.predict_proba(sc.transform(emb))[:, 1]
    cal = fit_platt_calibrator(p, y)
    return Screener(task="synthetic", representation="bandpower_concat", scaler=sc, head=clf,
                    calibrator=cal, conformal=0.2,
                    heldout={"metric": "AUROC", "auroc": 0.9, "auroc_ci": [0.85, 0.95],
                             "n_subjects": 10, "n_windows": n},
                    meta={"label": "synthetic screen", "encoder": "test",
                          "caveat": "test only"}), emb, y


class ScreenerMechanicsTest(unittest.TestCase):
    def test_predict_and_score(self):
        s, emb, y = _synthetic_screener()
        probs = s.predict_windows(emb)
        self.assertEqual(probs.shape, (len(y),))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        # calibrated probs separate the classes
        self.assertGreater(probs[y == 1].mean(), probs[y == 0].mean() + 0.2)
        res = s.score_subject(emb[y == 1])
        self.assertIn(res["risk_band"], {"low", "watch", "elevated", "high"})
        self.assertEqual(len(res["interval"]), 2)
        self.assertLessEqual(res["interval"][0], res["probability"])
        self.assertGreaterEqual(res["interval"][1], res["probability"])

    def test_save_load_roundtrip(self):
        s, emb, _ = _synthetic_screener()
        before = s.predict_windows(emb)
        with tempfile.TemporaryDirectory() as d:
            p = s.save(Path(d) / "scr")
            self.assertTrue((p / "manifest.json").exists())
            manifest = json.loads((p / "manifest.json").read_text())
            self.assertEqual(manifest["format"], "dvxr-screener/1")
            from dvxr.serve.screener import Screener
            s2 = Screener.load(p)
        after = s2.predict_windows(emb)
        np.testing.assert_allclose(before, after)
        self.assertEqual(s2.heldout["auroc"], s.heldout["auroc"])


def _data_dir(name):
    return Path("data/real") / name


@unittest.skipUnless(_data_dir("WESAD").exists(), "WESAD real data required")
class ScreenerWesadTest(unittest.TestCase):
    def test_wesad_stress_reproduces_strong_auroc(self):
        from dvxr.serve.screener import fit_screener
        s = fit_screener("wesad_stress", n_repeats=2, n_folds=5, seed=7)
        # benchmark records wesad stress AUROC ~0.95 (band-power); allow head/CV variance
        self.assertGreater(s.heldout["auroc"], 0.80)
        with tempfile.TemporaryDirectory() as d:
            s.save(Path(d) / "w")
            from dvxr.serve.screener import Screener
            self.assertEqual(Screener.load(Path(d) / "w").task, "wesad_stress")


def _labram_ready():
    try:
        from dvxr.bench.labram_bench import _weights_reachable
        return _weights_reachable() and _data_dir("mumtaz_mdd").exists()
    except Exception:
        return False


@unittest.skipUnless(_labram_ready(), "LaBraM weights + Mumtaz cohort required")
class ScreenerDepressionTest(unittest.TestCase):
    def test_depression_labram_reproduces_benchmark(self):
        from dvxr.serve.screener import fit_screener
        s = fit_screener("mumtaz_depression", n_repeats=3, n_folds=5, seed=7)
        # benchmark: LaBraM depression AUROC ~0.96; the product must reproduce it
        self.assertGreater(s.heldout["auroc"], 0.85)
        self.assertEqual(s.representation, "labram_eeg")
        res = s.score_subject(np.zeros((4, 200)))   # shape-valid dummy embedding
        self.assertIn(res["risk_band"], {"low", "watch", "elevated", "high"})


if __name__ == "__main__":
    unittest.main()
