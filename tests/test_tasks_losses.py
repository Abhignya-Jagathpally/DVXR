from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS  # noqa: E402
from dvxr.tasks.heads import (  # noqa: E402
    CLASSIFICATION_TASKS,
    calibrate_probabilities,
    forecast_interval_coverage,
)
from dvxr.tasks.train import population_and_personalized_metrics  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

CFG = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1,
                     codebook_size=32, epochs=25, seed=7)
INPUT_DIMS = {"eeg": 10, "wearable_phys": 8, "cgm": 6}


def _batch(B=32, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    feats = {m: torch.randn(B, f, generator=g) for m, f in INPUT_DIMS.items()}
    # two labeled classification tasks + a forecast target, all learnable from feats
    y0 = (feats["eeg"][:, 0] > 0).long()
    y1 = (feats["wearable_phys"][:, 0] > 0).long()
    labels = {CLASSIFICATION_TASKS[0]: y0, CLASSIFICATION_TASKS[1]: y1}
    target = feats["cgm"][:, 0] * 2.0 + 1.0
    return feats, labels, target


class NumpyTaskHelpersTest(unittest.TestCase):
    def test_calibrated_probs_in_unit_interval(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 200)
        truth = (raw + rng.normal(0, 0.1, 200) > 0.5).astype(int)
        cal = calibrate_probabilities(raw, truth)
        self.assertTrue(((cal >= 0) & (cal <= 1)).all())

    def test_forecast_interval_coverage_computed(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(0, 1, 100)
        pred = truth + rng.normal(0, 0.3, 100)
        radius, cov, lo, hi = forecast_interval_coverage(
            pred[50:], truth[50:], pred[:50], truth[:50], alpha=0.10)
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)
        self.assertTrue(np.isfinite(radius))

    def test_population_and_personalized_metrics(self):
        rng = np.random.default_rng(2)
        n = 120
        sids = np.array([f"s{i % 6}" for i in range(n)])
        probs = rng.uniform(0, 1, n)
        truth = (probs > 0.5).astype(int)
        m = population_and_personalized_metrics(sids, probs, truth)
        self.assertIn("population_ece", m)
        self.assertIn("personalized_ece", m)
        self.assertTrue(((m["personalized_probs"] >= 0) &
                         (m["personalized_probs"] <= 1)).all())


@unittest.skipUnless(HAVE_TORCH, "torch required for multi-task training")
class MultiTaskTrainingTest(unittest.TestCase):
    def _train(self, uncertainty=False):
        from dvxr.tasks.model import build_multitask_model
        from dvxr.tasks.train import train_multitask
        feats, labels, target = _batch()
        model = build_multitask_model(CFG, INPUT_DIMS)
        with tempfile.TemporaryDirectory() as tmp:
            res = train_multitask(
                model, feats, labels, forecast_target=target, config=CFG,
                log_path=str(Path(tmp) / "train_log.csv"),
                uncertainty_weighting=uncertainty)
            self.assertTrue(Path(res["log_path"]).exists())
        return res

    def test_losses_finite_and_decrease(self):
        res = self._train()
        hist = res["history"]
        for row in hist:
            for k, v in row.items():
                self.assertTrue(np.isfinite(v), f"{k} not finite")
        self.assertLess(np.mean([h["total"] for h in hist[-3:]]),
                        np.mean([h["total"] for h in hist[:3]]))
        # component terms are all present and finite
        for term in ("vq", "recon", "align"):
            self.assertIn(term, hist[0])

    def test_uncertainty_weighting_runs_and_reports_sigma(self):
        res = self._train(uncertainty=True)
        self.assertIsNotNone(res["sigmas"])
        self.assertTrue(all(np.isfinite(v) for v in res["sigmas"].values()))

    def test_determinism(self):
        r1 = self._train()
        r2 = self._train()
        self.assertAlmostEqual(r1["history"][-1]["total"],
                               r2["history"][-1]["total"], places=5)

    def test_calibrated_probabilities_from_model(self):
        from dvxr.tasks.model import build_multitask_model
        feats, labels, _ = _batch()
        model = build_multitask_model(CFG, INPUT_DIMS)
        model.eval()
        probs = model.probabilities(feats)
        for t, p in probs.items():
            arr = p.detach().numpy()
            self.assertTrue(((arr >= 0) & (arr <= 1)).all())
            np.testing.assert_allclose(arr.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
