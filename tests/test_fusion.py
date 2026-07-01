from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS, MODALITIES  # noqa: E402
from dvxr.fusion.aggregate import (  # noqa: E402
    confidence_weighted,
    ensemble_avg,
    weighted_late,
)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

CFG = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1, seed=7)
STRATEGIES = ["early", "intermediate", "late_weighted", "attention", "cross_modal"]


def _latents(present=None, B=5, d=8, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    mods = present or list(MODALITIES)
    return {m: torch.randn(B, d, generator=g) for m in mods}


class AggregateTest(unittest.TestCase):
    def test_ensemble_and_weighted(self):
        probs = {"a": np.array([[0.2, 0.8]]), "b": np.array([[0.6, 0.4]])}
        np.testing.assert_allclose(ensemble_avg(probs), [[0.4, 0.6]])
        np.testing.assert_allclose(
            weighted_late(probs, {"a": 3.0, "b": 1.0}), [[0.3, 0.7]])

    def test_confidence_weighted_defers_to_confident(self):
        # modality A is confident, B is maximally uncertain -> result ~ A
        probs = {"A": np.array([[0.99, 0.01]]), "B": np.array([[0.5, 0.5]])}
        out = confidence_weighted(probs)
        self.assertGreater(out[0, 0], 0.95)


@unittest.skipUnless(HAVE_TORCH, "torch required for fusion strategies")
class FusionStrategyTest(unittest.TestCase):
    def test_each_strategy_joint_shape(self):
        from dvxr.fusion.strategies import get_fusion_strategy
        lat = _latents()
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                fusion = get_fusion_strategy(name, CFG)
                out = fusion(lat)
                self.assertEqual(tuple(out.h.shape), (5, CFG.d_f))
                self.assertEqual(set(out.present), set(MODALITIES))

    def test_missing_modality_still_runs(self):
        from dvxr.fusion.strategies import get_fusion_strategy
        present = [m for m in MODALITIES if m != "cgm"][:3]  # arbitrary subset
        lat = _latents(present=present)
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                out = get_fusion_strategy(name, CFG)(lat)
                self.assertEqual(tuple(out.h.shape), (5, CFG.d_f))
                self.assertEqual(set(out.present), set(present))
                self.assertNotIn("cgm", out.present)

    def test_attention_weights_sum_to_one(self):
        from dvxr.fusion.strategies import get_fusion_strategy
        present = ["eeg", "wearable_phys", "cgm"]
        lat = _latents(present=present)
        for name in ("attention", "cross_modal"):
            out = get_fusion_strategy(name, CFG)(lat)
            self.assertIsNotNone(out.attention)
            total = sum(out.attention[m] for m in out.attention)  # (B,)
            np.testing.assert_allclose(total.detach().numpy(),
                                       np.ones(5), rtol=1e-5, atol=1e-5)

    def test_late_fusion_weights_sum_to_one(self):
        from dvxr.fusion.strategies import get_fusion_strategy
        present = ["eeg", "wearable_phys", "cgm"]
        out = get_fusion_strategy("late_weighted", CFG)(_latents(present=present))
        self.assertIsNotNone(out.weights)
        total = float(sum(float(out.weights[m]) for m in out.weights))
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_determinism(self):
        from dvxr.fusion.strategies import get_fusion_strategy
        lat = _latents()
        h1 = get_fusion_strategy("cross_modal", CFG)(lat).h.detach().numpy()
        h2 = get_fusion_strategy("cross_modal", CFG)(lat).h.detach().numpy()
        np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAVE_TORCH, "torch required for CACMFModel")
class CACMFModelTest(unittest.TestCase):
    def test_fuse_and_exports(self):
        from dvxr.fusion.model import build_cacmf_model
        model = build_cacmf_model(CFG.with_(fusion_strategy="cross_modal"))
        model.eval()
        lat = _latents()
        out = model.fuse(lat)
        self.assertEqual(tuple(out.h.shape), (5, CFG.d_f))
        self.assertIsNotNone(model.attention_weights())
        self.assertIsNotNone(model.vq_loss())
        with tempfile.TemporaryDirectory() as tmp:
            paths = model.export_latents(lat, out_dir=tmp)
            self.assertTrue(Path(paths["h"]).exists())
            self.assertTrue(Path(paths["codes"]).exists())
            self.assertTrue(Path(paths["attention"]).exists())

    def test_late_strategy_exposes_fusion_weights(self):
        from dvxr.fusion.model import build_cacmf_model
        model = build_cacmf_model(CFG.with_(fusion_strategy="late_weighted"))
        model.eval()
        model.fuse(_latents())
        self.assertIsNotNone(model.fusion_weights())


if __name__ == "__main__":
    unittest.main()
