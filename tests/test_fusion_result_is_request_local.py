"""PR5: fusion explanations must be request-local — interleaved calls must not clobber each other via
the shared mutable CACMFModel._last state (spec §11 concurrency correction)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _torch():
    try:
        import torch  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(_torch(), "torch not installed")
class FusionResultRequestLocalTest(unittest.TestCase):
    def _model(self):
        from dvxr.config import DEFAULTS
        from dvxr.fusion.model import build_cacmf_model
        cfg = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1, seed=7)
        mods = ["eeg", "wearable_phys", "cgm"]
        return build_cacmf_model(cfg, mods), cfg, mods

    def _latents(self, cfg, mods, scale):
        import torch
        torch.manual_seed(0)
        return {m: torch.randn(4, cfg.d) * scale for m in mods}

    def test_result_is_independent_of_a_later_call(self):
        import torch
        model, cfg, mods = self._model()
        model.eval()
        with torch.no_grad():
            r1 = model.fuse_result(self._latents(cfg, mods, 1.0))
            h1 = r1.h.clone()
            # a second, different fusion must not mutate the first result
            _r2 = model.fuse_result(self._latents(cfg, mods, 5.0))
        self.assertTrue(torch.equal(r1.h, h1), "first FusionResult was mutated by a later call")

    def test_two_results_differ_for_different_inputs(self):
        import torch
        model, cfg, mods = self._model()
        model.eval()
        with torch.no_grad():
            r1 = model.fuse_result(self._latents(cfg, mods, 1.0))
            r2 = model.fuse_result(self._latents(cfg, mods, 5.0))
        self.assertFalse(torch.equal(r1.h, r2.h))

    def test_result_carries_its_own_explanation(self):
        import torch
        model, cfg, mods = self._model()
        model.eval()
        with torch.no_grad():
            r = model.fuse_result(self._latents(cfg, mods, 1.0))
        self.assertIsNotNone(r.h)
        self.assertIsInstance(r.codes, dict)
        self.assertIsInstance(r.vq_loss, float)


if __name__ == "__main__":
    unittest.main()
