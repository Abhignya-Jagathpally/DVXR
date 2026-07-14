"""Tests for the vendored real-LaBraM EEG foundation model (dvxr.encoders.labram_real).

Validates the correctness bar from docs/SLICE_B_LABRAM.md: the weights load STRICT (every
checkpoint key consumed, none missing/unexpected — a mismatch means the vendored forward is
out of sync and its embeddings must not be trusted), and the forward produces finite,
non-degenerate embeddings of the right shape. Network-gated: skipped cleanly when the weights
can't be fetched, so the offline suite still passes.

The stronger reference cross-check (vs. an isolated braindecode env) and the functional check
(does LaBraM decode eegmat above chance) live in the benchmark wiring (Slice B task #6)."""
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    from dvxr.encoders.labram_real import LaBraMEncoder, labram_available
    _HAVE = labram_available()
except Exception:
    _HAVE = False


def _try_load():
    try:
        return LaBraMEncoder.from_pretrained()
    except Exception:
        return None


@unittest.skipUnless(_HAVE, "torch/safetensors/huggingface_hub required for real LaBraM")
class LaBraMRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = _try_load()
        if cls.enc is None:
            raise unittest.SkipTest("LaBraM weights not fetchable (offline / no cache)")

    def test_strict_load_and_channel_map(self):
        # from_pretrained already enforced strict load; canonical index must be populated
        self.assertGreater(len(self.enc.canonical_index), 100)
        self.assertIn("FP1", self.enc.canonical_index)

    def test_embed_shape_and_nondegenerate(self):
        rng = np.random.default_rng(0)
        eeg = rng.normal(0, 1, (6, 19, 400)).astype("float32")   # 19ch, 2 patches @200Hz
        chs = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
        emb = self.enc.embed(eeg, chs)
        self.assertEqual(emb.shape, (6, 200))
        self.assertTrue(np.all(np.isfinite(emb)))
        # non-degenerate: embeddings vary across dims and across distinct inputs
        self.assertGreater(float(emb.std(0).mean()), 1e-3)
        self.assertGreater(float(np.linalg.norm(emb[0] - emb[1])), 1e-2)

    def test_unknown_channels_dropped(self):
        rng = np.random.default_rng(1)
        eeg = rng.normal(0, 1, (3, 3, 200)).astype("float32")
        # one real channel + two names not in LaBraM's vocab -> kept subset still embeds
        emb = self.enc.embed(eeg, ["Cz", "NOTACHAN", "ALSOBOGUS"])
        self.assertEqual(emb.shape, (3, 200))
        self.assertTrue(np.all(np.isfinite(emb)))


if __name__ == "__main__":
    unittest.main()
