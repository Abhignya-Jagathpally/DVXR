from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

from dvxr.encoders.codebook import (  # noqa: E402
    VQBiosignalEncoder,
    get_vector_quantizer_class,
)


def _fixture(n=64, f=8, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(f)]
    # three latent clusters so multiple codes get used
    centers = rng.normal(size=(3, f)) * 3.0
    rows = np.vstack([centers[i % 3] + rng.normal(scale=0.3, size=f) for i in range(n)])
    return pd.DataFrame(rows, columns=cols), cols


@unittest.skipUnless(HAVE_TORCH, "torch is required for the VQ codebook")
class VectorQuantizerTest(unittest.TestCase):
    def test_shapes_and_perplexity(self):
        VQ = get_vector_quantizer_class()
        K, d = 32, 6
        q = VQ(num_codes=K, dim=d)
        z = torch.randn(20, d)
        out = q(z, training=False)
        self.assertEqual(tuple(out.quantized.shape), (20, d))
        self.assertEqual(tuple(out.indices.shape), (20,))
        ppl = float(out.perplexity)
        self.assertTrue(1.0 <= ppl <= K, f"perplexity {ppl} outside (1, {K}]")
        self.assertGreater(ppl, 1.0)  # varied input -> more than one code

    def test_straight_through_gradient(self):
        VQ = get_vector_quantizer_class()
        q = VQ(num_codes=16, dim=4)
        z = torch.randn(10, 4, requires_grad=True)
        out = q(z, training=False)
        out.quantized.sum().backward()
        self.assertIsNotNone(z.grad)
        # straight-through: d(quantized)/dz == 1 everywhere
        self.assertTrue(torch.allclose(z.grad, torch.ones_like(z.grad)))

    def test_ema_moves_codebook(self):
        VQ = get_vector_quantizer_class()
        q = VQ(num_codes=16, dim=4)
        before = q.codebook.clone()
        z = torch.randn(50, 4)
        q(z, training=True)
        self.assertFalse(torch.allclose(before, q.codebook),
                         "EMA update should move the codebook in training mode")


@unittest.skipUnless(HAVE_TORCH, "torch is required for the VQ codebook")
class VQBiosignalEncoderTest(unittest.TestCase):
    def test_fit_transform_shape(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=5, codebook_size=32, seed=7)
        emb = enc.fit_transform(frame, cols)
        self.assertEqual(emb.shape, (len(frame), 8))
        self.assertTrue(list(emb.columns) == [f"embed_{i:02d}" for i in range(8)])

    def test_loss_decreases(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=20, codebook_size=32, seed=7)
        enc.fit_transform(frame, cols)
        hist = enc._loss_history
        self.assertGreater(len(hist), 5)
        self.assertLess(np.mean(hist[-3:]), np.mean(hist[:3]),
                        "training loss should decrease over epochs")

    def test_perplexity_in_range(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=8, codebook_size=64, seed=7)
        enc.fit_transform(frame, cols)
        ppl = enc.perplexity(frame)
        self.assertTrue(1.0 <= ppl <= 64, f"perplexity {ppl} outside (1, 64]")
        self.assertGreater(ppl, 1.0)

    def test_determinism(self):
        frame, cols = _fixture()
        e1 = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                n_heads=2, epochs=6, codebook_size=32, seed=7)
        e2 = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                n_heads=2, epochs=6, codebook_size=32, seed=7)
        a = e1.fit_transform(frame, cols).to_numpy()
        b = e2.fit_transform(frame, cols).to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    def test_quantize_and_usage_and_export(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=6, codebook_size=32, seed=7)
        enc.fit_transform(frame, cols)
        idx, quant = enc.quantize(frame)
        self.assertEqual(len(idx), len(frame))
        self.assertEqual(quant.shape, (len(frame), 8))
        usage = enc.codebook_usage(frame)
        self.assertEqual(int(usage["count"].sum()), len(frame))
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, npy_path = enc.export(frame, tmp)
            self.assertTrue(Path(csv_path).exists())
            self.assertTrue(Path(npy_path).exists())

    def test_save_and_from_pretrained(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=6, codebook_size=32, seed=7)
        emb1 = enc.fit_transform(frame, cols)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "vq.pt"
            enc.save(p)
            enc2 = VQBiosignalEncoder.from_pretrained(p)
        emb2 = enc2.transform(frame)
        np.testing.assert_allclose(emb1.to_numpy(), emb2.to_numpy(), rtol=1e-5, atol=1e-6)

    def test_gradient_saliency_inherited(self):
        frame, cols = _fixture()
        enc = VQBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1,
                                 n_heads=2, epochs=4, codebook_size=32, seed=7)
        enc.fit_transform(frame, cols)
        sal = enc.gradient_saliency(frame, cols)
        self.assertEqual(sal.shape, (len(frame), len(cols)))
        self.assertTrue((sal.to_numpy() >= 0).all())


if __name__ == "__main__":
    unittest.main()
