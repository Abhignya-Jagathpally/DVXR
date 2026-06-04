from __future__ import annotations

"""Tests for NeuralBiosignalEncoder (goal1_pipeline.neural_encoders)."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from goal1_pipeline.neural_encoders import NeuralBiosignalEncoder


def _make_frame(n_rows: int = 40, n_features: int = 12, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Build a reproducible test DataFrame."""
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i:02d}" for i in range(n_features)]
    data = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    frame = pd.DataFrame(data, columns=cols)
    return frame, cols


class TestOutputShape(unittest.TestCase):
    """fit_transform returns correct shape and column names."""

    def test_shape_and_columns(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=3, seed=7)
        result = enc.fit_transform(frame, cols)
        self.assertEqual(result.shape, (40, 8))
        expected_cols = [f"embed_{i:02d}" for i in range(8)]
        self.assertEqual(list(result.columns), expected_cols)

    def test_index_preserved(self):
        frame, cols = _make_frame()
        frame.index = pd.RangeIndex(start=100, stop=140)
        enc = NeuralBiosignalEncoder(embedding_dim=4, hidden_dim=8, n_layers=1, n_heads=2, epochs=2, seed=7)
        result = enc.fit_transform(frame, cols)
        pd.testing.assert_index_equal(result.index, frame.index)


class TestDeterminism(unittest.TestCase):
    """Two encoders with the same seed produce identical output."""

    def test_same_seed_identical_output(self):
        frame, cols = _make_frame()
        enc1 = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=7)
        enc2 = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=7)
        r1 = enc1.fit_transform(frame, cols)
        r2 = enc2.fit_transform(frame, cols)
        np.testing.assert_allclose(r1.values, r2.values, rtol=1e-5, atol=1e-6,
                                   err_msg="Identical seeds must produce identical embeddings")

    def test_different_seed_different_output(self):
        frame, cols = _make_frame()
        enc1 = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=7)
        enc2 = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=99)
        r1 = enc1.fit_transform(frame, cols)
        r2 = enc2.fit_transform(frame, cols)
        self.assertFalse(np.allclose(r1.values, r2.values),
                         "Different seeds should (very likely) produce different embeddings")


class TestTransform(unittest.TestCase):
    """transform reuses fitted params without retraining."""

    def test_transform_same_as_fit_transform(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=7)
        emb_fit = enc.fit_transform(frame, cols)
        emb_transform = enc.transform(frame)
        np.testing.assert_allclose(emb_fit.values, emb_transform.values, rtol=1e-5, atol=1e-6,
                                   err_msg="transform must reproduce fit_transform output on training data")

    def test_transform_before_fit_raises(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder()
        with self.assertRaises(RuntimeError):
            enc.transform(frame)

    def test_transform_shape(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder(embedding_dim=6, hidden_dim=12, n_layers=1, n_heads=2, epochs=3, seed=7)
        enc.fit_transform(frame, cols)
        # Encode a different (smaller) split of the same frame
        result = enc.transform(frame.iloc[:10])
        self.assertEqual(result.shape, (10, 6))
        pd.testing.assert_index_equal(result.index, frame.iloc[:10].index)


class TestSaveLoad(unittest.TestCase):
    """save + from_pretrained roundtrip gives identical transform output."""

    def test_roundtrip(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=5, seed=7)
        enc.fit_transform(frame, cols)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pt"
            enc.save(path)
            loaded = NeuralBiosignalEncoder.from_pretrained(path)

        result_orig = enc.transform(frame)
        result_loaded = loaded.transform(frame)
        np.testing.assert_allclose(result_orig.values, result_loaded.values, rtol=1e-5, atol=1e-6,
                                   err_msg="Loaded encoder must produce identical embeddings")

    def test_save_before_fit_raises(self):
        enc = NeuralBiosignalEncoder()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                enc.save(Path(tmp) / "never.pt")


class TestGradientSaliency(unittest.TestCase):
    """gradient_saliency returns correct shape and is non-negative."""

    def test_shape(self):
        frame, cols = _make_frame(n_rows=40, n_features=12)
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=3, seed=7)
        enc.fit_transform(frame, cols)
        sal = enc.gradient_saliency(frame, cols)
        self.assertEqual(sal.shape, (40, 12))

    def test_columns_match(self):
        frame, cols = _make_frame(n_rows=40, n_features=12)
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=3, seed=7)
        enc.fit_transform(frame, cols)
        sal = enc.gradient_saliency(frame, cols)
        self.assertEqual(list(sal.columns), cols)

    def test_non_negative(self):
        frame, cols = _make_frame(n_rows=40, n_features=12)
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=3, seed=7)
        enc.fit_transform(frame, cols)
        sal = enc.gradient_saliency(frame, cols)
        self.assertTrue((sal.values >= 0).all(), "Saliency values must be non-negative (abs gradient)")

    def test_saliency_before_fit_raises(self):
        frame, cols = _make_frame()
        enc = NeuralBiosignalEncoder()
        with self.assertRaises(RuntimeError):
            enc.gradient_saliency(frame, cols)

    def test_index_preserved(self):
        frame, cols = _make_frame()
        frame.index = pd.RangeIndex(start=50, stop=90)
        enc = NeuralBiosignalEncoder(embedding_dim=8, hidden_dim=16, n_layers=1, n_heads=2, epochs=3, seed=7)
        enc.fit_transform(frame, cols)
        sal = enc.gradient_saliency(frame)
        pd.testing.assert_index_equal(sal.index, frame.index)


if __name__ == "__main__":
    unittest.main()
