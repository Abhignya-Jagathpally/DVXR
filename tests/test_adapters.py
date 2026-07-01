from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS  # noqa: E402
from dvxr.encoders import ADAPTERS, ModalityEncoderRegistry  # noqa: E402
from dvxr.encoders.base import make_primary_backend  # noqa: E402
from dvxr.encoders.cgm_adapter import CGMAdapter  # noqa: E402
from dvxr.encoders.eeg_adapter import EEGAdapter  # noqa: E402

# small, fast, fully-offline config
CFG = DEFAULTS.with_(d=8, use_real_weights=False, epochs=5, codebook_size=32)

FALLBACK_NAMES = {"vq_biosignal", "pca_feature", "cgm_summary",
                  "ehr_code_timeline", "omics_linear"}


def _fixture(n=24, f=10, seed=1):
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(f)]
    centers = rng.normal(size=(3, f)) * 3.0
    rows = np.vstack([centers[i % 3] + rng.normal(scale=0.4, size=f) for i in range(n)])
    return pd.DataFrame(rows, columns=cols), cols


class AdapterFallbackTest(unittest.TestCase):
    def test_all_adapters_emit_d_wide_latents(self):
        frame, cols = _fixture()
        for modality, Adapter in ADAPTERS.items():
            with self.subTest(modality=modality):
                enc = Adapter(CFG)
                out = enc.fit_transform(frame, cols)
                self.assertEqual(out.shape, (len(frame), CFG.d))
                self.assertEqual(list(out.columns),
                                 [f"z_{i:02d}" for i in range(CFG.d)])
                # a real fallback ran and was reported
                self.assertNotEqual(enc.used_encoder, "unset")
                self.assertIn(enc.used_encoder, FALLBACK_NAMES)

    def test_transform_matches_after_fit(self):
        frame, cols = _fixture()
        enc = CGMAdapter(CFG)
        a = enc.fit_transform(frame, cols)
        b = enc.transform(frame)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5, atol=1e-6)

    def test_determinism(self):
        frame, cols = _fixture()
        e1 = EEGAdapter(CFG).fit_transform(frame, cols).to_numpy()
        e2 = EEGAdapter(CFG).fit_transform(frame, cols).to_numpy()
        np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-6)

    def test_registry_selects_right_adapter(self):
        reg = ModalityEncoderRegistry(CFG)
        self.assertIsInstance(reg.get("cgm"), CGMAdapter)
        self.assertIsInstance(reg.get("eeg"), EEGAdapter)
        self.assertIs(reg.get("cgm"), reg.get("cgm"))  # cached
        with self.assertRaises(KeyError):
            reg.get("no_such_modality")

    def test_capability_check_falls_back_offline(self):
        """use_real_weights=True but LaBraM/braindecode absent -> offline fallback,
        no network access, and the fallback is reported."""
        frame, cols = _fixture()
        cfg = CFG.with_(use_real_weights=True)
        self.assertIsNone(make_primary_backend("eeg", cfg))  # braindecode loader -> None
        enc = EEGAdapter(cfg)
        out = enc.fit_transform(frame, cols)
        self.assertEqual(out.shape, (len(frame), cfg.d))
        self.assertIn(enc.used_encoder, FALLBACK_NAMES)

    def test_save_and_from_pretrained_pickle_backend(self):
        frame, cols = _fixture()
        enc = CGMAdapter(CFG)
        emb1 = enc.fit_transform(frame, cols)
        with tempfile.TemporaryDirectory() as tmp:
            enc.save(tmp)
            enc2 = CGMAdapter.from_pretrained(tmp)
        emb2 = enc2.transform(frame)
        np.testing.assert_allclose(emb1.to_numpy(), emb2.to_numpy(),
                                   rtol=1e-5, atol=1e-6)

    def test_save_and_from_pretrained_vq_backend(self):
        frame, cols = _fixture()
        enc = EEGAdapter(CFG)
        emb1 = enc.fit_transform(frame, cols)
        with tempfile.TemporaryDirectory() as tmp:
            enc.save(tmp)
            enc2 = EEGAdapter.from_pretrained(tmp)
        emb2 = enc2.transform(frame)
        np.testing.assert_allclose(emb1.to_numpy(), emb2.to_numpy(),
                                   rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
