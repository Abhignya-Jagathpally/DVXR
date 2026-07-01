from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS, MODALITIES  # noqa: E402
from dvxr.explain.codebook_usage import (  # noqa: E402
    codebook_histogram,
    codebook_perplexity,
    top_codes_per_label,
)
from dvxr.schemas import REQUIRED_EVENT_COLUMNS  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

CFG = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1, codebook_size=16, seed=7)
BASE = pd.Timestamp("2026-01-01T00:00:00Z")


def _events():
    rows = []
    for sec in range(0, 120, 5):
        for mod, chan, val in [("eda", "eda", 2.0), ("cgm", "glucose", 120.0),
                               ("eeg", "eeg", 0.3)]:
            rows.append({
                "subject_id": "s1", "session_id": "sess1",
                "timestamp_utc": BASE + pd.Timedelta(seconds=sec),
                "source_system": "t", "device": "t", "modality": mod,
                "channel": chan, "value": val, "unit": "u",
                "sampling_rate_hz": 1.0, "quality_flag": "ok",
                "label_name": "", "label_value": ""})
    return pd.DataFrame(rows)[REQUIRED_EVENT_COLUMNS]


def _latents(B=6, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(B, CFG.d, generator=g) for m in MODALITIES}


class CodebookUsageTest(unittest.TestCase):
    def test_histogram_sums_to_sample_count(self):
        idx = {"eeg": np.array([0, 1, 1, 2, 2, 2]),
               "cgm": np.array([3, 3, 3, 3, 3, 3])}
        hist = codebook_histogram(idx)
        for m, n in [("eeg", 6), ("cgm", 6)]:
            self.assertEqual(int(hist[hist.modality == m]["count"].sum()), n)

    def test_perplexity_range(self):
        pp = codebook_perplexity({"eeg": np.array([0, 1, 2, 3, 4, 5])})
        self.assertGreater(pp["eeg"], 1.0)

    def test_top_codes_per_label_lift(self):
        codes = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # code 0 concentrated in positives
        top = top_codes_per_label(codes, labels, top_n=3)
        self.assertEqual(int(top.iloc[0]["code_index"]), 0)
        self.assertGreater(top.iloc[0]["lift"], 1.0)


@unittest.skipUnless(HAVE_TORCH, "torch required for attention/codes")
class ExplainBundleTest(unittest.TestCase):
    def test_attention_normalized(self):
        from dvxr.explain.attention_maps import attention_table
        from dvxr.fusion.strategies import get_fusion_strategy
        fo = get_fusion_strategy("attention", CFG)(_latents())
        tbl = attention_table(fo)
        per_sample = tbl.groupby("sample")["attention"].sum()
        np.testing.assert_allclose(per_sample.to_numpy(), np.ones(6), rtol=1e-5, atol=1e-5)

    def test_late_weights_normalized(self):
        from dvxr.explain.attention_maps import attention_table
        from dvxr.fusion.strategies import get_fusion_strategy
        fo = get_fusion_strategy("late_weighted", CFG)(_latents())
        tbl = attention_table(fo)
        # one weight per modality; unique weights sum to 1
        w = tbl.drop_duplicates("modality")["weight"].to_numpy()
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_explain_prediction_returns_four_blocks(self):
        from dvxr.explain.report import explain_prediction
        from dvxr.fusion.model import build_cacmf_model
        model = build_cacmf_model(CFG.with_(fusion_strategy="cross_modal"))
        model.eval()
        feat = pd.DataFrame(np.random.RandomState(0).randn(6, 5),
                            columns=[f"f{i}" for i in range(5)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "explanation.md"
            blocks = explain_prediction(
                events=_events(), cacmf_model=model, latents=_latents(),
                feature_frame=feat, feature_columns=list(feat.columns),
                out_path=str(out))
            self.assertTrue(out.exists())
        for key in ("physiological_biomarkers", "neural_saliency",
                    "modality_attention", "active_codes"):
            self.assertIsNotNone(blocks[key], f"{key} missing")

    def test_determinism(self):
        from dvxr.explain.attention_maps import attention_table
        from dvxr.fusion.strategies import get_fusion_strategy
        a = attention_table(get_fusion_strategy("attention", CFG)(_latents(seed=3)))
        b = attention_table(get_fusion_strategy("attention", CFG)(_latents(seed=3)))
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
