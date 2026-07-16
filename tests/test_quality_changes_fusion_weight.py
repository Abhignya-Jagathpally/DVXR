"""PR5: quality-aware gated fusion down-weights a stale/low-quality/OOD modality even when it is
confident (spec §5, §9, §11)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.fusion.aggregate import quality_gated  # noqa: E402


def _confident_probs(p1):
    # two modalities, both confident; cgm votes p1 for the positive class, eeg votes the opposite
    B = 4
    cgm = np.tile([1 - p1, p1], (B, 1))
    eeg = np.tile([p1, 1 - p1], (B, 1))
    return {"cgm": cgm, "eeg": eeg}


class QualityGatedFusionTest(unittest.TestCase):
    def test_equal_quality_is_balanced(self):
        probs = _confident_probs(0.9)
        fused, w = quality_gated(probs, return_weights=True)
        self.assertAlmostEqual(w["cgm"], w["eeg"], places=6)
        self.assertAlmostEqual(fused[0, 1], 0.5, places=6)     # symmetric votes cancel

    def test_low_quality_modality_is_down_weighted(self):
        probs = _confident_probs(0.9)
        fused, w = quality_gated(probs, quality={"eeg": 0.1, "cgm": 1.0}, return_weights=True)
        self.assertGreater(w["cgm"], w["eeg"])                 # cgm dominates
        self.assertGreater(fused[0, 1], 0.5)                   # fused shifts toward cgm's vote

    def test_stale_modality_is_down_weighted(self):
        _f, w = quality_gated(_confident_probs(0.9), freshness={"eeg": 0.05}, return_weights=True)
        self.assertLess(w["eeg"], w["cgm"])

    def test_ood_modality_is_down_weighted(self):
        _f, w = quality_gated(_confident_probs(0.9), ood={"eeg": 0.95}, return_weights=True)
        self.assertLess(w["eeg"], w["cgm"])

    def test_unavailable_modality_gets_zero_weight(self):
        _f, w = quality_gated(_confident_probs(0.9), availability={"eeg": 0.0}, return_weights=True)
        self.assertAlmostEqual(w["eeg"], 0.0, places=6)

    def test_all_unreliable_falls_back_to_mean_not_divide_by_zero(self):
        fused = quality_gated(_confident_probs(0.9), availability={"eeg": 0.0, "cgm": 0.0})
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertAlmostEqual(fused[0, 1], 0.5, places=6)     # mean fallback of symmetric votes


if __name__ == "__main__":
    unittest.main()
