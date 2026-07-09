"""Slice 7: the soft-prompt LLM predictor (rep:llm). Skip-guarded on a local LLM being
available (transformers + a cached model); offline CI without the model skips cleanly."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.bench.representations import llm_available  # noqa: E402

CGM = Path(__file__).resolve().parents[1] / "data" / "real" / "cgmacros"
_RUNNABLE = llm_available() and bool(list(CGM.glob("**/CGMacros-*.csv")))


@unittest.skipUnless(_RUNNABLE, "local LLM or CGMacros data absent")
class LlmPredictorTest(unittest.TestCase):
    def _task(self):
        from dvxr.bench.tasks import cgmacros_diabetes_task
        return cgmacros_diabetes_task(subjects=8)

    def test_embeddings_nonconstant_and_predictive_shape(self):
        from dvxr.llm.predictor import llm_window_embeddings
        t = self._task()
        emb = llm_window_embeddings(t, seed=7)
        self.assertEqual(emb.shape[0], t.n)
        self.assertGreater((emb.std(0) > 1e-6).sum(), 0)  # soft tokens influence hidden state

    def test_missing_modality_still_predicts(self):
        from dvxr.llm.predictor import llm_window_embeddings
        t = self._task()
        full = llm_window_embeddings(t, seed=7)
        dropped = llm_window_embeddings(t, seed=7, drop=[t.modalities[0]])
        self.assertEqual(full.shape, dropped.shape)      # interoperability: still produces output
        self.assertFalse(np.allclose(full, dropped))     # and the missing modality changes it

    def test_modality_attribution_sums_to_one(self):
        from dvxr.llm.predictor import modality_attribution
        attr = modality_attribution(self._task(), seed=7)
        self.assertEqual(set(attr), set(self._task().modalities))
        self.assertAlmostEqual(sum(attr.values()), 1.0, places=5)

    def test_rep_llm_runs_through_head(self):
        from dvxr.bench.representations import evaluate_representation
        t = self._task()
        tr = list(range(0, t.n, 2))
        te = list(range(1, t.n, 2))
        pred = evaluate_representation(t, "llm", tr, te, seed=7)
        self.assertEqual(len(pred), len(te))
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))  # calibrated probability


if __name__ == "__main__":
    unittest.main()
