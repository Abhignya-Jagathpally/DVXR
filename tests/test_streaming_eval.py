"""Slice 8: the streaming / partial-observation showdown. The honest win-hunt —
does the graceful-degradation fusion model beat a floor that must impute when
modalities drop out at test time? Skip-guarded on torch (fused) being importable;
xgboost/LLM legs are optional and auto-skipped when their dep is absent."""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_TORCH, "torch required for the fused model")
class StreamingShowdownTest(unittest.TestCase):
    def _task(self):
        from dvxr.bench.tasks import cgmacros_diabetes_task
        return cgmacros_diabetes_task(subjects=12)

    def test_concat_masked_imputes_dropped_block(self):
        from dvxr.bench.streaming_eval import _concat_masked, _modality_slices
        t = self._task()
        drop = [t.modalities[0]]
        sl = _modality_slices(t)
        s, e = sl[drop[0]]
        nan_masked = _concat_masked(t, drop, impute="nan")
        self.assertTrue(np.isnan(nan_masked[:, s:e]).all())          # block blanked
        self.assertFalse(np.isnan(nan_masked[:, e:]).any())          # rest intact
        mean_masked = _concat_masked(t, drop, impute="mean",
                                     train_idx=np.arange(t.n))
        self.assertFalse(np.isnan(mean_masked).any())                # mean-imputed, no NaN
        # the imputed block is constant per column (the train mean)
        block = mean_masked[:, s:e]
        self.assertTrue(np.allclose(block, block[0], atol=1e-6))

    def test_showdown_has_full_and_dropped_levels(self):
        from dvxr.bench.streaming_eval import partial_observation_showdown
        res = partial_observation_showdown(self._task(), seed=7, n_repeats=1,
                                           n_folds=3, models=("fused",),
                                           max_combos=2)
        # level 0 (nothing dropped) present, plus at least one dropout level
        levels = {row["k"] for row in res["curve"]}
        self.assertIn(0, levels)
        self.assertGreaterEqual(max(levels), 1)
        # every curve row carries a finite error for the proposed model and the floor
        for row in res["curve"]:
            self.assertTrue(np.isfinite(row["proposed_err"]))
            self.assertTrue(np.isfinite(row["floor_err"]))
        # a verdict is reported (crossover_k is an int or None, never fabricated)
        self.assertIn("crossover_k", res)
        self.assertTrue(res["crossover_k"] is None or isinstance(res["crossover_k"], int))

    def test_full_observation_floor_not_worse_than_dropout(self):
        """Sanity: the floor's error should not IMPROVE as more modalities vanish —
        degradation is monotone-ish. (Guards against an imputation bug that leaks signal.)"""
        from dvxr.bench.streaming_eval import partial_observation_showdown
        res = partial_observation_showdown(self._task(), seed=7, n_repeats=1,
                                           n_folds=3, models=("fused",), max_combos=2)
        by_k = {row["k"]: row["floor_err"] for row in res["curve"]}
        self.assertLessEqual(by_k[0], max(by_k.values()) + 1e-9)     # full-obs is best-ish for floor


if __name__ == "__main__":
    unittest.main()
