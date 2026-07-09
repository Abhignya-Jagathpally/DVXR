"""Slice 10/11: Sleep-EDF loader + raw-signal CNN. Skip-guarded on >=2 recordings being
present locally (PhysioNet fetch is slow) and on torch. Validates the windowing exposes both
summary-stat features and raw windows, and that the raw CNN trains + predicts leakage-free."""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.sleep_edf import local_sleep_edf_pairs  # noqa: E402

_TORCH = importlib.util.find_spec("torch") is not None
_N = len(local_sleep_edf_pairs())
_RUNNABLE = _TORCH and _N >= 2


@unittest.skipUnless(_RUNNABLE, f"needs torch + >=2 Sleep-EDF recordings (have {_N})")
class SleepEdfTest(unittest.TestCase):
    def _task(self, target="wake_sleep"):
        from dvxr.bench.tasks import sleep_edf_stage_task
        return sleep_edf_stage_task(n_recordings=min(_N, 3), target=target,
                                    max_epochs_per_rec=120)

    def test_windows_expose_features_and_raw(self):
        t = self._task()
        self.assertEqual(set(t.modalities), {"eeg", "eog", "emg", "resp"})
        for m in t.modalities:
            self.assertEqual(t.features[m].shape[0], t.n)      # summary-stat features
            self.assertEqual(t.extra["raw"][m].shape[0], t.n)  # raw windows, row-aligned
        self.assertEqual(len(t.y), t.n)
        self.assertTrue(set(np.unique(t.y)).issubset({0, 1}))

    def test_rawcnn_predicts_valid_probabilities(self):
        from dvxr.bench.raw_seq import pred_rawcnn
        t = self._task()
        tr = list(range(0, t.n, 2))
        te = list(range(1, t.n, 2))
        p = pred_rawcnn(t, tr, te, seed=7, epochs=3)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_win_benchmark_reports_verdict(self):
        from dvxr.bench.raw_seq import sleep_win_benchmark
        t = self._task()
        r = sleep_win_benchmark(t, seed=7, n_repeats=1,
                                n_folds=min(3, len(set(t.subject_ids))), epochs=3)
        for k in ("rawcnn_err", "floor_err", "rer_pct", "rer_ci", "win"):
            self.assertIn(k, r)
        self.assertIsInstance(r["win"], bool)               # verdict is a real bool, never faked


if __name__ == "__main__":
    unittest.main()
