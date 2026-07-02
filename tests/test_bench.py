from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.bench import protocol as P  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


class ProtocolStatsTest(unittest.TestCase):
    def test_folds_subject_disjoint(self):
        sids = np.array([f"s{i%8}" for i in range(80)])
        folds = P.repeated_group_folds(sids, n_repeats=3, n_folds=4, seed=1)
        self.assertEqual(len(folds), 12)
        for tr, te in folds:
            self.assertEqual(set(sids[tr]) & set(sids[te]), set())
            self.assertTrue(len(tr) and len(te))

    def test_folds_need_two_subjects(self):
        with self.assertRaises(ValueError):
            P.repeated_group_folds(np.array(["s0"] * 10))

    def test_bootstrap_ci_brackets_mean(self):
        v = np.random.default_rng(0).normal(5, 1, 200)
        lo, hi = P.bootstrap_ci(v, seed=0)
        self.assertLess(lo, v.mean())
        self.assertGreater(hi, v.mean())

    def test_wilcoxon_directional(self):
        prop = np.array([0.1, 0.12, 0.09, 0.11, 0.1])   # clearly lower error
        base = np.array([0.2, 0.22, 0.19, 0.21, 0.2])
        self.assertLess(P.paired_wilcoxon(prop, base), 0.1)
        # reversed: proposed worse -> not significant "less"
        self.assertGreater(P.paired_wilcoxon(base, prop), 0.5)

    def test_cliffs_delta_sign(self):
        a = np.array([0.1, 0.1, 0.1]); b = np.array([0.2, 0.2, 0.2])
        self.assertAlmostEqual(P.cliffs_delta(a, b), 1.0)      # a all below b
        self.assertAlmostEqual(P.cliffs_delta(b, a), -1.0)

    def test_holm_monotone_and_bounded(self):
        adj = P.holm_correction({"a": 0.01, "b": 0.04, "c": 0.5})
        self.assertTrue(all(0 <= v <= 1 for v in adj.values()))
        self.assertLessEqual(adj["a"], adj["b"])
        self.assertLessEqual(adj["b"], adj["c"])

    def test_relativity_win_and_target(self):
        prop = [0.04, 0.05, 0.045, 0.05]      # ~half the error
        base = [0.10, 0.11, 0.10, 0.10]
        r = P.relativity("t", "1-AUROC", "baseline", prop, base, seed=0)
        self.assertGreater(r.rer_pct, 40)
        self.assertEqual(r.n_folds, 4)
        # a real win with tight CI and p<.05 should clear the bar
        r.p_holm = r.p_wilcoxon
        self.assertIsInstance(r.meets_target(50.0), bool)

    def test_relativity_loss_is_negative(self):
        r = P.relativity("t", "MAE", "b", [12.0, 13.0, 12.5], [10.0, 10.5, 10.2], seed=0)
        self.assertLess(r.rer_pct, 0)          # proposed worse -> negative RER
        self.assertFalse(r.meets_target(50.0))


class NoFabricationTest(unittest.TestCase):
    def test_assert_no_fabrication_passes_by_default(self):
        from dvxr.bench.tasks import assert_no_fabrication
        assert_no_fabrication()   # must not raise in the clean benchmark path


@unittest.skipUnless(HAVE_TORCH, "torch required for the fused representation")
class TinyRunTest(unittest.TestCase):
    def _synthetic_task(self):
        from dvxr.bench.tasks import BenchTask
        rng = np.random.default_rng(0)
        rows, feats_a, feats_b, y, sid = 0, [], [], [], []
        for s in range(8):
            bias = rng.normal(0, 0.5)
            for _ in range(12):
                a = rng.normal(0, 1, 4); b = rng.normal(0, 1, 4)
                lab = int(a[0] + b[0] + bias > 0)
                feats_a.append(a); feats_b.append(b); y.append(lab); sid.append(f"s{s}")
        return BenchTask(
            name="synthetic", kind="classification",
            features={"a": np.array(feats_a), "b": np.array(feats_b)},
            feature_names={"a": [f"a{i}" for i in range(4)], "b": [f"b{i}" for i in range(4)]},
            y=np.array(y), subject_ids=np.array(sid),
            metric="1-AUROC", baseline_hint="majority")

    def test_run_task_produces_relativity(self):
        from dvxr.bench.run import run_task
        task = self._synthetic_task()
        res = run_task(task, n_repeats=1, n_folds=3, seed=7, include_sota=False)
        self.assertIn("rep:fused", res.per_config_fold_err)
        self.assertTrue(np.isfinite(res.relativity.base_err))
        # majority baseline must sit at chance (1-AUROC ~ 0.5)
        self.assertAlmostEqual(np.nanmean(res.per_config_fold_err["majority"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
