from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS, AGGREGATIONS, FUSION_STRATEGIES  # noqa: E402
from dvxr.eval.ablation import (  # noqa: E402
    ablation_summary,
    make_synthetic_dataset,
    run_ablation,
)
from dvxr.eval.splits import subject_holdout_split  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

CFG = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1, codebook_size=16, seed=7)


class SplitTest(unittest.TestCase):
    def test_subjects_disjoint(self):
        sids = np.array([f"s{i%6}" for i in range(60)])
        tr, te = subject_holdout_split(sids, test_frac=0.34, seed=1)
        self.assertEqual(set(sids[tr]) & set(sids[te]), set())
        self.assertGreater(len(te), 0)


@unittest.skipUnless(HAVE_TORCH, "torch required for the ablation harness")
class AblationTest(unittest.TestCase):
    def setUp(self):
        self.ds = make_synthetic_dataset(n_subjects=14, per_subject=10, seed=0)
        self.df = run_ablation(self.ds, config=CFG, test_frac=0.3, seed=7)

    def test_one_row_per_task_config(self):
        n_mods = len(self.ds["features"])
        cls_rows = self.df[self.df.task == "stress_detection"]
        # singles + 5 fusion strategies + 3 aggregators
        self.assertEqual(len(cls_rows),
                         n_mods + len(FUSION_STRATEGIES) + len(AGGREGATIONS))
        fc_rows = self.df[self.df.task == "glucose"]
        self.assertEqual(len(fc_rows), n_mods + len(FUSION_STRATEGIES) + 1)
        self.assertEqual(set(self.df["config_type"]),
                         {"single", "fusion", "aggregation"})

    def test_metrics_finite_where_defined(self):
        cls = self.df[self.df.task == "stress_detection"]
        for col in ("f1", "accuracy", "ece"):
            self.assertTrue(np.isfinite(cls[col]).all(), f"{col} not finite")
        fc = self.df[self.df.task == "glucose"]
        for col in ("mae", "coverage"):
            self.assertTrue(np.isfinite(fc[col]).all(), f"{col} not finite")

    def test_does_not_assert_fused_beats_single(self):
        # We only assert both are present and comparable — NOT that fusion wins.
        cls = self.df[self.df.task == "stress_detection"]
        self.assertIn("single", set(cls.config_type))
        self.assertIn("fusion", set(cls.config_type))

    def test_summary_renders(self):
        md = ablation_summary(self.df)
        self.assertIn("stress_detection", md)
        self.assertIn("glucose", md)

    def test_determinism(self):
        df2 = run_ablation(self.ds, config=CFG, test_frac=0.3, seed=7)
        np.testing.assert_allclose(
            self.df["accuracy"].to_numpy(), df2["accuracy"].to_numpy(),
            rtol=1e-6, atol=1e-6, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
