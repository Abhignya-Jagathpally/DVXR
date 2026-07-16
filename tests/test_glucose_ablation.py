"""PR17 / Gate 8: the glucose ablation honestly evaluates the arms a single cohort supports (CGM-only,
CGM+wearable) and refuses every EEG/fused arm as cannot_evaluate — the release gate that keeps the
fused headline gated until synchronized data exists (spec §9)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.eval.glucose_ablation import (  # noqa: E402
    GATED_ARMS,
    HONEST_ARMS,
    run_glucose_ablation,
)
from dvxr.targets import ExcursionThresholds  # noqa: E402


def _synthetic_cohort(n_subjects=12, n=200):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        glu = np.clip(base + rs.normal(0, 12, n) + (25 * np.sin(np.arange(n) / 6.0) if high else 0),
                      55, 320)
        hr = np.clip(70 + (15 if high else 0) + rs.normal(0, 5, n), 45, 160)
        mets = np.clip(1.0 + rs.gamma(1.2, 0.4, n), 0.5, 8.0)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts,
                                    "glucose": glu, "hr": hr, "mets": mets}))
    return pd.concat(frames, ignore_index=True)


class GlucoseAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cgm = _synthetic_cohort()
        cls.rep = run_glucose_ablation(cgm, thresholds=ExcursionThresholds(history_minutes=120),
                                       seeds=(1, 2), anchor_stride=8, max_anchors_per_subject=30)

    def test_eeg_and_fused_arms_cannot_be_evaluated(self):
        for arm in GATED_ARMS:
            self.assertIn(arm, self.rep.gated)
            self.assertEqual(self.rep.gated[arm]["status"], "cannot_evaluate")

    def test_honest_arms_report_metrics(self):
        for arm in HONEST_ARMS:
            self.assertIn(arm, self.rep.honest)
            res = self.rep.honest[arm]
            if res.get("status") == "insufficient_data":
                continue
            self.assertIn("auroc", res)
            self.assertGreaterEqual(res["auroc"], 0.0)
            self.assertEqual(res["modality_scope"], arm)

    def test_paired_delta_has_a_ci_and_honest_verdict(self):
        if self.rep.paired_delta is None:
            self.skipTest("both arms did not produce predictions on this tiny fixture")
        d = self.rep.paired_delta
        self.assertIn("point", d)
        self.assertEqual(len(d["ci95"]), 2)
        # adds_value is TRUE only when the CI lower bound clears 0 — a real gate, not a hope
        self.assertEqual(d["adds_value"], d["ci95"][0] > 0)

    def test_no_fabricated_fused_number_anywhere(self):
        # the gated arms must never carry an auroc/number
        for arm in GATED_ARMS:
            self.assertNotIn("auroc", self.rep.gated[arm])


if __name__ == "__main__":
    unittest.main()
