"""PR32 / P0-4: the corrected ablation methodology. The alert threshold is frozen on a calibration
fold (no test leakage), arms are paired by exact example key (not truncation), the delta CI bootstraps
PARTICIPANTS (not rows), test folds are disjoint by subject, and person-time is real."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.eval.glucose_ablation import (  # noqa: E402
    _paired_auroc_delta, _subject_folds, run_glucose_ablation)
from dvxr.eval.splits import InsufficientSubjectsError, subject_kfold  # noqa: E402
from dvxr.targets import ExcursionThresholds  # noqa: E402


def _cohort(n_subjects=15, n=220):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        glu = np.clip(base + rs.normal(0, 12, n) + (25 * np.sin(np.arange(n) / 6.0) if high else 0), 55, 320)
        hr = np.clip(70 + (15 if high else 0) + rs.normal(0, 5, n), 45, 160)
        mets = np.clip(1.0 + rs.gamma(1.2, 0.4, n), 0.5, 8.0)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts,
                                    "glucose": glu, "hr": hr, "mets": mets}))
    return pd.concat(frames, ignore_index=True)


class SplitsTest(unittest.TestCase):
    def test_kfold_test_subjects_are_disjoint_and_cover_all(self):
        subs = np.array([f"s{i//4}" for i in range(40)])   # 10 subjects, repeated rows
        folds = subject_kfold(subs, n_folds=5, seed=1)
        seen_subjects = set()
        for _tr, te in folds:
            te_subj = set(subs[te])
            self.assertFalse(te_subj & seen_subjects)      # disjoint across folds
            seen_subjects |= te_subj
            # a test fold's subjects never appear in its own train fold
            self.assertFalse(te_subj & set(subs[_tr]))
        self.assertEqual(seen_subjects, set(subs))         # covers everyone once

    def test_kfold_raises_on_too_few_subjects(self):
        with self.assertRaises(InsufficientSubjectsError):
            subject_kfold(np.array(["a", "b"]), n_folds=5)


class PairingAndBootstrapTest(unittest.TestCase):
    def test_delta_pairs_by_exact_key_and_bootstraps_participants(self):
        # two arms over the SAME keys; arm A ranks better. Keys deliberately in different dict order.
        pooled_b = {f"s{s}|k{i}": {"y": i % 2, "p": 0.5, "subject": f"s{s}"}
                    for s in range(6) for i in range(4)}
        pooled_a = {k: {"y": v["y"], "p": 0.9 if v["y"] else 0.1, "subject": v["subject"]}
                    for k, v in pooled_b.items()}
        d = _paired_auroc_delta(pooled_b, pooled_a, n_boot=100, seed=0)
        self.assertEqual(d["bootstrap_unit"], "participant")
        self.assertEqual(d["n_paired"], len(pooled_b))
        self.assertGreater(d["point"], 0.0)                # arm A separates, arm B does not

    def test_disjoint_keys_yield_no_pairing(self):
        d = _paired_auroc_delta({"k1": {"y": 1, "p": 0.9, "subject": "s1"}},
                                {"k2": {"y": 0, "p": 0.1, "subject": "s2"}}, n_boot=10)
        self.assertEqual(d["n_paired"], 0)


class EndToEndMethodologyTest(unittest.TestCase):
    def test_report_carries_corrected_metrics_and_person_time(self):
        rep = run_glucose_ablation(_cohort(), thresholds=ExcursionThresholds(history_minutes=120),
                                   seeds=(1,), anchor_stride=8, max_anchors_per_subject=30, n_folds=5)
        for arm, res in rep.honest.items():
            if res.get("status") == "insufficient_data":
                continue
            for field in ("auprc", "ece", "sensitivity_at_frozen_threshold",
                          "false_alerts_per_participant_day", "n_subjects_test"):
                self.assertIn(field, res)
        if rep.paired_delta and rep.paired_delta.get("n_paired"):
            self.assertEqual(rep.paired_delta["bootstrap_unit"], "participant")


if __name__ == "__main__":
    unittest.main()
