"""PR37: a reportable run can never silently leak one person's windows across train/cal/test.

The legacy ``models.py`` fell back to a ROW-LEVEL split when fewer than three subjects were present,
placing windows from the same person into all three folds. That fallback now RAISES for reportable runs
and is reachable only behind an explicit demo opt-in. Also pins that every proxy clinical task is flagged
SCAFFOLDING ONLY (no unflagged circular-label task can masquerade as a real result).
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.clinical_tasks import CLINICAL_TASKS  # noqa: E402
from dvxr.models import (  # noqa: E402
    ScientificValidityError,
    _group_train_calibration_test_split,
)


def _frame(subjects):
    rows = [{"subject_id": s, "x": float(i)} for i, s in enumerate(subjects)]
    return pd.DataFrame(rows)


class RowLevelLeakGuardTest(unittest.TestCase):
    def test_fewer_than_three_subjects_raises_by_default(self):
        f = _frame(["s0"] * 5 + ["s1"] * 5)            # 2 subjects
        with self.assertRaises(ScientificValidityError):
            _group_train_calibration_test_split(f, f["subject_id"])

    def test_demo_optin_falls_back_with_a_warning(self):
        f = _frame(["s0"] * 6 + ["s1"] * 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tr, cal, te = _group_train_calibration_test_split(
                f, f["subject_id"], allow_row_level_fallback=True)
        self.assertTrue(any("demo-only fallback" in str(w.message) for w in caught))
        self.assertEqual(len(tr) + len(cal) + len(te), len(f))

    def test_three_or_more_subjects_split_is_subject_disjoint(self):
        subs = np.array(["s0", "s1", "s2", "s3", "s4"])
        f = _frame(np.repeat(subs, 4))
        tr, cal, te = _group_train_calibration_test_split(f, f["subject_id"])
        g = f["subject_id"].to_numpy()
        tr_s, cal_s, te_s = set(g[tr]), set(g[cal]), set(g[te])
        self.assertFalse(tr_s & te_s)                   # no subject in both train and test
        self.assertFalse(tr_s & cal_s)
        self.assertFalse(cal_s & te_s)


class ProxyTaskFlagTest(unittest.TestCase):
    def test_every_proxy_task_is_flagged_scaffolding(self):
        for task in CLINICAL_TASKS:
            desc = task.proxy_description or ""
            if not desc:                                # a direct-label task (e.g. stress_detection)
                continue
            # a proxy label derived from a median split of its own features must say SCAFFOLDING ONLY
            if "median" in desc.lower() and "proxy" in desc.lower():
                self.assertIn("SCAFFOLDING ONLY", desc,
                              f"proxy task {task.name!r} is not flagged SCAFFOLDING ONLY")


if __name__ == "__main__":
    unittest.main()
