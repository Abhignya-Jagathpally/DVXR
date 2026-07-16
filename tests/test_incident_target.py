"""PR35 / P0-3: the excursion target distinguishes INCIDENT onset from persistence.

An early-warning model must predict a *new* excursion for someone currently in range — not merely
detect that an already-hyperglycaemic participant stays high (a persistence detector). These tests pin
the incident / persistent / recovery taxonomy and the `label_definition="incident"` reportable set.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.targets import ExcursionThresholds, build_excursion_labels  # noqa: E402
from dvxr.targets.excursion import _anchor_state, _classify_outcome  # noqa: E402


def _cohort():
    t = pd.date_range("2020-01-01", periods=20, freq="15min")
    return pd.concat([
        pd.DataFrame({"subject_id": "incident", "timestamp": t, "glucose": [100] * 10 + [200] * 10}),
        pd.DataFrame({"subject_id": "persistent", "timestamp": t, "glucose": [210] * 20}),
        pd.DataFrame({"subject_id": "recovery", "timestamp": t, "glucose": [200] * 10 + [110] * 10}),
    ], ignore_index=True)


class ClassifierUnitTest(unittest.TestCase):
    def test_anchor_state_thresholds(self):
        thr = ExcursionThresholds()
        self.assertEqual(_anchor_state(100.0, thr), "in_range")
        self.assertEqual(_anchor_state(60.0, thr), "hypo")
        self.assertEqual(_anchor_state(200.0, thr), "hyper")
        self.assertEqual(_anchor_state(None, thr), "unknown")

    def test_outcome_taxonomy(self):
        # in range at t + future excursion → incident onset (primary label 1)
        self.assertEqual(_classify_outcome("in_range", 1, None), ("incident_excursion", 1))
        self.assertEqual(_classify_outcome("in_range", 0, None), ("no_excursion", 0))
        # out of range at t → persistence (still out) or recovery (back in range); excluded from incident
        self.assertEqual(_classify_outcome("hyper", 1, False), ("persistent_excursion", None))
        self.assertEqual(_classify_outcome("hyper", 1, True), ("recovery", None))
        # censored → nothing
        self.assertEqual(_classify_outcome("in_range", None, None), (None, None))


class TaxonomyTableTest(unittest.TestCase):
    def setUp(self):
        self.thr = ExcursionThresholds(history_minutes=60, horizons_minutes=(30,))

    def test_incident_only_scores_in_range_anchors(self):
        df = build_excursion_labels(_cohort(), thresholds=self.thr, subject_col="subject_id",
                                    label_definition="incident")
        rep = df[df.censored == False]  # noqa: E712
        # every reportable incident anchor was IN RANGE at t
        self.assertEqual(set(rep["anchor_state"].unique()), {"in_range"})
        # the "incident" subject contributes at least one positive incident onset
        inc = rep[(rep.subject_id == "incident") & (rep.label == 1)]
        self.assertGreater(len(inc), 0)
        self.assertTrue((inc["outcome_class"] == "incident_excursion").all())

    def test_persistent_anchor_is_excluded_from_incident_set(self):
        df = build_excursion_labels(_cohort(), thresholds=self.thr, subject_col="subject_id",
                                    label_definition="incident")
        # the always-high subject has NO reportable incident rows (all censored out_of_range_at_anchor)
        persistent = df[df.subject_id == "persistent"]
        self.assertTrue((persistent["censored"] == True).all())  # noqa: E712
        self.assertTrue((persistent["censor_reason"] == "out_of_range_at_anchor").any())

    def test_recovery_is_labelled_and_distinct_from_persistence(self):
        df = build_excursion_labels(_cohort(), thresholds=self.thr, subject_col="subject_id")
        classes = set(df["outcome_class"].dropna().unique())
        self.assertIn("recovery", classes)
        self.assertIn("persistent_excursion", classes)
        self.assertIn("incident_excursion", classes)

    def test_any_definition_is_unchanged_backcompat(self):
        # the default "any" label still counts ANY future out-of-range sample (persistence included),
        # so the always-high subject is positive under "any" but censored under "incident".
        any_df = build_excursion_labels(_cohort(), thresholds=self.thr, subject_col="subject_id")
        rep = any_df[any_df.censored == False]  # noqa: E712
        persistent = rep[rep.subject_id == "persistent"]
        self.assertGreater(len(persistent), 0)
        self.assertTrue((persistent["label"] == 1).all())

    def test_invalid_label_definition_raises(self):
        with self.assertRaises(ValueError):
            build_excursion_labels(_cohort(), thresholds=self.thr, subject_col="subject_id",
                                   label_definition="bogus")


if __name__ == "__main__":
    unittest.main()
