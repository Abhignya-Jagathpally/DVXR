"""PR3: public component cohorts stay separate — no cohort co-registers EEG+CGM, and loaders keep
subject ids namespaced per cohort so rows from different cohorts can never masquerade as one person
(spec §1.A vs §1.B)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.cohort import COHORT_REGISTRY  # noqa: E402


class PublicCohortsSeparateTest(unittest.TestCase):
    def test_no_public_cohort_coregisters_eeg_and_cgm(self):
        """The core gap that keeps the glucose fusion claim research-stage."""
        for cid, spec in COHORT_REGISTRY.items():
            if cid == "synthetic":
                continue
            mods = spec.synchronized_modalities
            self.assertFalse({"eeg", "cgm"} <= set(mods),
                             f"{cid} unexpectedly co-registers BOTH eeg and cgm")

    def test_every_cohort_declares_a_synchronized_set(self):
        for cid, spec in COHORT_REGISTRY.items():
            self.assertTrue(spec.synchronized_modalities, f"{cid} has an empty synchronized set")
            self.assertEqual(spec.cohort_id, cid)

    def test_loaders_namespace_subject_ids_per_cohort(self):
        """Each loader prefixes subject ids with its cohort tag, so a raw concat of two cohorts never
        collides two different people onto one id (the structural guard against cross-joining)."""
        import inspect

        from dvxr import loaders
        src = inspect.getsource(loaders)
        # representative per-cohort namespacing tokens present in the loaders
        for token in ("mimic_", "shanghai_", "cgmacros_", "deap_", "noneeg_"):
            self.assertIn(token, src, f"expected per-cohort subject-id namespacing token {token!r}")

    def test_no_cross_cohort_join_helper_exists(self):
        """There must be no utility that merges two cohorts' events as one subject population."""
        from dvxr import loaders
        for bad in ("merge_cohorts", "cross_join_cohorts", "combine_subjects_across_cohorts"):
            self.assertFalse(hasattr(loaders, bad), f"loaders exposes a cross-cohort join: {bad}")


if __name__ == "__main__":
    unittest.main()
