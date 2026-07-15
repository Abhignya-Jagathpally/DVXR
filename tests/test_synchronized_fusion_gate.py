"""PR3: the fusion honesty gate (spec §1.B, §4) blocks multimodal claims a cohort can't support."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.cohort import (  # noqa: E402
    COHORT_REGISTRY,
    GLUCOSE_FUSION_MODALITIES,
    SynchronyError,
    can_fuse,
    cohort_synchrony,
    require_synchronized_for_fusion,
)


class FusionGateTest(unittest.TestCase):
    def test_single_modality_is_never_gated(self):
        require_synchronized_for_fusion("wesad", ["wearable_phys"])   # no raise
        self.assertTrue(can_fuse("mumtaz", ["eeg"]))

    def test_within_cohort_synchronized_fusion_is_allowed(self):
        require_synchronized_for_fusion("deap", ["eeg", "wearable_phys"])   # DEAP co-registers both
        self.assertTrue(can_fuse("cgmacros", ["cgm", "wearable_phys", "behavior"]))

    def test_eeg_plus_cgm_is_blocked_on_every_public_cohort(self):
        for cid, spec in COHORT_REGISTRY.items():
            if cid == "synthetic":
                continue
            with self.assertRaises(SynchronyError, msg=f"{cid} must not permit EEG+CGM fusion"):
                require_synchronized_for_fusion(cid, ["eeg", "cgm"])
            self.assertFalse(can_fuse(cid, ["eeg", "cgm"]))

    def test_glucose_product_fusion_set_is_unsupported_by_public_data(self):
        for cid in COHORT_REGISTRY:
            if cid == "synthetic":
                continue
            self.assertFalse(can_fuse(cid, GLUCOSE_FUSION_MODALITIES),
                             f"{cid} must not support the glucose EEG+CGM+wearable fusion set")

    def test_unknown_cohort_is_denied(self):
        self.assertFalse(can_fuse("not-a-cohort", ["eeg", "cgm"]))
        with self.assertRaises(SynchronyError):
            require_synchronized_for_fusion("not-a-cohort", ["eeg", "cgm"])

    def test_synthetic_fixture_is_synchronized(self):
        require_synchronized_for_fusion("synthetic", ["eeg", "wearable_phys", "cgm"])  # no raise
        self.assertTrue(cohort_synchrony("synthetic").synchronized_same_subject)

    def test_ablation_refuses_unsynchronized_fusion(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        import numpy as np
        from dvxr.eval.ablation import make_synthetic_dataset, run_ablation
        ds = make_synthetic_dataset(n_subjects=6, per_subject=6, seed=0)
        ds["cohort_id"] = "wesad"          # claim EEG+wearable+CGM fusion on a wearable-only cohort
        with self.assertRaises(SynchronyError):
            run_ablation(ds, seed=7)


if __name__ == "__main__":
    unittest.main()
