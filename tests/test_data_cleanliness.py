"""Slice 5: data-quality regression checks on the real extracts. Skip-guarded so CI
stays offline-safe. Codifies the value-level audit (ranges, dedup keys, referencing)
so the loaders can't silently regress into emitting dirty data."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from dvxr.loaders import (  # noqa: E402
    load_cgmacros_dataset,
    load_deap_raw_bdf,
    load_wesad_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
WESAD = ROOT / "data" / "real" / "WESAD"
CGM = ROOT / "data" / "real" / "cgmacros"
RAW_BDF = ROOT / "data" / "real" / "deap" / "raw_bdf"

BASE_KEY = ["subject_id", "session_id", "timestamp_utc", "modality", "channel"]


@unittest.skipUnless(list(WESAD.glob("S*/S*.pkl")), "WESAD absent")
class WesadCleanTest(unittest.TestCase):
    def test_chest_wrist_channels_do_not_collide(self):
        df = load_wesad_dataset(WESAD, subjects=2, max_samples_per_channel=1000)
        # chest and wrist share modalities (EDA/TEMP/ACC); the channel must carry the
        # location so their rows never collapse onto the same canonical key.
        self.assertEqual(df.duplicated(BASE_KEY).sum(), 0)
        chans = set(df["channel"])
        self.assertIn("chest_eda", chans)
        self.assertIn("wrist_eda", chans)


@unittest.skipUnless(list(CGM.glob("**/CGMacros-*.csv")), "CGMacros absent")
class CgmacrosCleanTest(unittest.TestCase):
    def test_dual_cgm_disambiguated_by_source(self):
        df = load_cgmacros_dataset(CGM, subjects=2)
        # Libre + Dexcom are both (cgm, glucose) at the same timestamps by design; the
        # extra glucose_source column must fully disambiguate them.
        self.assertEqual(df.duplicated(BASE_KEY + ["glucose_source"]).sum(), 0)
        glu = df[df["modality"] == "cgm"]["value"]
        self.assertTrue(glu.between(30, 450).all())  # plausible mg/dL sensor range


@unittest.skipUnless(list(RAW_BDF.glob("*.bdf")), "raw DEAP .bdf absent")
class DeapRawCleanTest(unittest.TestCase):
    def test_referenced_eeg_and_no_status_channel(self):
        df = load_deap_raw_bdf(sorted(RAW_BDF.glob("*.bdf"))[0], max_seconds=12)
        eeg = df[df["modality"] == "eeg"]
        self.assertEqual(eeg["channel"].nunique(), 32)
        self.assertNotIn("Status", set(df["channel"]))  # trigger channel dropped
        # average-referenced + high-passed -> per-channel means near zero, plausible uV
        self.assertLess(eeg.groupby("channel")["value"].mean().abs().mean(), 5.0)
        self.assertLess(np.abs(eeg["value"]).max(), 1000.0)


if __name__ == "__main__":
    unittest.main()
