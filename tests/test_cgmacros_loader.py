"""Slice 3: CGMacros multimodal loader. Logic is verified on a synthetic fixture that
matches the published column dictionary; a skip-guarded test also runs against the real
PhysioNet extract when present (data/real/cgmacros)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.loaders import (  # noqa: E402
    _diabetes_status_from_a1c,
    load_cgmacros_dataset,
    load_cgmacros_subject,
)
from dvxr.schemas import REQUIRED_EVENT_COLUMNS, summarize_events  # noqa: E402

CGM_DIR = Path(__file__).resolve().parents[1] / "data" / "real" / "cgmacros"


def _synthetic_subject_csv(path: Path):
    rows = []
    base = pd.Timestamp("2021-06-01 08:00")
    for i in range(10):
        rows.append(
            {
                "Timestamp": (base + pd.Timedelta(minutes=i)).strftime("%m/%d/%Y %H:%M"),
                "Libre GL": 100 + i,
                "Dexcom GL": 98 + i,
                "HR": 70 + i,
                "Calories (Activity)": 1.2,
                "Mets": 12,
                "Meal Type": "Breakfast" if i == 2 else "",
                "Calories": 450 if i == 2 else "",
                "Carbs": 60 if i == 2 else "",
                "Protein": 20 if i == 2 else "",
                "Fat": 15 if i == 2 else "",
                "Fiber": 5 if i == 2 else "",
                "Amount Consumed": 100 if i == 2 else "",
                "Image Path": "photos/b.jpg" if i == 2 else "",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class CGMacrosLoaderTest(unittest.TestCase):
    def test_diabetes_status_thresholds(self):
        self.assertEqual(_diabetes_status_from_a1c(6.7), "diabetes")
        self.assertEqual(_diabetes_status_from_a1c(6.0), "prediabetes")
        self.assertEqual(_diabetes_status_from_a1c(5.2), "healthy")
        self.assertEqual(_diabetes_status_from_a1c(None), "")

    def test_synthetic_subject_multimodal(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "CGMacros-001.csv"
            _synthetic_subject_csv(csv)
            df = load_cgmacros_subject(csv, diabetes_status="prediabetes")
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns)
        mods = set(summarize_events(df).modalities)
        self.assertEqual({"cgm", "wearable_phys", "behavior"}, mods)
        # both CGM sources preserved via the extra column
        cgm = df[df["modality"] == "cgm"]
        self.assertEqual(set(cgm["glucose_source"]), {"libre", "dexcom"})
        # meal event carries type + photo extras
        behavior = df[df["modality"] == "behavior"]
        self.assertTrue((behavior["meal_type"] == "Breakfast").all())
        self.assertIn("photos/b.jpg", set(behavior["meal_photo_path"]))
        # real diabetes label attached
        self.assertEqual(set(df["label_value"]), {"prediabetes"})


@unittest.skipUnless(list(CGM_DIR.glob("**/CGMacros-*.csv")), "real CGMacros extract not present")
class CGMacrosRealDataTest(unittest.TestCase):
    def test_real_dataset_loads(self):
        df = load_cgmacros_dataset(CGM_DIR, subjects=3)
        mods = set(summarize_events(df).modalities)
        self.assertTrue({"cgm", "wearable_phys", "behavior"}.issubset(mods))
        # bio.csv provides ehr strata when present
        self.assertIn("cgm", mods)


if __name__ == "__main__":
    unittest.main()
