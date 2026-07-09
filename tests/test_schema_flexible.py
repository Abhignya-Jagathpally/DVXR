"""Slice A: the canonical schema treats the 13 columns as a required floor, not an
exact set. Extra dataset-specific columns must survive validate_events."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.schemas import REQUIRED_EVENT_COLUMNS, validate_events  # noqa: E402


def _base_row(**overrides):
    row = {
        "subject_id": "s1",
        "session_id": "ses1",
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "source_system": "test",
        "device": "dev",
        "modality": "cgm",
        "channel": "glucose",
        "value": 100.0,
        "unit": "mg/dL",
        "sampling_rate_hz": 1.0,
        "quality_flag": "ok",
        "label_name": "",
        "label_value": "",
    }
    row.update(overrides)
    return row


class SchemaFlexibleTest(unittest.TestCase):
    def test_required_columns_still_enforced(self):
        df = pd.DataFrame([_base_row()]).drop(columns=["value"])
        with self.assertRaises(ValueError):
            validate_events(df)

    def test_exact_thirteen_unchanged(self):
        clean = validate_events(pd.DataFrame([_base_row()]))
        self.assertEqual(list(clean.columns), REQUIRED_EVENT_COLUMNS)

    def test_extra_columns_preserved(self):
        df = pd.DataFrame(
            [
                _base_row(glucose_source="libre", meal_photo_path="p/a.jpg"),
                _base_row(glucose_source="dexcom", meal_photo_path=""),
            ]
        )
        clean = validate_events(df)
        # required columns come first, extras preserved after
        self.assertEqual(clean.columns[: len(REQUIRED_EVENT_COLUMNS)].tolist(), REQUIRED_EVENT_COLUMNS)
        self.assertIn("glucose_source", clean.columns)
        self.assertIn("meal_photo_path", clean.columns)
        self.assertEqual(set(clean["glucose_source"]), {"libre", "dexcom"})


if __name__ == "__main__":
    unittest.main()
