"""PR4: a MISSING feature must be distinguishable from a genuine 0.0 (spec §8, §9)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.features import (  # noqa: E402
    MISSING_MASK_SUFFIX,
    add_missingness_masks,
    feature_columns,
    missingness_columns,
)


class MissingIsNotZeroTest(unittest.TestCase):
    def _frame(self):
        # row 0: genuine zero; row 1: missing (NaN); row 2: normal value
        return pd.DataFrame({
            "subject_id": ["s", "s", "s"],
            "eeg_alpha": [0.0, np.nan, 1.5],
            "target": ["high", "low", "high"],
        })

    def test_mask_distinguishes_missing_from_zero(self):
        masked = add_missingness_masks(self._frame(), ["eeg_alpha"])
        present = masked[f"eeg_alpha{MISSING_MASK_SUFFIX}"].tolist()
        self.assertEqual(present, [1.0, 0.0, 1.0])   # true 0.0 is present; NaN is missing

    def test_mask_columns_are_not_features(self):
        masked = add_missingness_masks(self._frame(), ["eeg_alpha"])
        self.assertIn("eeg_alpha" + MISSING_MASK_SUFFIX, missingness_columns(masked))
        self.assertNotIn("eeg_alpha" + MISSING_MASK_SUFFIX, feature_columns(masked))
        self.assertIn("eeg_alpha", feature_columns(masked))

    def test_zero_fill_after_masking_is_recoverable(self):
        masked = add_missingness_masks(self._frame(), ["eeg_alpha"])
        filled = masked.fillna(0.0)
        # both the true-zero row and the was-missing row now read 0.0 ...
        self.assertEqual(filled["eeg_alpha"].tolist(), [0.0, 0.0, 1.5])
        # ... but the mask still tells them apart (this is the property the old fillna(0) destroyed)
        self.assertEqual(filled[f"eeg_alpha{MISSING_MASK_SUFFIX}"].tolist(), [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
