"""PR4: a scientific/production run must never fabricate subjects or flip labels (spec §12)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.clinical_tasks import ScientificValidityError, derive_task_labels  # noqa: E402
from dvxr.sample_data import generate_public_like_events  # noqa: E402


def _events(subjects, seed=7):
    with tempfile.TemporaryDirectory() as d:
        return generate_public_like_events(Path(d) / "e.csv", subjects=subjects, minutes=18,
                                           eeg_channels=6, eeg_rate_hz=16.0, seed=seed)


class ScientificLabelIntegrityTest(unittest.TestCase):
    def test_adequate_data_passes_scientific_mode(self):
        frame = derive_task_labels(_events(6), "stress_detection")   # default allow_synthetic=False
        self.assertGreaterEqual(frame["subject_id"].nunique(), 4)
        self.assertGreaterEqual(len(set(frame["target"].unique())), 2)

    def test_too_few_subjects_raises_in_scientific_mode(self):
        with self.assertRaises(ScientificValidityError):
            derive_task_labels(_events(2), "stress_detection")

    def test_single_class_proxy_raises_in_scientific_mode(self):
        # cognitive_workload's median-split proxy collapses to one class on this fixture; scientific
        # mode refuses to flip labels to manufacture the second class.
        with self.assertRaises(ScientificValidityError):
            derive_task_labels(_events(6), "cognitive_workload")

    def test_synthetic_flag_allows_smoke_fixture(self):
        # the same inadequate data is permitted ONLY with the explicit smoke flag
        frame = derive_task_labels(_events(2), "stress_detection", allow_synthetic=True)
        self.assertGreaterEqual(frame["subject_id"].nunique(), 4)   # synthesized up to the floor

    def test_no_synth_subject_ids_leak_into_scientific_output(self):
        frame = derive_task_labels(_events(6), "stress_detection")
        self.assertFalse(any(str(s).startswith("synth_") for s in frame["subject_id"].unique()))


if __name__ == "__main__":
    unittest.main()
