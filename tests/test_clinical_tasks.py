"""tests/test_clinical_tasks.py — Tests for the clinical_tasks module."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.clinical_tasks import (
    CLINICAL_TASKS,
    clinical_tasks_table,
    derive_task_labels,
    train_clinical_task,
)
from goal1_pipeline.sample_data import generate_public_like_events


def _make_events(subjects=6, minutes=18, seed=7):
    with tempfile.TemporaryDirectory() as tmpdir:
        events = generate_public_like_events(
            Path(tmpdir) / "events.csv",
            subjects=subjects,
            minutes=minutes,
            eeg_channels=6,
            eeg_rate_hz=16.0,
            seed=seed,
        )
    return events


class TestClinicalTasksTable(unittest.TestCase):
    def test_table_has_seven_rows(self):
        table = clinical_tasks_table()
        self.assertEqual(len(table), 7, f"Expected 7 rows, got {len(table)}")

    def test_table_columns(self):
        table = clinical_tasks_table()
        for col in ("name", "positive_label", "negative_label", "proxy_description", "source_modalities"):
            self.assertIn(col, table.columns)

    def test_registry_has_seven_tasks(self):
        self.assertEqual(len(CLINICAL_TASKS), 7)


class TestCognitiveWorkload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = _make_events(subjects=6, minutes=18, seed=7)

    def test_derive_labels_both_classes(self):
        frame = derive_task_labels(self.events, "cognitive_workload")
        classes = set(frame["target"].unique())
        self.assertIn("high_workload", classes, f"Missing high_workload in {classes}")
        self.assertIn("low_workload", classes, f"Missing low_workload in {classes}")

    def test_derive_labels_enough_subjects(self):
        frame = derive_task_labels(self.events, "cognitive_workload")
        self.assertGreaterEqual(frame["subject_id"].nunique(), 4)

    def test_train_returns_model_with_accuracy(self):
        frame = derive_task_labels(self.events, "cognitive_workload")
        model = train_clinical_task(frame, "cognitive_workload")
        self.assertIsInstance(model.metrics, dict)
        self.assertIn("accuracy", model.metrics)

    def test_train_predictions_have_probability_col(self):
        frame = derive_task_labels(self.events, "cognitive_workload")
        model = train_clinical_task(frame, "cognitive_workload")
        self.assertIn("cognitive_workload_probability", model.predictions.columns)


class TestGlucoseInstability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = _make_events(subjects=6, minutes=18, seed=42)

    def test_derive_labels_both_classes(self):
        frame = derive_task_labels(self.events, "glucose_instability")
        classes = set(frame["target"].unique())
        self.assertIn("unstable", classes, f"Missing unstable in {classes}")
        self.assertIn("stable", classes, f"Missing stable in {classes}")

    def test_derive_labels_enough_subjects(self):
        frame = derive_task_labels(self.events, "glucose_instability")
        self.assertGreaterEqual(frame["subject_id"].nunique(), 4)

    def test_train_returns_model_with_accuracy(self):
        frame = derive_task_labels(self.events, "glucose_instability")
        model = train_clinical_task(frame, "glucose_instability")
        self.assertIsInstance(model.metrics, dict)
        self.assertIn("accuracy", model.metrics)

    def test_train_predictions_have_probability_col(self):
        frame = derive_task_labels(self.events, "glucose_instability")
        model = train_clinical_task(frame, "glucose_instability")
        self.assertIn("glucose_instability_probability", model.predictions.columns)


class TestStressDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = _make_events(subjects=6, minutes=18, seed=7)

    def test_derive_labels_both_classes(self):
        frame = derive_task_labels(self.events, "stress_detection")
        classes = set(frame["target"].unique())
        self.assertIn("stress", classes, f"Missing stress in {classes}")
        self.assertIn("non_stress", classes, f"Missing non_stress in {classes}")

    def test_derive_labels_enough_subjects(self):
        frame = derive_task_labels(self.events, "stress_detection")
        self.assertGreaterEqual(frame["subject_id"].nunique(), 4)

    def test_train_returns_model_with_accuracy(self):
        frame = derive_task_labels(self.events, "stress_detection")
        model = train_clinical_task(frame, "stress_detection")
        self.assertIsInstance(model.metrics, dict)
        self.assertIn("accuracy", model.metrics)

    def test_train_predictions_have_probability_col(self):
        frame = derive_task_labels(self.events, "stress_detection")
        model = train_clinical_task(frame, "stress_detection")
        self.assertIn("stress_detection_probability", model.predictions.columns)


class TestAllTasksSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = _make_events(subjects=6, minutes=18, seed=99)

    def test_all_tasks_derive_and_train(self):
        for task in CLINICAL_TASKS:
            with self.subTest(task=task.name):
                frame = derive_task_labels(self.events, task.name)
                self.assertGreaterEqual(len(frame), 4, f"Too few windows for {task.name}")
                model = train_clinical_task(frame, task.name)
                self.assertIn("accuracy", model.metrics, f"No accuracy in {task.name} metrics")


if __name__ == "__main__":
    unittest.main()
