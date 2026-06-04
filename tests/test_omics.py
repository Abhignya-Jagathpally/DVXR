"""Tests for goal1_pipeline.omics — multi-omics ingestion and feature building."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure ROOT/src is importable regardless of how tests are invoked
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from goal1_pipeline.omics import build_omics_features, generate_omics_like_table, load_omics_table
from goal1_pipeline.schemas import validate_events


class TestGenerateOmicsLikeTable(unittest.TestCase):
    """Tests for generate_omics_like_table."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_csv = Path(self._tmpdir.name) / "omics_events.csv"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_returns_validated_frame(self):
        """generate_omics_like_table must return a validate_events-clean frame."""
        events = generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        # Should not raise — re-validation is the canonical check
        clean = validate_events(events)
        self.assertEqual(len(clean), len(events))

    def test_csv_written(self):
        """Output CSV must exist after generation."""
        generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        self.assertTrue(self.tmp_csv.exists(), "Output CSV was not written.")

    def test_modality_is_omics(self):
        """All rows must have modality=='omics'."""
        events = generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        self.assertTrue(
            (events["modality"] == "omics").all(),
            "Expected all rows to have modality=='omics'.",
        )

    def test_two_label_classes(self):
        """Both 'high_risk' and 'low_risk' must appear in label_value."""
        events = generate_omics_like_table(self.tmp_csv, subjects=8, seed=23)
        classes = set(events["label_value"].unique())
        self.assertIn("high_risk", classes, "Missing 'high_risk' label class.")
        self.assertIn("low_risk", classes, "Missing 'low_risk' label class.")

    def test_subject_count(self):
        """Number of unique subjects must match the subjects parameter."""
        n = 6
        events = generate_omics_like_table(self.tmp_csv, subjects=n, seed=23)
        self.assertEqual(events["subject_id"].nunique(), n)

    def test_feature_channels_present(self):
        """gene_, protein_, and metab_ channels must all be present."""
        events = generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        channels = set(events["channel"].unique())
        has_gene = any(c.startswith("gene_") for c in channels)
        has_protein = any(c.startswith("protein_") for c in channels)
        has_metab = any(c.startswith("metab_") for c in channels)
        self.assertTrue(has_gene, "No gene_ channels found.")
        self.assertTrue(has_protein, "No protein_ channels found.")
        self.assertTrue(has_metab, "No metab_ channels found.")

    def test_units_by_omic_type(self):
        """gene_→expr, protein_→abundance, metab_→conc."""
        events = generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        for _, row in events.iterrows():
            if row["channel"].startswith("gene_"):
                self.assertEqual(row["unit"], "expr")
            elif row["channel"].startswith("protein_"):
                self.assertEqual(row["unit"], "abundance")
            elif row["channel"].startswith("metab_"):
                self.assertEqual(row["unit"], "conc")

    def test_sampling_rate_is_zero(self):
        """sampling_rate_hz must be 0.0 for omics (static panel)."""
        events = generate_omics_like_table(self.tmp_csv, subjects=4, seed=23)
        self.assertTrue(
            (events["sampling_rate_hz"] == 0.0).all(),
            "Expected sampling_rate_hz==0.0 for all omics events.",
        )


class TestBuildOmicsFeatures(unittest.TestCase):
    """Tests for build_omics_features."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_csv = Path(self._tmpdir.name) / "omics_events.csv"
        self.subjects = 8
        self.events = generate_omics_like_table(
            self.tmp_csv, subjects=self.subjects, n_genes=10, n_proteins=5, n_metabolites=5, seed=23
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_one_row_per_subject(self):
        """build_omics_features must return exactly subjects rows."""
        features = build_omics_features(self.events)
        self.assertEqual(
            len(features),
            self.subjects,
            f"Expected {self.subjects} rows, got {len(features)}.",
        )

    def test_subject_id_column(self):
        """Output must contain a subject_id column."""
        features = build_omics_features(self.events)
        self.assertIn("subject_id", features.columns)

    def test_target_column_values(self):
        """target column must only contain 'high_risk' or 'low_risk'."""
        features = build_omics_features(self.events)
        self.assertIn("target", features.columns, "Missing 'target' column.")
        valid_targets = {"high_risk", "low_risk"}
        actual = set(features["target"].unique())
        self.assertTrue(
            actual.issubset(valid_targets),
            f"Unexpected target values: {actual - valid_targets}",
        )

    def test_both_target_classes(self):
        """Both 'high_risk' and 'low_risk' must appear in the target column."""
        features = build_omics_features(self.events)
        classes = set(features["target"].unique())
        self.assertIn("high_risk", classes, "Missing 'high_risk' in target.")
        self.assertIn("low_risk", classes, "Missing 'low_risk' in target.")

    def test_numeric_omic_columns_present(self):
        """There must be at least one numeric omic feature column."""
        features = build_omics_features(self.events)
        non_meta = [c for c in features.columns if c not in ("subject_id", "session_id", "target")]
        self.assertGreater(len(non_meta), 0, "No omic feature columns found.")
        # All omic columns should be numeric
        for col in non_meta:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(features[col]),
                f"Omic column '{col}' is not numeric.",
            )

    def test_session_id_column(self):
        """Output must contain a session_id column."""
        features = build_omics_features(self.events)
        self.assertIn("session_id", features.columns)

    def test_omic_column_count(self):
        """Number of omic columns must equal n_genes + n_proteins + n_metabolites."""
        features = build_omics_features(self.events)
        omic_cols = [c for c in features.columns if c not in ("subject_id", "session_id", "target")]
        expected = 10 + 5 + 5  # n_genes + n_proteins + n_metabolites
        self.assertEqual(len(omic_cols), expected)


class TestLoadOmicsTable(unittest.TestCase):
    """Tests for load_omics_table (wide CSV ingestion)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # Build a small wide CSV by generating events and pivoting manually
        self.wide_csv = Path(self._tmpdir.name) / "wide.csv"
        self.events_csv = Path(self._tmpdir.name) / "events.csv"
        self.subjects = 4
        self.events_ref = generate_omics_like_table(
            self.events_csv,
            subjects=self.subjects,
            n_genes=5,
            n_proteins=3,
            n_metabolites=2,
            seed=99,
        )
        # Build wide CSV from the reference events
        import pandas as pd

        pivot = self.events_ref.pivot_table(
            index="subject_id", columns="channel", values="value", aggfunc="mean"
        ).reset_index()
        label_map = (
            self.events_ref.groupby("subject_id")["label_value"].first().reset_index()
        )
        wide = pivot.merge(label_map, on="subject_id")
        wide.to_csv(self.wide_csv, index=False)
        self.wide_label_col = "label_value"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_produces_valid_events(self):
        """load_omics_table must return a validate_events-clean frame."""
        events = load_omics_table(self.wide_csv, label_col=self.wide_label_col)
        clean = validate_events(events)
        self.assertEqual(len(clean), len(events))

    def test_modality_is_omics(self):
        """Loaded events must all have modality=='omics'."""
        events = load_omics_table(self.wide_csv, label_col=self.wide_label_col)
        self.assertTrue((events["modality"] == "omics").all())

    def test_subject_count_matches(self):
        """Unique subjects must match original synthetic count."""
        events = load_omics_table(self.wide_csv, label_col=self.wide_label_col)
        self.assertEqual(events["subject_id"].nunique(), self.subjects)


import pandas as pd  # noqa: E402  (needed for setUp above)

if __name__ == "__main__":
    unittest.main()
