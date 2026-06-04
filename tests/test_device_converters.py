"""Tests for device converter scripts: Galea, EMOTIV, and VR session ingestion.

Each test calls convert(None, <tempfile>, demo=True) to exercise the full
conversion pipeline against synthetic data, then verifies the returned
DataFrame satisfies the canonical event schema requirements.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# Make goal1_pipeline importable
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.schemas import REQUIRED_EVENT_COLUMNS, validate_events


# ---- Loader helpers ---------------------------------------------------------

def _load_script(name: str):
    """Load a script from ROOT/scripts/<name>.py as a module."""
    script_path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- Tests ------------------------------------------------------------------

class TestGaleaConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("convert_galea_subject")

    def test_demo_returns_nonempty_frame(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertFalse(df.empty, "Galea demo convert() returned empty DataFrame")

    def test_demo_has_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_demo_passes_validate_events(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        # validate_events will raise on any schema violation
        validated = validate_events(df)
        self.assertFalse(validated.empty)

    def test_demo_modalities(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        modalities = set(df["modality"].unique())
        self.assertIn("eeg", modalities, "Expected 'eeg' modality in Galea output")
        self.assertIn("eda", modalities, "Expected 'eda' modality in Galea output")
        self.assertIn("ppg", modalities, "Expected 'ppg' modality in Galea output")

    def test_demo_device_and_source(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertTrue((df["device"] == "galea").all(), "device should be 'galea'")
        self.assertTrue(
            (df["source_system"] == "galea_headset").all(),
            "source_system should be 'galea_headset'",
        )

    def test_demo_eeg_unit(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        eeg_units = df.loc[df["modality"] == "eeg", "unit"].unique()
        self.assertIn("uV", eeg_units, "EEG channels should have unit 'uV'")

    def test_demo_values_numeric_no_nan(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertFalse(df["value"].isna().any(), "value column must not contain NaN")
        self.assertFalse(
            df["sampling_rate_hz"].isna().any(), "sampling_rate_hz must not contain NaN"
        )


class TestEmotivConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("convert_emotiv_subject")

    def test_demo_epocx_returns_nonempty_frame(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        self.assertFalse(df.empty, "EMOTIV EPOC X demo returned empty DataFrame")

    def test_demo_epocx_channel_count(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        n_channels = df["channel"].nunique()
        self.assertEqual(n_channels, 14, f"EPOC X should have 14 channels, got {n_channels}")

    def test_demo_epocx_channel_names(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        channels = set(df["channel"].unique())
        expected = {
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        }
        self.assertEqual(channels, expected, f"Channel names mismatch: {channels}")

    def test_demo_flex_channel_count(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="flex")
        n_channels = df["channel"].nunique()
        self.assertEqual(n_channels, 32, f"FLEX should have 32 channels, got {n_channels}")

    def test_demo_modality_is_eeg(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        modalities = set(df["modality"].unique())
        self.assertEqual(modalities, {"eeg"}, "EMOTIV should only produce 'eeg' modality")

    def test_demo_passes_validate_events(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        validated = validate_events(df)
        self.assertFalse(validated.empty)

    def test_demo_has_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_demo_unit_is_uv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        self.assertTrue((df["unit"] == "uV").all(), "All EMOTIV EEG channels should be in 'uV'")

    def test_demo_device_field(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True, device="epocx")
        self.assertTrue((df["device"] == "epocx").all())

    def test_demo_values_numeric_no_nan(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertFalse(df["value"].isna().any(), "value column must not contain NaN")

    def test_invalid_device_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            with self.assertRaises(ValueError):
                self.mod.convert(None, tmp.name, demo=True, device="unknown_device")


class TestVRSessionIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("ingest_vr_session")

    def test_demo_returns_nonempty_frame(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertFalse(df.empty, "VR session demo convert() returned empty DataFrame")

    def test_demo_has_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        for col in REQUIRED_EVENT_COLUMNS:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_demo_passes_validate_events(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        validated = validate_events(df)
        self.assertFalse(validated.empty)

    def test_demo_motion_modality(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        modalities = set(df["modality"].unique())
        self.assertIn("motion", modalities, "Expected 'motion' modality in VR output")

    def test_demo_behavior_modality(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        modalities = set(df["modality"].unique())
        self.assertIn("behavior", modalities, "Expected 'behavior' modality in VR output")

    def test_demo_ppg_modality(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        modalities = set(df["modality"].unique())
        self.assertIn("ppg", modalities, "Expected 'ppg' modality in VR output (HR overlay)")

    def test_demo_device_and_source(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertTrue((df["device"] == "vr_ar_headset").all(), "device should be 'vr_ar_headset'")
        self.assertTrue(
            (df["source_system"] == "vr_session").all(),
            "source_system should be 'vr_session'",
        )

    def test_demo_motion_units(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        motion = df[df["modality"] == "motion"]
        units = set(motion["unit"].unique())
        self.assertTrue(
            units.issubset({"m", "deg"}),
            f"Motion channels should use 'm' or 'deg', got {units}",
        )

    def test_demo_values_numeric_no_nan(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        self.assertFalse(df["value"].isna().any(), "value column must not contain NaN")
        self.assertFalse(
            df["sampling_rate_hz"].isna().any(), "sampling_rate_hz must not contain NaN"
        )

    def test_demo_head_pose_channels_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df = self.mod.convert(None, tmp.name, demo=True)
        motion_channels = set(df.loc[df["modality"] == "motion", "channel"].unique())
        expected_pose = {"pos_x", "pos_y", "pos_z", "yaw", "pitch", "roll"}
        self.assertTrue(
            expected_pose.issubset(motion_channels),
            f"Expected head pose channels {expected_pose}, found {motion_channels}",
        )


if __name__ == "__main__":
    unittest.main()
