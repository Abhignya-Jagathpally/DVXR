from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SAMPLE_EMOTIV = ROOT / "data" / "sample" / "emotiv"
SAMPLE_GALEA = ROOT / "data" / "sample" / "openbci"

try:
    import matplotlib  # noqa: F401
    import sklearn  # noqa: F401
    from dvxr.bci_real import ingest_emotiv, ingest_galea, epoch_emotiv, COMMAND_CLASSES
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "matplotlib/sklearn/bci deps required")
class BCISmokeTest(unittest.TestCase):
    def test_ingest_from_sample_directory(self):
        """A fresh clone (no .zip) can ingest the committed sample directory."""
        emo = ingest_emotiv(SAMPLE_EMOTIV)
        self.assertEqual(len(emo.ch_names), 14)
        self.assertGreater(emo.fs, 0)
        gal = ingest_galea(SAMPLE_GALEA, max_seconds=30)
        self.assertGreaterEqual(len(gal.ch_names), 1)

    def test_ingest_accepts_direct_csv(self):
        csvs = list(SAMPLE_EMOTIV.rglob("*.csv"))
        csvs = [p for p in csvs if "intervalMarker" not in p.name]
        self.assertTrue(csvs)
        emo = ingest_emotiv(csvs[0])
        self.assertEqual(len(emo.ch_names), 14)

    def test_epoch_does_not_crash_on_sample(self):
        import pandas as pd
        emo = ingest_emotiv(SAMPLE_EMOTIV)
        win = epoch_emotiv(emo, win_s=1.0, step_s=0.5, power_thresh=0.05)
        self.assertIsInstance(win, pd.DataFrame)   # frame (possibly empty), no crash

    def test_pipeline_runs_end_to_end_on_sample(self):
        """run_bci_pipeline.main() on the sample returns a metrics dict, no crash."""
        import run_bci_pipeline as bci
        report = bci.main(emotiv_src=str(SAMPLE_EMOTIV), galea_src=str(SAMPLE_GALEA))
        self.assertIsInstance(report, dict)
        self.assertEqual(report["data_source"], "sample")
        self.assertEqual(report["labels_source"], "emotiv_mc_engine")
        # committed full-run metrics.json must NOT be clobbered by a sample run
        self.assertFalse((ROOT / "outputs" / "bci" / "metrics.json").read_text().startswith(
            '{\n  "generated_by": "scripts/run_bci_pipeline.py",\n  "data_source": "sample"'))

    def test_resolver_defaults_to_sample_without_full_data(self):
        import run_bci_pipeline as bci
        path, is_full = bci.resolve_emotiv(None)
        # with no full recording present, resolver must fall back to the committed sample
        if not is_full:
            self.assertIn("sample", str(path).lower())


if __name__ == "__main__":
    unittest.main()
