from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.eval.paper import build_paper_tables, df_to_booktabs  # noqa: E402


class BooktabsTest(unittest.TestCase):
    def test_escaping_and_structure(self):
        df = pd.DataFrame({"config_name": ["late_weighted"], "auroc": [0.873]})
        tex = df_to_booktabs(df, "Cap", "tab:x")
        self.assertIn(r"\toprule", tex)
        self.assertIn(r"\bottomrule", tex)
        self.assertIn(r"late\_weighted", tex)   # underscore escaped
        self.assertIn("0.873", tex)


class BuildPaperTablesTest(unittest.TestCase):
    def _fixture_outputs(self, d: Path):
        pd.DataFrame({
            "task": ["stress_detection", "stress_detection"],
            "config_type": ["single", "fusion"],
            "config_name": ["wearable_phys", "cross_modal"],
            "auroc": [0.948, 0.398], "f1": [0.90, 0.73], "accuracy": [0.87, 0.58],
            "ece": [0.14, 0.02], "mae": [float("nan"), float("nan")],
            "coverage": [float("nan"), float("nan")],
        }).to_csv(d / "ablation_table.csv", index=False)
        pd.DataFrame({
            "code_index": [0, 1, 2], "count": [3, 2, 1],
            "frequency": [0.5, 0.333, 0.167],
        }).to_csv(d / "codebook_usage.csv", index=False)

    def test_builds_valid_tex_without_latex(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            out.mkdir()
            tables = Path(tmp) / "paper" / "tables"
            self._fixture_outputs(out)
            manifest = build_paper_tables(out, tables)

            self.assertIn("ablation", manifest)
            self.assertIn("codebook", manifest)
            abl_tex = (tables / "ablation.tex").read_text()
            self.assertIn(r"\toprule", abl_tex)
            self.assertIn("0.948", abl_tex)                  # traces to fixture
            self.assertIn(r"wearable\_phys", abl_tex)
            self.assertTrue((tables / "MANIFEST.txt").exists())
            # every table records its source outputs/ file
            for info in manifest.values():
                self.assertTrue(Path(info["source"]).exists())

    def test_missing_outputs_skipped_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            out.mkdir()
            manifest = build_paper_tables(out, Path(tmp) / "tables")
            self.assertEqual(manifest, {})   # nothing invented


if __name__ == "__main__":
    unittest.main()
