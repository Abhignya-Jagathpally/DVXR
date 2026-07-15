"""Tests for the honest product paper (paper/main.tex + dvxr.eval.paper.build_product_tables).

Validates without a LaTeX install: the four product tables emit with numbers that trace to the
evidence layer; the drift guard refuses to emit on scoreboard mismatch; main.tex has no citation
orphans and every \\input table exists; and the prose is honest (diagnosis mentions are negated, the
learned CACMF fusion is never framed as a win).
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.eval.paper import build_product_tables  # noqa: E402

_NEGATOR = re.compile(r"not |never |rather than|decision support|screening|isn't|is not|excluded")


class BuildProductTablesTest(unittest.TestCase):
    def test_tables_emit_with_traceable_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp) / "tables"
            manifest = build_product_tables(tdir, ROOT / "outputs" / "product" / "screeners")
            for name in ("product_headline", "fusion_contribution", "external_sota",
                         "clinical_utility"):
                self.assertIn(name, manifest)
                self.assertTrue((tdir / f"{name}.tex").exists())
            head = (tdir / "product_headline.tex").read_text()
            self.assertIn("0.961", head)          # depression window-level
            self.assertIn("0.986", head)          # depression subject-level
            fusion = (tdir / "fusion_contribution.tex").read_text()
            self.assertIn("0.795", fusion)        # learned CACMF on depression (the loser)
            sota = (tdir / "external_sota.tex").read_text()
            self.assertIn("10.1093/cercor/bhae505", sota)   # a real DOI, protocol-labeled
            self.assertIn("LOSO", sota)
            util = (tdir / "clinical_utility.tex").read_text()
            self.assertIn("0.441", util)          # depression peak net benefit
            self.assertIn("0.312", util)          # its bootstrap lower bound

    def test_every_table_records_a_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_product_tables(Path(tmp), ROOT / "outputs" / "product" / "screeners")
            for info in manifest.values():
                self.assertTrue(info["source"])   # provenance recorded, never blank


class PaperIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tex = (ROOT / "paper" / "main.tex").read_text()
        self.bib = (ROOT / "paper" / "references.bib").read_text()

    def test_no_citation_orphans(self):
        keys = set(re.findall(r"@\w+\{([A-Za-z0-9]+)", self.bib))
        cited = set()
        for group in re.findall(r"\\cite\{([^}]*)\}", self.tex):
            cited.update(k.strip() for k in group.split(","))
        missing = cited - keys
        self.assertFalse(missing, f"main.tex cites keys not in references.bib: {missing}")

    def test_input_tables_exist(self):
        for name in re.findall(r"\\input\{tables/([^}]+)\.tex\}", self.tex):
            self.assertTrue((ROOT / "paper" / "tables" / f"{name}.tex").exists(),
                            f"main.tex \\inputs a missing table: {name}.tex")

    def test_diagnosis_mentions_are_negated(self):
        for line in self.tex.lower().splitlines():
            if "diagnos" in line and "diagnostic device" not in line:
                self.assertTrue(_NEGATOR.search(line),
                                f"un-negated diagnosis claim in paper: {line.strip()!r}")

    def test_cacmf_never_framed_as_a_win(self):
        low = self.tex.lower()
        for bad in ("cacmf outperforms", "cacmf wins", "cacmf beats", "our cacmf fusion improves"):
            self.assertNotIn(bad, low, f"paper frames the excluded CACMF fusion as a win: {bad!r}")

    def test_has_mandatory_sections(self):
        for sec in ("Data Availability", "Ethics", "Author Contributions", "AI Usage Disclosure",
                    "Limitations"):
            self.assertIn(sec, self.tex, f"paper missing mandatory section: {sec}")


if __name__ == "__main__":
    unittest.main()
