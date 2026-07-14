"""Tests for the dvxr command-line toolkit (dvxr.cli).

Parser wiring is exercised with no data; the fit->predict->report round trip is gated on the
WESAD cohort (band-power, no LaBraM) so it runs offline/CPU in seconds.
"""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class CliParserTest(unittest.TestCase):
    def test_subcommands_wire_to_funcs(self):
        from dvxr.cli import build_parser
        p = build_parser()
        for cmd in ("fit", "predict", "report", "demo"):
            ns = p.parse_args([cmd] if cmd != "fit" else ["fit", "--task", "wesad_stress",
                                                          "--out", "x"])
            self.assertTrue(hasattr(ns, "func"))

    def test_predict_requires_known_task(self):
        from dvxr.cli import build_parser
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["predict", "--task", "not_a_task"])


def _wesad():
    return Path("data/real/WESAD").exists()


@unittest.skipUnless(_wesad(), "WESAD real data required")
class CliRoundtripTest(unittest.TestCase):
    def test_fit_then_predict_then_report(self):
        from dvxr.cli import main
        with tempfile.TemporaryDirectory() as d:
            scr = Path(d) / "scr"
            self.assertEqual(main(["fit", "--task", "wesad_stress", "--out", str(scr),
                                    "--repeats", "2", "--folds", "5"]), 0)
            self.assertTrue((scr / "manifest.json").exists())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["predict", "--screener", str(scr), "--no-narrative"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("DVXR Screen", out)
            self.assertIn("held-out AUROC", out)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["report", "--screener", str(scr)])
            self.assertEqual(rc, 0)
            self.assertIn("held-out AUROC", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
