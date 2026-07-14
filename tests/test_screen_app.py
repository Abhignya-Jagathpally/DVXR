"""Tests for the live Streamlit app (scripts/screen_app.py) — the parts testable without Streamlit.

The app imports streamlit *inside* main(), so the module imports cleanly without it and main() prints
an install hint rather than crashing. The pure-Python helpers (meter markup, task table) are checked
directly.
"""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ScreenAppImportGuardTest(unittest.TestCase):
    def test_imports_without_streamlit(self):
        import scripts.screen_app as app          # must not require streamlit at import
        self.assertTrue(hasattr(app, "main"))

    @unittest.skipIf(importlib.util.find_spec("streamlit") is not None,
                     "streamlit installed; skip the absent-dependency hint test")
    def test_main_prints_install_hint_without_streamlit(self):
        import scripts.screen_app as app
        buf = io.StringIO()
        with redirect_stdout(buf):
            app.main()
        out = buf.getvalue().lower()
        self.assertIn("streamlit", out)
        self.assertIn("pip install", out)

    def test_meter_html_self_contained(self):
        from scripts.screen_app import _meter_html
        h = _meter_html(0.8, "high")
        self.assertIn("<div", h)
        self.assertNotIn("http://", h)
        self.assertNotIn("https://", h)


if __name__ == "__main__":
    unittest.main()
