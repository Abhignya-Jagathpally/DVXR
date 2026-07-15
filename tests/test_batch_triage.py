"""Tests for cohort triage (dvxr.serve.batch).

Fast: the ranking + self-contained HTML on a synthetic screener + tiny cohort (no data). Gated:
depression triage surfaces MDD cases above healthy controls (the clinically meaningful case).
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


class _CohortScreener:
    """Minimal screener stub: probability = mean of a subject's embedding column 0 (clipped)."""
    task = "toy"
    representation = "bandpower_concat"
    heldout = {"auroc": 0.9, "auroc_subject": 0.95}
    meta = {"label": "Toy screen"}

    def score_subject(self, emb):
        p = float(np.clip(np.mean(emb[:, 0]), 0, 1))
        return {"probability": round(p, 4), "risk_band":
                ("high" if p >= 0.75 else "elevated" if p >= 0.5 else "watch" if p >= 0.25 else "low"),
                "interval": [max(0, round(p - 0.1, 4)), min(1, round(p + 0.1, 4))], "n_windows": len(emb)}


def _patch_embed_cohort(monkeypatch_target):
    """Provide a deterministic cohort: 6 subjects, 3 high-risk (label 1), 3 low (label 0)."""
    import dvxr.serve.batch as batch

    def fake_embed_cohort(task_name, representation=None):
        subs = np.repeat([f"s{i}" for i in range(6)], 5)
        base = np.repeat([0.85, 0.80, 0.78, 0.15, 0.12, 0.10], 5)   # per-subject risk
        emb = np.column_stack([base, np.zeros(len(base))])
        y = (base > 0.5).astype(int)
        return emb, y, subs, None
    batch.__dict__.setdefault("_orig", None)
    return fake_embed_cohort


class TriageMechanicsTest(unittest.TestCase):
    def setUp(self):
        import dvxr.serve.screener as screener_mod
        self._orig = screener_mod.embed_cohort
        screener_mod.embed_cohort = _patch_embed_cohort(None)

    def tearDown(self):
        import dvxr.serve.screener as screener_mod
        screener_mod.embed_cohort = self._orig

    def test_ranks_high_risk_first_and_writes_artifacts(self):
        from dvxr.serve.batch import triage_cohort, render_triage_html, write_triage
        s = _CohortScreener()
        df = triage_cohort(s, "toy")
        self.assertEqual(len(df), 6)
        # highest-risk subject ranked #1, and it is a case
        self.assertEqual(df.iloc[0]["rank"], 1)
        self.assertGreaterEqual(df.iloc[0]["probability"], df.iloc[-1]["probability"])
        self.assertEqual(df.iloc[0]["cohort_label"], 1)
        # top half are the cases
        self.assertTrue((df.head(3)["cohort_label"] == 1).all())
        html = render_triage_html(df, s, "toy")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)              # self-contained
        self.assertIn("not a diagnosis", " ".join(html.lower().split()))  # ws-robust
        with tempfile.TemporaryDirectory() as d:
            write_triage(s, d, "toy")
            self.assertTrue((Path(d) / "triage.csv").exists())
            self.assertTrue((Path(d) / "triage.html").exists())


def _labram_ready():
    try:
        from dvxr.bench.labram_bench import _weights_reachable
        return _weights_reachable() and Path("data/real/mumtaz_mdd").exists() and \
            (Path("outputs/product/screeners/mumtaz_depression/manifest.json").exists())
    except Exception:
        return False


@unittest.skipUnless(_labram_ready(), "LaBraM weights + Mumtaz cohort + cached screener required")
class TriageDepressionTest(unittest.TestCase):
    def test_cases_outrank_controls(self):
        from dvxr.serve.screener import Screener
        from dvxr.serve.batch import triage_cohort
        s = Screener.load("outputs/product/screeners/mumtaz_depression")
        df = triage_cohort(s, "mumtaz_depression")
        cases = df[df.cohort_label == 1]["probability"].mean()
        controls = df[df.cohort_label == 0]["probability"].mean()
        self.assertGreater(cases, controls + 0.1)       # clear separation
        self.assertGreater(df.head(len(df) // 2)["cohort_label"].mean(), 0.7)  # top-half mostly cases


if __name__ == "__main__":
    unittest.main()
