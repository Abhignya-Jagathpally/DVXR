"""Tests for the live single-subject pipeline engine (dvxr.serve.live).

Fast path: the orchestration (stages, per-window trace, timings, validated flag) on a fake
encoder-free band-power task + a synthetic screener — no data, no LaBraM. Gated integration: the
LaBraM live single-subject embed reproduces the cohort score EXACTLY (live ≡ cohort).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fake_bandpower_task():
    """A minimal BenchTask with band-power features for 2 subjects (no EEG/LaBraM)."""
    from dvxr.bench.tasks import BenchTask
    n = 40
    rng = np.random.default_rng(0)
    subs = np.array(["A"] * 20 + ["B"] * 20)
    feats = {"eda": rng.normal(0, 1, (n, 6))}
    task = BenchTask(name="wesad_stress", kind="classification", features=feats,
                     feature_names={"eda": [f"eda_c_{i}" for i in range(6)]},
                     y=(subs == "B").astype(int), subject_ids=subs, metric="1-AUROC",
                     baseline_hint="majority", raw_windows=None,
                     extra={"_representation": "bandpower_concat"})
    return task


def _synthetic_screener(dim=6):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from dvxr.calibration import fit_platt_calibrator
    from dvxr.serve.screener import Screener
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    emb = rng.normal(0, 1, (n, dim)) + y[:, None] * 1.4
    sc = StandardScaler().fit(emb)
    clf = LogisticRegression(max_iter=500).fit(sc.transform(emb), y)
    p = clf.predict_proba(sc.transform(emb))[:, 1]
    cal = fit_platt_calibrator(p, y)
    return Screener(task="wesad_stress", representation="bandpower_concat", scaler=sc, head=clf,
                    calibrator=cal, conformal=0.2,
                    heldout={"metric": "AUROC", "auroc": 0.9, "auroc_ci": [0.85, 0.95]},
                    meta={"label": "test stress", "encoder": "band-power", "caveat": "test only"})


class LiveOrchestrationTest(unittest.TestCase):
    def test_run_screening_live_structure(self):
        from dvxr.serve.live import run_screening_live
        task = _fake_bandpower_task()
        scr = _synthetic_screener()
        stages = []
        out = run_screening_live(scr, task, "B", on_stage=lambda k, m: stages.append(k))
        # every declared stage fired, in order
        for s in ("raw", "embed", "calibrate", "score", "explain", "done"):
            self.assertIn(s, stages)
        self.assertEqual(out["source"], "cohort")
        self.assertTrue(out["validated"])
        self.assertEqual(len(out["window_probs"]), 20)          # subject B has 20 windows
        self.assertTrue(all(0 <= p <= 1 for p in out["window_probs"]))
        self.assertIn(out["result"]["risk_band"], {"low", "watch", "elevated", "high"})
        self.assertIn("total", out["stage_timings"])
        self.assertTrue(out["drivers"])

    def test_unknown_subject_raises(self):
        from dvxr.serve.live import embed_subject_live
        with self.assertRaises(ValueError):
            embed_subject_live(_fake_bandpower_task(), "ZZ")


class BuildTaskFromEventsTest(unittest.TestCase):
    def test_stub_label_yields_windows(self):
        """Uploaded events with empty labels still window (a placeholder label is stamped)."""
        from dvxr.serve.live import build_task_from_events
        import pandas as pd
        # 30s of 2-channel EEG at 64 Hz, no labels
        rate = 64.0
        n = int(30 * rate)
        t0 = pd.Timestamp("2026-01-01T00:00:00Z")
        rng = np.random.default_rng(2)
        rows = []
        for ch in ("Fp1", "Fp2"):
            rows.append(pd.DataFrame({
                "subject_id": "up", "session_id": "s", "device": "x", "source_system": "up",
                "timestamp_utc": t0 + pd.to_timedelta(np.arange(n) / rate, unit="s"),
                "modality": "eeg", "channel": ch, "value": rng.normal(0, 10, n), "unit": "uV",
                "sampling_rate_hz": rate, "quality_flag": "ok", "label_name": "", "label_value": ""}))
        events = pd.concat(rows, ignore_index=True)
        task, sid = build_task_from_events(events, task_name="mumtaz_depression", window_seconds=8)
        self.assertEqual(sid, "up")
        self.assertGreaterEqual(len(task.subject_ids), 3)       # ~3 windows in 30s @ 8s
        self.assertIn("eeg", task.modalities)
        self.assertEqual(task.extra["window_seconds"], 8)


def _labram_ready():
    try:
        from dvxr.bench.labram_bench import _weights_reachable
        return _weights_reachable() and Path("data/real/mumtaz_mdd").exists()
    except Exception:
        return False


@unittest.skipUnless(_labram_ready(), "LaBraM weights + Mumtaz cohort required")
class LiveEqualsCohortTest(unittest.TestCase):
    def test_live_single_subject_reproduces_cohort_score(self):
        from dvxr.serve.screener import Screener, embed_cohort, fit_screener
        from dvxr.serve.live import run_screening_live, get_encoder
        d = Path("outputs/product/screeners/mumtaz_depression")
        scr = Screener.load(d) if (d / "manifest.json").exists() else fit_screener("mumtaz_depression")
        emb_all, y, subjects, task = embed_cohort("mumtaz_depression", scr.representation)
        task.name = "mumtaz_depression"
        task.extra["_representation"] = scr.representation
        enc = get_encoder()
        for sid in ("MDD_S9", "H_S9"):
            mask = subjects == sid
            ref = scr.score_subject(emb_all[mask])
            out = run_screening_live(scr, task, sid, encoder=enc)
            self.assertAlmostEqual(ref["probability"], out["result"]["probability"], places=3)
            self.assertEqual(ref["risk_band"], out["result"]["risk_band"])


if __name__ == "__main__":
    unittest.main()
