"""Wrap: the FastAPI wrapper (dvxr.serve.asgi) deploys the Starlette product app and, with a committed
artifact provisioned, serves a REAL CGM prediction over HTTP — while the fused report abstains."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.prediction.forecast_service import CgmOnlyGlucoseForecastService  # noqa: E402
from dvxr.prediction.registry import FORECAST_REGISTRY_NAME, RISK_REGISTRY_NAME  # noqa: E402
from dvxr.prediction.service import CgmOnlyExcursionService  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels  # noqa: E402


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _cohort(n_subjects=14, n=200):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        base = 165.0 if i % 2 == 0 else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        vals = np.clip(base + rs.normal(0, 10, n) + (28 * np.sin(np.arange(n) / 7.0 + i)), 55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": vals}))
    return pd.concat(frames, ignore_index=True)


@unittest.skipUnless(_have("fastapi") and _have("starlette.testclient"),
                     "fastapi / starlette test client not installed (deploy extra)")
class DeployWrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = str(Path(self.tmp) / "dvxr.db")
        self.root = str(Path(self.tmp) / "artifacts")
        thr = ExcursionThresholds(history_minutes=120)
        cgm = _cohort()
        anchors = sorted({t for _, g in cgm.groupby("subject_id")
                          for t in pd.to_datetime(g["timestamp"]).iloc[8::4]})
        ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id",
                                    label_definition="incident")
        risk = CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr, seed=7)
        forecast = CgmOnlyGlucoseForecastService.fit(cgm, thresholds=thr, anchors=anchors, seed=7)
        reg = open_local_stores(self.db)
        for name, svc, sub in ((RISK_REGISTRY_NAME, risk, "excursion"),
                               (FORECAST_REGISTRY_NAME, forecast, "forecast")):
            p = Path(self.root) / sub
            svc.save(p)
            sha = json.loads((p / "manifest.json").read_text())["artifact_sha256"]
            reg.models.register(name, svc.model_version,
                                {"artifact_path": sub, "artifact_sha256": sha,
                                 "model_version": svc.model_version}, active=True)
        # seed recent CGM events for patient P1 under the dev tenant (unsafe_dev principal)
        sub = cgm[cgm.subject_id == "s0"].copy().iloc[:130]
        self.cutoff = pd.to_datetime(sub["timestamp"]).iloc[-1].isoformat()
        reg.events.append_events([
            {"event_id": f"e{i}", "patient_id": "P1", "tenant_id": "dev", "modality": "cgm",
             "observed_at_utc": pd.Timestamp(r["timestamp"]).isoformat(),
             "value": float(r["glucose"]), "quality_score": 0.9}
            for i, r in sub.iterrows()])
        os.environ.update(DVXR_DB_PATH=self.db, DVXR_ARTIFACT_ROOT=self.root,
                          DVXR_UNSAFE_DEV="1", DVXR_REQUIRE_CONSENT="0")

    def _client(self):
        from fastapi.testclient import TestClient
        from dvxr.serve.asgi import build_app
        return TestClient(build_app())

    def test_health_carries_disclaimer(self):
        r = self._client().get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("not a diagnosis", json.dumps(r.json()).lower())

    def test_root_serves_the_user_facing_spa(self):
        # The wrapper now serves the full product web app at "/" (superseding the old JSON landing);
        # the machine-readable summary moved to /ui/config.
        r = self._client().get("/")
        self.assertEqual(r.status_code, 200)                      # not a 404 at the bare URL
        self.assertIn("text/html", r.headers.get("content-type", ""))
        self.assertIn("DVXR NeuroGlycemic Sentinel", r.text)
        self.assertIn("Generate a risk review", r.text)           # the risk-review workspace is present

    def test_ui_config_reports_fused_abstains(self):
        cfg = self._client().get("/ui/config")
        self.assertEqual(cfg.status_code, 200)
        self.assertEqual(cfg.json()["fused_report_status"], "abstains_until_synchronized_artifact")

    def test_cgm_report_serves_a_real_prediction(self):
        c = self._client()
        r = c.post("/v1/risk-reports", json={"patient_id": "P1", "report_type": "cgm_glucose_risk",
                                             "data_cutoff_at": self.cutoff,
                                             "prediction_horizons_minutes": [30, 60]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "completed")           # a real number, loaded from the artifact
        self.assertIsNotNone(body["prediction"]["risk"])
        self.assertIsNotNone(body["prediction"]["forecast"])

    def test_fused_report_abstains_through_the_wrapper(self):
        c = self._client()
        r = c.post("/v1/risk-reports", json={"patient_id": "P1", "report_type": "stress_glucose_risk"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "abstained")       # no fused artifact can exist

    def tearDown(self):
        for k in ("DVXR_DB_PATH", "DVXR_ARTIFACT_ROOT", "DVXR_UNSAFE_DEV", "DVXR_REQUIRE_CONSENT"):
            os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
