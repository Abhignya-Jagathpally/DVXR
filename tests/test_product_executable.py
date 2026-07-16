"""PR34 / P0-1 + P0-2: the product API is genuinely executable.

The Generate API must LOAD a committed CGM artifact from the model registry and return a real
prediction (excursion risk + calibrated continuous forecast) WITHOUT training during the request, and it
must resolve an empty cutoff to a concrete instant. The fused report has no committed artifact and
abstains by construction. These tests pin exactly those properties.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dvxr.contracts import GenerateRequest  # noqa: E402
from dvxr.prediction.forecast_service import CgmOnlyGlucoseForecastService  # noqa: E402
from dvxr.prediction.registry import (  # noqa: E402
    FORECAST_REGISTRY_NAME,
    RISK_REGISTRY_NAME,
    resolve_predictor,
)
from dvxr.prediction.service import CgmOnlyExcursionService, PredictionInputs  # noqa: E402
from dvxr.serve.orchestrate import generate_risk_report  # noqa: E402
from dvxr.storage import open_local_stores  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels, history_slice  # noqa: E402


def _synthetic_cgm(n_subjects=14, n=200):
    rs = np.random.RandomState(0)
    frames = []
    for i in range(n_subjects):
        high = i % 2 == 0
        base = 165.0 if high else 110.0
        ts = pd.date_range("2020-01-01", periods=n, freq="15min")
        vals = np.clip(base + rs.normal(0, 10, n) + (28 * np.sin(np.arange(n) / 7.0 + i)), 55, 320)
        frames.append(pd.DataFrame({"subject_id": f"s{i}", "timestamp": ts, "glucose": vals}))
    return pd.concat(frames, ignore_index=True)


_FIT_CACHE = {}


def _fit_both(thr):
    """Fit both services once per threshold config and memoize — the fitted services are immutable and
    read-only across tests, so re-fitting per test method would only bloat the suite. Tests that need a
    fresh registry still create their own in-memory registry (cheap)."""
    key = (thr.history_minutes, thr.version)
    if key not in _FIT_CACHE:
        cgm = _synthetic_cgm()
        # thin the anchors (every 4th sample) so the fit is fast without changing what it verifies
        anchors = sorted({t for _, g in cgm.groupby("subject_id")
                          for t in pd.to_datetime(g["timestamp"]).iloc[8::4]})
        ex = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
        risk = CgmOnlyExcursionService.fit(cgm, ex, thresholds=thr, seed=7)
        forecast = CgmOnlyGlucoseForecastService.fit(cgm, thresholds=thr, seed=7)
        _FIT_CACHE[key] = (risk, forecast, cgm)
    return _FIT_CACHE[key]


def _register(root, risk, forecast, registry):
    """Save both artifacts under `root` and register them ACTIVE (mirrors scripts/build_cgm_artifact)."""
    rp = Path(root) / "excursion"
    risk.save(rp)
    registry.register(RISK_REGISTRY_NAME, risk.model_version,
                      {"kind": "cgm_excursion", "artifact_path": "excursion",
                       "artifact_sha256": json.loads((rp / "manifest.json").read_text())["artifact_sha256"],
                       "model_version": risk.model_version}, active=True)
    fp = Path(root) / "forecast"
    forecast.save(fp)
    registry.register(FORECAST_REGISTRY_NAME, forecast.model_version,
                      {"kind": "cgm_forecast", "artifact_path": "forecast",
                       "artifact_sha256": json.loads((fp / "manifest.json").read_text())["artifact_sha256"],
                       "model_version": forecast.model_version}, active=True)


def _cgm_events(cgm, patient="P1", subject="s0", n=140):
    sub = cgm[cgm.subject_id == subject].copy().iloc[:n]
    cutoff = pd.to_datetime(sub["timestamp"]).iloc[-1].isoformat()
    events = [{"event_id": f"e{i}", "patient_id": patient, "tenant_id": "default", "modality": "cgm",
               "observed_at_utc": pd.Timestamp(r["timestamp"]).isoformat(), "value": float(r["glucose"]),
               "quality_score": 0.9}
              for i, r in sub.iterrows()]
    return events, cutoff


class ArtifactRoundTripTest(unittest.TestCase):
    def test_risk_and_forecast_services_roundtrip_identically(self):
        thr = ExcursionThresholds(history_minutes=120)
        risk, forecast, cgm = _fit_both(thr)
        anchor = pd.to_datetime(cgm[cgm.subject_id == "s0"]["timestamp"]).iloc[120]
        hist = history_slice(cgm, anchor, thresholds=thr, subject_col="subject_id", subject_id="s0")
        inp = PredictionInputs("cgm_glucose_risk", [30, 60], cgm_history=hist,
                               requested_modalities=["cgm"], cutoff=anchor.isoformat())
        with tempfile.TemporaryDirectory() as d:
            risk.save(Path(d) / "r")
            forecast.save(Path(d) / "f")
            r2 = CgmOnlyExcursionService.load(Path(d) / "r")
            f2 = CgmOnlyGlucoseForecastService.load(Path(d) / "f")
        self.assertEqual(risk.predict(inp).risk, r2.predict(inp).risk)
        self.assertEqual(forecast.predict(inp).forecast, f2.predict(inp).forecast)

    def test_load_rejects_tampered_artifact(self):
        thr = ExcursionThresholds(history_minutes=120)
        risk, _f, _c = _fit_both(thr)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r"
            risk.save(p)
            (p / "model.joblib").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                CgmOnlyExcursionService.load(p)


class ForecastBaselineTest(unittest.TestCase):
    def test_learned_forecaster_is_scored_against_persistence_and_linear(self):
        thr = ExcursionThresholds(history_minutes=120)
        _risk, forecast, _cgm = _fit_both(thr)
        for h in (30, 60):
            rep = forecast.baseline_report.get(h)
            self.assertIsNotNone(rep)
            for k in ("learned_rmse", "persistence_rmse", "linear_rmse",
                      "beats_persistence", "beats_linear"):
                self.assertIn(k, rep)


class RegistryResolverTest(unittest.TestCase):
    def setUp(self):
        self.stores = open_local_stores(":memory:")
        self.pred, self.audit, self.consent, self.registry = self.stores
        self.consent.set_scope("P1", {"purposes": ["research"]})
        self.thr = ExcursionThresholds(history_minutes=120)
        self.risk, self.forecast, self.cgm = _fit_both(self.thr)

    def test_api_loads_committed_artifact_and_serves_risk_plus_forecast_without_training(self):
        events, cutoff = _cgm_events(self.cgm)
        req = GenerateRequest(patient_id="P1", report_type="cgm_glucose_risk", user_role="researcher",
                              data_cutoff_at=cutoff, prediction_horizons_minutes=[30, 60])
        with tempfile.TemporaryDirectory() as d:
            _register(d, self.risk, self.forecast, self.registry)
            # the API must NEVER fit during a request — make any in-request fit blow up
            with mock.patch.object(CgmOnlyExcursionService, "fit",
                                   side_effect=AssertionError("trained during request!")), \
                 mock.patch.object(CgmOnlyGlucoseForecastService, "fit",
                                   side_effect=AssertionError("trained during request!")):
                out = generate_risk_report(
                    req, prediction_store=self.pred, audit_store=self.audit,
                    consent_store=self.consent, events=events,
                    model_registry=self.registry, artifact_root=d)
        self.assertEqual(out["status"], "completed")
        risk = out["prediction"]["risk"]
        self.assertIsNotNone(risk)
        self.assertIn("excursion_30m", risk)
        self.assertIn("excursion_60m", risk)
        fc = out["prediction"]["forecast"]
        self.assertIsNotNone(fc)                                     # forecast rides with the risk report
        for k in ("glucose_30m", "glucose_60m"):
            self.assertIn(k, fc)
            self.assertIn("point", fc[k])
            self.assertIn("lower", fc[k])
            self.assertIn("upper", fc[k])
        self.assertTrue(out["prediction"]["forecast_interval_version"].startswith(
            "empirical-conformal/"))
        self.assertEqual(out["prediction"]["forecast_coverage_target"], 0.9)

    def test_forecast_only_report_type_serves_a_forecast(self):
        events, cutoff = _cgm_events(self.cgm)
        req = GenerateRequest(patient_id="P1", report_type="cgm_glucose_forecast", user_role="researcher",
                              data_cutoff_at=cutoff, prediction_horizons_minutes=[30, 60])
        with tempfile.TemporaryDirectory() as d:
            _register(d, self.risk, self.forecast, self.registry)
            out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                       consent_store=self.consent, events=events,
                                       model_registry=self.registry, artifact_root=d)
        self.assertIsNotNone(out["prediction"]["forecast"])

    def test_fused_report_abstains_through_the_resolver(self):
        events, cutoff = _cgm_events(self.cgm)
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk", user_role="researcher",
                              data_cutoff_at=cutoff, prediction_horizons_minutes=[30, 60])
        with tempfile.TemporaryDirectory() as d:
            _register(d, self.risk, self.forecast, self.registry)
            out = generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                       consent_store=self.consent, events=events,
                                       model_registry=self.registry, artifact_root=d)
        self.assertEqual(out["status"], "abstained")                # no fused artifact can exist
        self.assertIsNone(out["prediction"]["risk"])

    def test_resolver_fails_closed_when_nothing_registered(self):
        # empty registry → abstain, never fit
        svc = resolve_predictor("cgm_glucose_risk", model_registry=self.registry, artifact_root="/nope")
        b = svc.predict(PredictionInputs("cgm_glucose_risk", [30], requested_modalities=["cgm"]))
        self.assertTrue(b.abstained)

    def test_resolver_fails_closed_on_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            _register(d, self.risk, self.forecast, self.registry)
            # corrupt the registry's recorded hash → drift/tamper → fail closed
            self.registry.register(RISK_REGISTRY_NAME, self.risk.model_version,
                                   {"kind": "cgm_excursion", "artifact_path": "excursion",
                                    "artifact_sha256": "deadbeef", "model_version": self.risk.model_version},
                                   active=True)
            svc = resolve_predictor("cgm_glucose_risk", model_registry=self.registry, artifact_root=d)
            events, _ = _cgm_events(self.cgm)
            anchor = pd.to_datetime(self.cgm[self.cgm.subject_id == "s0"]["timestamp"]).iloc[120]
            hist = history_slice(self.cgm, anchor, thresholds=self.thr, subject_col="subject_id",
                                 subject_id="s0")
            b = svc.predict(PredictionInputs("cgm_glucose_risk", [30, 60], cgm_history=hist,
                                             requested_modalities=["cgm"], cutoff=anchor.isoformat()))
        self.assertTrue(b.abstained)


class CutoffResolutionTest(unittest.TestCase):
    def setUp(self):
        self.pred, self.audit, self.consent, _m = open_local_stores(":memory:")
        self.consent.set_scope("P1", {"purposes": ["research"]})

    def _run(self, clock):
        req = GenerateRequest(patient_id="P1", report_type="stress_glucose_risk", user_role="researcher")
        return generate_risk_report(req, prediction_store=self.pred, audit_store=self.audit,
                                    consent_store=self.consent, clock=clock)

    def test_empty_cutoff_is_resolved_to_a_concrete_instant(self):
        fixed = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
        out = self._run(lambda: fixed)
        self.assertEqual(out["prediction"]["data_cutoff_at"], fixed.isoformat())

    def test_same_clock_is_deterministic_and_different_nows_differ(self):
        t1 = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 16, 12, 5, 0, tzinfo=timezone.utc)
        rid_a = self._run(lambda: t1)["request_id"]
        rid_b = self._run(lambda: t1)["request_id"]
        rid_c = self._run(lambda: t2)["request_id"]
        self.assertEqual(rid_a, rid_b)                              # same instant → same fingerprint
        self.assertNotEqual(rid_a, rid_c)                          # a different "now" → different id


if __name__ == "__main__":
    unittest.main()
