"""Tests for POST /v1/research/predict (dvxr.serve.research_predict + the API route).

Exercised via Starlette's TestClient (like tests/test_api.py). These pass WITHOUT any committed
artifact — the handler degrades to the labelled deterministic simulation fallback — and stay green
when real artifacts are trained. They also assert the honesty invariants the audit enforces: the
diabetes outcome is never validated, abstains on missing metabolic input, and every response carries
the not-a-diagnosis disclaimer + a model version.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from starlette.testclient import TestClient
    from dvxr.serve.api import create_app, DISCLAIMER
    _HAVE_STARLETTE = True
except Exception:  # pragma: no cover - optional api extra
    _HAVE_STARLETTE = False

SAMPLE_BODY = {
    "features": {
        "hba1c": 6.8, "fasting_glucose": 130, "bmi": 31.0,
        "cgm_mean": 155, "cgm_std": 42, "time_above_range": 0.45,
        "heart_rate": 92, "hrv_rmssd": 22, "eda": 9.0, "resp_rate": 20, "skin_temp": 32.5,
        "eeg_delta": 0.6, "eeg_theta": 0.7, "eeg_alpha": 0.35, "eeg_beta": 0.55,
        "eeg_gamma": 0.3, "frontal_alpha_asymmetry": -0.8,
        "age": 58, "phq9": 14, "gad7": 12, "sleep_hours": 5.0,
    },
    "outcome": "diabetes_status",
    "horizons_minutes": [30, 60],
}

_TARGETS = {"stress", "anxiety", "depression", "cognitive_workload", "glucose_instability"}


@unittest.skipUnless(_HAVE_STARLETTE, "starlette not installed (api extra)")
class ResearchPredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def _post(self, body):
        return self.client.post("/v1/research/predict", json=body)

    # (a) full body → 200 with all target_predictions + selected_outcome + contributions + disclaimer
    def test_full_body_returns_all_sections(self):
        r = self._post(SAMPLE_BODY)
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertEqual(set(b["target_predictions"].keys()), _TARGETS)
        for t, tp in b["target_predictions"].items():
            self.assertIsNotNone(tp["probability"], f"{t} should have a probability")
            self.assertIn("model_version", tp)
            self.assertIn("evidence_status", tp)
        sel = b["selected_outcome"]
        self.assertEqual(sel["name"], "diabetes_status")
        self.assertIsNotNone(sel["probability"])
        self.assertFalse(sel["validated_for_clinical_use"])
        self.assertIn(sel["evidence_status"], {"experimental", "simulation"})
        self.assertTrue(b["contributions"], "expected signed contributions")
        for c in b["contributions"]:
            self.assertEqual(set(c), {"factor", "signed_contribution", "direction", "method"})
            self.assertEqual(c["method"], "linear")
        self.assertIn("30_minutes", b["forecast"])
        self.assertIn("60_minutes", b["forecast"])
        self.assertEqual(b["disclaimer"], DISCLAIMER)

    # (b) changing a glucose value changes the diabetes/selected result (not hardcoded)
    def test_glucose_change_moves_diabetes_outcome(self):
        low = dict(SAMPLE_BODY, features=dict(SAMPLE_BODY["features"],
                   hba1c=5.2, fasting_glucose=88, cgm_mean=95, cgm_std=12, time_above_range=0.05))
        high = dict(SAMPLE_BODY, features=dict(SAMPLE_BODY["features"],
                    hba1c=9.5, fasting_glucose=190, cgm_mean=210, cgm_std=70, time_above_range=0.8))
        p_low = self._post(low).json()["selected_outcome"]["probability"]
        p_high = self._post(high).json()["selected_outcome"]["probability"]
        self.assertNotAlmostEqual(p_low, p_high, msg="diabetes outcome is insensitive to glucose inputs")
        self.assertGreater(p_high, p_low, "worse glycemic inputs should not lower diabetes risk")

    # (c) changing an EEG value moves depression/anxiety/workload but NOT stress or glucose_instability
    def test_eeg_change_is_modality_scoped(self):
        base = self._post(SAMPLE_BODY).json()["target_predictions"]
        shifted = dict(SAMPLE_BODY, features=dict(SAMPLE_BODY["features"],
                       eeg_alpha=1.8, eeg_beta=0.05, eeg_theta=1.5, frontal_alpha_asymmetry=1.5))
        after = self._post(shifted).json()["target_predictions"]
        for t in ("depression", "anxiety", "cognitive_workload"):
            self.assertNotAlmostEqual(base[t]["probability"], after[t]["probability"],
                                      msg=f"{t} should react to neural inputs")
        for t in ("stress", "glucose_instability"):
            self.assertAlmostEqual(base[t]["probability"], after[t]["probability"], places=6,
                                   msg=f"{t} must NOT react to neural inputs (modality leak)")

    # (d) omitting all metabolic inputs → diabetes abstains (probability null), no default overwrite
    def test_missing_metabolic_abstains(self):
        feats = {k: v for k, v in SAMPLE_BODY["features"].items()
                 if k not in {"hba1c", "fasting_glucose", "bmi", "cgm_mean", "cgm_std", "time_above_range"}}
        r = self._post({"features": feats, "outcome": "diabetes_status"})
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertEqual(b["status"], "abstained")
        sel = b["selected_outcome"]
        self.assertIsNone(sel["probability"], "diabetes must abstain with no metabolic input")
        self.assertFalse(sel["validated_for_clinical_use"])
        self.assertIn("reason_codes", b)
        self.assertIn("missing_or_stale_data", b)

    def test_missing_metabolic_abstention_survives_glucose_instability_view(self):
        # glucose_instability is metabolic too → also abstains without metabolic input
        feats = {k: v for k, v in SAMPLE_BODY["features"].items()
                 if k not in {"hba1c", "fasting_glucose", "bmi", "cgm_mean", "cgm_std", "time_above_range"}}
        b = self._post({"features": feats, "outcome": "glucose_instability"}).json()
        self.assertIsNone(b["selected_outcome"]["probability"])

    # (e) every response carries DISCLAIMER + a model_version
    def test_every_response_has_disclaimer_and_model_version(self):
        for body in (SAMPLE_BODY, {"features": {"heart_rate": 88}}, {"features": {}}):
            b = self._post(body).json()
            self.assertEqual(b["disclaimer"], DISCLAIMER)
            self.assertIn("model_version", b["selected_outcome"])

    def test_validation_failure_is_400_with_disclaimer(self):
        r = self._post({"features": {"heart_rate": "not-a-number"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("disclaimer", r.json())
        r2 = self._post({"features": SAMPLE_BODY["features"], "outcome": "cancer"})
        self.assertEqual(r2.status_code, 400)

    def test_out_of_range_value_is_clamped_and_warned(self):
        b = self._post({"features": {"heart_rate": 9999}}).json()
        self.assertTrue(any("out of range" in w for w in b["input_quality"]["warnings"]))

    # (f) no route trains during the request
    def test_request_does_not_train(self):
        import dvxr.serve.research_predict as rp

        called = {"fit": False}
        real_fit = rp.load_research_models

        # if any sklearn .fit were invoked in-request it would be a training call; assert the loader
        # only READS artifacts (or the static simulation fallback) — never fits.
        import sklearn.linear_model as lm
        orig = lm.LogisticRegression.fit

        def _tripwire(self, *a, **k):
            called["fit"] = True
            return orig(self, *a, **k)

        lm.LogisticRegression.fit = _tripwire
        try:
            r = self._post(SAMPLE_BODY)
        finally:
            lm.LogisticRegression.fit = orig
        self.assertEqual(r.status_code, 200)
        self.assertFalse(called["fit"], "the request path must never fit a model")


@unittest.skipUnless(_HAVE_STARLETTE, "starlette not installed (api extra)")
class ResearchPredictArtifactTest(unittest.TestCase):
    """The committed-artifact path: train to a temp dir, point the service at it, and confirm the
    endpoint loads real coefficients (evidence_status flips to experimental for real cohorts, or stays
    simulation for the fixture) — still never validated, still abstaining correctly."""

    def test_trained_artifact_loads_and_predicts(self):
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "_train_research_meta", Path(__file__).resolve().parents[1] / "scripts" / "train_research_meta.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            sys.argv = ["train_research_meta.py", "--synthetic", "--out", d]
            self.assertEqual(mod.main(), 0)
            self.assertTrue((Path(d) / "models.json").exists())
            self.assertTrue((Path(d) / "manifest.json").exists())
            import dvxr.serve.research_predict as rp
            rp._CACHE.clear()
            base, meta, prov = rp.load_research_models(d)
            self.assertEqual(prov, "committed")
            out = rp.run_research_prediction(SAMPLE_BODY, screener_root=d)
            self.assertFalse(out["selected_outcome"]["validated_for_clinical_use"])
            self.assertIsNotNone(out["selected_outcome"]["probability"])
        rp._CACHE.clear()


if __name__ == "__main__":
    unittest.main()
