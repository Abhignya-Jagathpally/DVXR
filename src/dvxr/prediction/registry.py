"""dvxr.prediction.registry — resolve the ACTIVE committed model artifact for a report type (P0-1).

The Generate API must never train during a request, and it must never silently abstain merely because
no predictor was hand-injected. This module is the missing link: given a ``report_type`` and a model
registry, it **loads** the committed artifact the registry marks active and returns a ready
``RiskPredictionService`` — or a safe abstainer, fail-closed, when there is nothing valid to load.

Routing (spec §2, §8.7):

* ``cgm_glucose_risk`` / ``glucose_risk`` → the active **CGM excursion** artifact. If a **CGM forecast**
  artifact is also active, the two are composed so the report carries both excursion probabilities and a
  calibrated continuous forecast.
* ``cgm_glucose_forecast`` → the active **CGM forecast** artifact.
* ``stress_glucose_risk`` (or any other multi-modal/fused type) → **AbstainingPredictionService**. The
  fused product requires synchronized same-subject EEG+wearable+CGM data that does not exist; there can
  be no committed fused artifact, so it abstains by construction (never a fabricated model).

Fail-closed everywhere: a missing registry, a missing/renamed artifact, or a sha256 mismatch yields an
abstainer with an honest reason — never an exception into the request path, never an on-the-fly fit.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from dvxr.prediction.forecast_service import CgmOnlyGlucoseForecastService
from dvxr.prediction.service import (
    AbstainingPredictionService,
    CgmOnlyExcursionService,
    PredictionBundle,
    PredictionInputs,
)

#: report types served by a committed CGM excursion artifact.
CGM_RISK_TYPES = frozenset({"cgm_glucose_risk", "glucose_risk"})
#: report type served by a committed CGM forecast artifact.
CGM_FORECAST_TYPE = "cgm_glucose_forecast"
#: registry names under which the offline builder registers each committed artifact.
RISK_REGISTRY_NAME = "cgm_glucose_risk"
FORECAST_REGISTRY_NAME = "cgm_glucose_forecast"

_FUSED_ABSTAIN_REASON = (
    "A reliable fused stress-glucose prediction cannot be produced: it requires synchronized "
    "same-subject EEG+wearable+CGM data, which does not exist in this deployment. No fused model "
    "artifact can be committed, so the service abstains.")


class _RiskWithForecast:
    """Composes the committed excursion classifier with the committed forecaster so a
    ``cgm_glucose_risk`` report carries BOTH per-horizon excursion probabilities and a calibrated
    continuous glucose forecast. The forecast is attached only when the risk service itself produced a
    number (an abstention stays an abstention); if the forecaster abstains, the excursion result stands
    unchanged."""

    def __init__(self, risk: CgmOnlyExcursionService, forecast: Optional[CgmOnlyGlucoseForecastService]):
        self._risk = risk
        self._forecast = forecast
        self.modality_scope = risk.modality_scope
        self.model_version = risk.model_version

    def predict(self, inputs: PredictionInputs) -> PredictionBundle:
        bundle = self._risk.predict(inputs)
        if bundle.abstained or self._forecast is None:
            return bundle
        fb = self._forecast.predict(inputs)
        if fb.abstained or not fb.forecast:
            return bundle
        return dataclasses.replace(
            bundle, forecast=fb.forecast, forecast_model_version=fb.forecast_model_version,
            forecast_interval_version=fb.forecast_interval_version,
            forecast_coverage_target=fb.forecast_coverage_target)


def _load_artifact(model_registry, artifact_root, name, cls):
    """Load the active artifact registered under ``name`` via ``cls.load``, verifying the registry's
    recorded sha256 against the artifact. Returns the loaded service, or None on ANY failure (nothing
    active, path missing, hash mismatch) — the caller turns None into a fail-closed abstention. Never
    raises into the request path, never fits."""
    if model_registry is None or artifact_root is None:
        return None
    try:
        row = model_registry.active(name)
    except Exception:
        return None
    if not row:
        return None
    meta = row.get("meta") or {}
    rel = meta.get("artifact_path")
    if not rel:
        return None
    path = Path(artifact_root) / rel
    if not path.exists():
        return None
    try:
        service = cls.load(path)
    except Exception:
        return None
    # cross-check the registry's recorded sha against the artifact's own manifest sha (drift/tamper)
    recorded = meta.get("artifact_sha256")
    if recorded:
        import json
        try:
            manifest_sha = json.loads((path / "manifest.json").read_text()).get("artifact_sha256")
        except Exception:
            return None
        if manifest_sha and manifest_sha != recorded:
            return None
    return service


def resolve_predictor(report_type: str, *, model_registry=None, artifact_root=None):
    """Return the active committed predictor for ``report_type`` (P0-1), or a fail-closed abstainer.

    Never trains: it only loads a committed artifact. Fused/multi-modal report types always abstain
    (no synchronized data ⇒ no committed fused artifact)."""
    if report_type == CGM_FORECAST_TYPE:
        svc = _load_artifact(model_registry, artifact_root, FORECAST_REGISTRY_NAME,
                             CgmOnlyGlucoseForecastService)
        return svc if svc is not None else AbstainingPredictionService(
            "No committed CGM forecast artifact is active in the registry; the API does not train at "
            "request time, so it abstains.")
    if report_type in CGM_RISK_TYPES:
        risk = _load_artifact(model_registry, artifact_root, RISK_REGISTRY_NAME, CgmOnlyExcursionService)
        if risk is None:
            return AbstainingPredictionService(
                "No committed CGM excursion artifact is active in the registry; the API does not train "
                "at request time, so it abstains.")
        forecast = _load_artifact(model_registry, artifact_root, FORECAST_REGISTRY_NAME,
                                  CgmOnlyGlucoseForecastService)
        return _RiskWithForecast(risk, forecast)
    # fused / multi-modal / unknown → safe abstain (guardrail: no fabricated fused model)
    return AbstainingPredictionService(_FUSED_ABSTAIN_REASON)
