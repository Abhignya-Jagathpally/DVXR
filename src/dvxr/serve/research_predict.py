"""dvxr.serve.research_predict — the ``POST /v1/research/predict`` scoring service (iteration 2).

Turns arbitrary user-entered feature values (metabolic / physiological / neural / clinical) into
per-target research ESTIMATES plus a selected diabetes outcome, signed contributions, a glucose
forecast, an input-quality report, and honest abstention. It NEVER trains at request time: it loads the
committed lightweight tabular artifacts written by ``scripts/train_research_meta.py`` (under
``outputs/product/research_models/``); when those are absent it degrades to a clearly-labelled
deterministic ILLUSTRATIVE fallback (``evidence_status == "simulation"``) rather than crash.

Honesty invariants (mirrored by ``tests/test_honesty_audit.py``):
  * the diabetes outcome always carries ``validated_for_clinical_use == False`` and an
    ``experimental``/``simulation`` evidence status — never a headline AUROC, never a diagnosis;
  * the fused glucose forecast follows ``prediction.registry.resolve_predictor`` and abstains without a
    committed CGM artifact — no fabricated fused claim (this is decision-support, never a diagnosis);
  * each target reads ONLY its own modality's inputs, so a neural value can never move a metabolic
    estimate (and vice-versa) — the modality scoping is honest, not incidental.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dvxr.calibration import risk_band
from dvxr.prediction.meta_model import DiabetesMetaModel, LinearHead

DEFAULT_RESEARCH_MODELS_ROOT = Path("outputs/product/research_models")

# --------------------------------------------------------------------------- canonical feature schema
# name -> (modality, reference_mean, reference_scale, valid_low, valid_high)
CANONICAL_FEATURES: Dict[str, Tuple[str, float, float, float, float]] = {
    # metabolic
    "hba1c": ("metabolic", 5.7, 1.2, 3.0, 20.0),
    "fasting_glucose": ("metabolic", 100.0, 30.0, 40.0, 500.0),
    "bmi": ("metabolic", 27.0, 6.0, 10.0, 80.0),
    "cgm_mean": ("metabolic", 120.0, 30.0, 40.0, 400.0),
    "cgm_std": ("metabolic", 25.0, 12.0, 0.0, 200.0),
    "time_above_range": ("metabolic", 0.25, 0.2, 0.0, 1.0),
    # physiological
    "heart_rate": ("physiological", 75.0, 15.0, 30.0, 220.0),
    "hrv_rmssd": ("physiological", 40.0, 20.0, 0.0, 300.0),
    "eda": ("physiological", 5.0, 3.0, 0.0, 100.0),
    "resp_rate": ("physiological", 15.0, 4.0, 4.0, 60.0),
    "skin_temp": ("physiological", 33.0, 2.0, 20.0, 45.0),
    # neural
    "eeg_delta": ("neural", 0.5, 0.3, 0.0, 5.0),
    "eeg_theta": ("neural", 0.4, 0.25, 0.0, 5.0),
    "eeg_alpha": ("neural", 0.5, 0.3, 0.0, 5.0),
    "eeg_beta": ("neural", 0.3, 0.2, 0.0, 5.0),
    "eeg_gamma": ("neural", 0.2, 0.15, 0.0, 5.0),
    "frontal_alpha_asymmetry": ("neural", 0.0, 0.3, -3.0, 3.0),
    # clinical
    "age": ("clinical", 45.0, 15.0, 0.0, 120.0),
    "phq9": ("clinical", 6.0, 5.0, 0.0, 27.0),
    "gad7": ("clinical", 5.0, 4.0, 0.0, 21.0),
    "sleep_hours": ("clinical", 7.0, 1.5, 0.0, 24.0),
}

# Documented request contract (spec / web client) field name -> canonical feature name.
# Lets the endpoint accept the published contract, not only internal canonical names.
_FIELD_ALIASES: Dict[str, str] = {
    # metabolic
    "hba1c_percent": "hba1c", "fasting_glucose_mg_dl": "fasting_glucose",
    "mean_glucose_mg_dl": "cgm_mean", "current_glucose_mg_dl": "cgm_mean",
    "glucose_std_mg_dl": "cgm_std", "time_above_180_percent": "time_above_range",
    # physiological
    "heart_rate_bpm": "heart_rate", "hrv_rmssd_ms": "hrv_rmssd", "eda_microsiemens": "eda",
    "respiration_rate_bpm": "resp_rate", "skin_temperature_c": "skin_temp",
    # neural
    "delta_relative_power": "eeg_delta", "theta_relative_power": "eeg_theta",
    "alpha_relative_power": "eeg_alpha", "beta_relative_power": "eeg_beta",
    # clinical: age / bmi / sleep_hours already canonical
}
# Contract fields expressed in different units than the canonical schema (multiplier into canonical).
_FIELD_SCALE: Dict[str, float] = {"time_above_180_percent": 0.01}  # percent -> fraction

MODALITIES = ("metabolic", "physiological", "neural", "clinical")

# per-target ordered feature list (the ONLY inputs that target reads). A target reads exactly one
# sensor modality (+ optional matching clinical score) so cross-modality leakage is impossible.
TARGET_FEATURES: Dict[str, List[str]] = {
    "stress": ["heart_rate", "hrv_rmssd", "eda", "resp_rate", "skin_temp"],
    "anxiety": ["eeg_beta", "eeg_alpha", "frontal_alpha_asymmetry", "gad7"],
    "depression": ["frontal_alpha_asymmetry", "eeg_alpha", "eeg_theta", "phq9"],
    "cognitive_workload": ["eeg_theta", "eeg_beta", "eeg_alpha", "eeg_delta"],
    "glucose_instability": ["cgm_std", "time_above_range", "cgm_mean", "hba1c", "fasting_glucose"],
}
TARGETS = tuple(TARGET_FEATURES.keys())

# the diabetes meta-model's own metabolic covariates + the base-model probabilities it stacks
META_METABOLIC_FEATURES = ["hba1c", "fasting_glucose", "bmi", "cgm_std", "time_above_range"]
META_PROB_FEATURES = ["prob_stress", "prob_anxiety", "prob_depression", "prob_cognitive_workload"]

SELECTABLE_OUTCOMES = ("diabetes_status", "glucose_instability", "diabetes_complication")

# --------------------------------------------------------------------------- illustrative fallback
# Hand-set, sign-honest coefficients (standardized units) used ONLY when no trained artifact is present.
# Every head built here is flagged evidence_status "simulation" and validated_for_clinical_use False.
_SIM_COEF: Dict[str, Dict[str, float]] = {
    "stress": {"heart_rate": 0.9, "hrv_rmssd": -0.8, "eda": 0.9, "resp_rate": 0.5, "skin_temp": -0.4},
    "anxiety": {"eeg_beta": 0.8, "eeg_alpha": -0.6, "frontal_alpha_asymmetry": -0.7, "gad7": 1.1},
    "depression": {"frontal_alpha_asymmetry": -0.9, "eeg_alpha": 0.5, "eeg_theta": 0.5, "phq9": 1.2},
    "cognitive_workload": {"eeg_theta": 0.9, "eeg_beta": 0.7, "eeg_alpha": -0.7, "eeg_delta": -0.3},
    "glucose_instability": {"cgm_std": 1.1, "time_above_range": 1.0, "cgm_mean": 0.8,
                            "hba1c": 0.9, "fasting_glucose": 0.6},
}
_SIM_META_COEF = {"hba1c": 1.3, "fasting_glucose": 0.9, "bmi": 0.5, "cgm_std": 0.6,
                  "time_above_range": 0.6, "prob_stress": 0.15, "prob_anxiety": 0.1,
                  "prob_depression": 0.1, "prob_cognitive_workload": 0.1}


def _sim_head(features: List[str], coef_map: Dict[str, float], version: str) -> LinearHead:
    mean, scale, coef = [], [], []
    for f in features:
        if f in CANONICAL_FEATURES:
            _, m, s, _, _ = CANONICAL_FEATURES[f]
        else:  # a stacked probability feature (already ~[0,1])
            m, s = 0.5, 0.25
        mean.append(m)
        scale.append(s)
        coef.append(coef_map.get(f, 0.0))
    return LinearHead(features=features, mean=mean, scale=scale, coef=coef, intercept=0.0,
                      model_version=version, evidence_status="simulation", auroc_oof=None)


def simulation_models() -> Tuple[Dict[str, LinearHead], DiabetesMetaModel]:
    """Deterministic illustrative fallback used when no committed artifact exists."""
    base = {t: _sim_head(TARGET_FEATURES[t], _SIM_COEF[t], f"research-{t}/v2-simulation")
            for t in TARGETS}
    meta_features = META_METABOLIC_FEATURES + META_PROB_FEATURES
    meta_head = _sim_head(meta_features, _SIM_META_COEF, "research-diabetes-meta/v2-simulation")
    meta = DiabetesMetaModel(head=meta_head, metabolic_features=META_METABOLIC_FEATURES,
                             prob_features=META_PROB_FEATURES)
    return base, meta


# --------------------------------------------------------------------------- artifact loading (cached)
_CACHE: Dict[str, Tuple[Dict[str, LinearHead], DiabetesMetaModel, str]] = {}


def resolve_models_root(screener_root: str | Path | None) -> Path:
    """Resolve where the committed research artifacts live from whatever the caller passed. The HTTP
    handler passes the SCREENER root (``outputs/product/screeners``); the models live in its sibling
    ``research_models``. A path that already contains ``models.json`` is used as-is (the training
    script / tests point directly at an output dir)."""
    if screener_root is None:
        return DEFAULT_RESEARCH_MODELS_ROOT
    p = Path(screener_root)
    if (p / "models.json").exists():
        return p
    sibling = p.parent / "research_models"
    if (sibling / "models.json").exists():
        return sibling
    return DEFAULT_RESEARCH_MODELS_ROOT


def load_research_models(root: str | Path | None = None
                         ) -> Tuple[Dict[str, LinearHead], DiabetesMetaModel, str]:
    """Load the committed per-target heads + diabetes meta-model, or the simulation fallback.

    Returns ``(base_heads, meta_model, provenance)`` where provenance is ``"committed"`` or
    ``"simulation"``. Never trains, never raises into the request path."""
    root = Path(root) if root is not None else DEFAULT_RESEARCH_MODELS_ROOT
    key = str(root.resolve()) if root else "_sim"
    if key in _CACHE:
        return _CACHE[key]
    models_path = root / "models.json"
    result: Tuple[Dict[str, LinearHead], DiabetesMetaModel, str]
    if models_path.exists():
        try:
            payload = json.loads(models_path.read_text())
            base = {t: LinearHead.from_dict(h) for t, h in payload["targets"].items()}
            meta = DiabetesMetaModel.from_dict(payload["diabetes_meta"])
            # fill any target missing from the artifact with its simulation head (graceful partial)
            sim_base, _sim_meta = simulation_models()
            for t in TARGETS:
                base.setdefault(t, sim_base[t])
            result = (base, meta, "committed")
        except Exception:  # noqa: BLE001 — a corrupt artifact must degrade, not crash
            base, meta = simulation_models()
            result = (base, meta, "simulation")
    else:
        base, meta = simulation_models()
        result = (base, meta, "simulation")
    _CACHE[key] = result
    return result


def artifact_sha256(root: str | Path | None = None) -> Optional[str]:
    root = Path(root) if root is not None else DEFAULT_RESEARCH_MODELS_ROOT
    p = root / "models.json"
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- request / response models
class ValidationError(ValueError):
    """A user-input validation failure → HTTP 400 with a JSON error + disclaimer."""


@dataclass
class FeaturePredictionRequest:
    """Validated request. ``features`` are canonical named values; ``outcome`` selects the diabetes
    view; ``horizons_minutes`` the forecast horizons."""

    features: Dict[str, float]
    outcome: str = "diabetes_status"
    horizons_minutes: List[int] = field(default_factory=lambda: [30, 60])
    patient_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict) -> "FeaturePredictionRequest":
        if not isinstance(payload, dict):
            raise ValidationError("request body must be a JSON object")
        raw = payload.get("features", payload.get("inputs"))
        if raw is None:
            # allow a flat body of feature:value pairs (minus the control keys)
            raw = {k: v for k, v in payload.items()
                   if k not in {"outcome", "selected_outcome", "horizons_minutes",
                                "prediction_horizons_minutes", "patient_id", "session_id",
                                "input_mode", "targets"}}
        if not isinstance(raw, dict):
            raise ValidationError("'features'/'inputs' must be an object")
        # flatten nested modality groups ({"metabolic": {...}, ...}) into one feature map
        flat: Dict[str, object] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        warnings: List[str] = []
        clean: Dict[str, float] = {}
        for name, value in flat.items():
            canon = _FIELD_ALIASES.get(name, name)
            if canon not in CANONICAL_FEATURES:
                warnings.append(f"ignored unknown feature '{name}'")
                continue
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValidationError(f"feature '{name}' must be numeric, got {value!r}")
            if not math.isfinite(v):
                raise ValidationError(f"feature '{name}' must be a finite number")
            v = v * _FIELD_SCALE.get(name, 1.0)
            _mod, _m, _s, lo, hi = CANONICAL_FEATURES[canon]
            if v < lo or v > hi:
                clamped = min(max(v, lo), hi)
                warnings.append(f"feature '{name}'={v} out of range [{lo}, {hi}] — clamped to {clamped}")
                v = clamped
            clean[canon] = v
        outcome = payload.get("outcome", payload.get("selected_outcome", "diabetes_status"))
        if outcome not in SELECTABLE_OUTCOMES:
            raise ValidationError(
                f"'outcome' must be one of {list(SELECTABLE_OUTCOMES)}, got {outcome!r}")
        horizons = payload.get("horizons_minutes",
                               payload.get("prediction_horizons_minutes", [30, 60]))
        if not isinstance(horizons, list) or not horizons:
            raise ValidationError("'horizons_minutes' must be a non-empty list of integers")
        try:
            horizons = [int(h) for h in horizons]
        except (TypeError, ValueError):
            raise ValidationError("'horizons_minutes' must be integers")
        if any(h <= 0 for h in horizons):
            raise ValidationError("'horizons_minutes' must be positive")
        pid = payload.get("patient_id", payload.get("session_id"))
        return cls(features=clean, outcome=outcome, horizons_minutes=horizons,
                   patient_id=(str(pid) if pid is not None else None), warnings=warnings)


@dataclass
class FeaturePredictionResponse:
    """Container mirrored into the JSON body by :func:`run_research_prediction`."""

    body: dict

    def to_dict(self) -> dict:
        return self.body


# --------------------------------------------------------------------------- scoring helpers
def _observed(features: Dict[str, float], names: List[str]) -> Dict[str, float]:
    return {n: features[n] for n in names if n in features}


def _confidence(n_observed: int, n_expected: int) -> float:
    if n_expected == 0:
        return 0.0
    return round(0.35 + 0.6 * (n_observed / n_expected), 3)


def _input_quality(features: Dict[str, float], warnings: List[str]) -> dict:
    present = {mod: False for mod in MODALITIES}
    for name in features:
        present[CANONICAL_FEATURES[name][0]] = True
    missing = [m for m, ok in present.items() if not ok]
    score = round(sum(1 for ok in present.values() if ok) / len(MODALITIES), 3)
    if score >= 0.75:
        overall = "ok"
    elif score >= 0.25:
        overall = "degraded"
    else:
        overall = "insufficient"
    return {"overall": overall, "score": score, "missing_modalities": missing,
            "warnings": list(warnings)}


def _forecast(features: Dict[str, float], horizons: List[int], *,
              screener_root=None) -> dict:
    """A glucose forecast via ``prediction.registry.resolve_predictor``. Without a committed CGM
    artifact (and without a real CGM history) the resolver abstains — we then emit a clearly-labelled
    illustrative simulation derived from the entered ``cgm_mean`` (widening interval per horizon), or
    nulls when no glucose input exists. Never a validated fused claim."""
    from dvxr.prediction.registry import resolve_predictor
    from dvxr.prediction.service import PredictionInputs

    svc = resolve_predictor("cgm_glucose_forecast", model_registry=None, artifact_root=None)
    bundle = svc.predict(PredictionInputs("cgm_glucose_forecast", horizons,
                                          requested_modalities=["cgm"]))
    out: Dict[str, dict] = {}
    if not bundle.abstained and bundle.forecast:
        for h in horizons:
            key = f"glucose_{h}m"
            f = bundle.forecast.get(key) or bundle.forecast.get(f"{h}_minutes")
            if f:
                out[f"{h}_minutes"] = {"point_mg_dl": f.get("point"), "lower_mg_dl": f.get("lower"),
                                       "upper_mg_dl": f.get("upper")}
        if out:
            out["basis"] = "committed_cgm_forecast_artifact"
            return out
    # resolver abstained (the expected research-stage default) → illustrative or null
    cgm = features.get("cgm_mean")
    if cgm is None:
        for h in horizons:
            out[f"{h}_minutes"] = {"point_mg_dl": None, "lower_mg_dl": None, "upper_mg_dl": None}
        out["basis"] = "abstained_no_committed_cgm_artifact"
        out["evidence_status"] = "abstained"
        return out
    drift = float(features.get("cgm_std", 20.0))
    for h in horizons:
        widen = drift * (1.0 + h / 60.0)
        out[f"{h}_minutes"] = {"point_mg_dl": round(float(cgm), 1),
                               "lower_mg_dl": round(float(cgm) - widen, 1),
                               "upper_mg_dl": round(float(cgm) + widen, 1)}
    out["basis"] = "illustrative_simulation_no_committed_cgm_artifact"
    out["evidence_status"] = "simulation"
    return out


# --------------------------------------------------------------------------- the service entry point
def run_research_prediction(payload: dict, *, screener_root=None) -> dict:
    """Score a research-prediction request end-to-end (validate → load committed artifacts → predict).

    NEVER trains: it only loads committed artifacts (or the labelled simulation fallback). Raises
    :class:`ValidationError` on bad input (handler maps to HTTP 400)."""
    from dvxr.serve.api import DISCLAIMER

    req = FeaturePredictionRequest.from_payload(payload)
    base_heads, meta, provenance = load_research_models(resolve_models_root(screener_root))

    prediction_id = "res-" + hashlib.sha256(
        json.dumps({"f": req.features, "o": req.outcome}, sort_keys=True).encode()).hexdigest()[:16]

    # --- per-target base predictions (each reads ONLY its modality's observed inputs) ---
    target_predictions: dict = {}
    prob_stack: Dict[str, float] = {}
    for t in TARGETS:
        head = base_heads[t]
        obs = _observed(req.features, TARGET_FEATURES[t])
        if not obs:  # no input for this modality → honest per-target abstention
            target_predictions[t] = {
                "probability": None, "risk_band": None, "confidence": 0.0,
                "model_version": head.model_version, "evidence_status": "abstained",
                "reason_codes": ["no_modality_input"]}
            continue
        p = head.predict_proba(obs)
        ev = head.evidence_status if provenance == "committed" else "simulation"
        target_predictions[t] = {
            "probability": round(float(p), 4), "risk_band": risk_band(p),
            "confidence": _confidence(len(obs), len(TARGET_FEATURES[t])),
            "model_version": head.model_version, "evidence_status": ev}
        prob_stack[f"prob_{t}"] = p

    # --- selected diabetes outcome (meta-model over metabolic covariates + OOF base probabilities) ---
    metabolic_obs = _observed(req.features, META_METABOLIC_FEATURES)
    selected = _selected_outcome(req.outcome, meta, base_heads, req.features, metabolic_obs,
                                 prob_stack, provenance)

    # --- honest linear contributions for the selected outcome (reuses explain.top_feature_attribution) ---
    contributions = _contributions_for(req.outcome, meta, base_heads, req.features,
                                        metabolic_obs, prob_stack)

    forecast = _forecast(req.features, req.horizons_minutes, screener_root=screener_root)
    quality = _input_quality(req.features, req.warnings)

    status = "ok"
    body: dict = {
        "prediction_id": prediction_id,
        "status": status,
        "research_stage": True,
        "evidence_provenance": provenance,
        "input_quality": quality,
        "target_predictions": target_predictions,
        "selected_outcome": selected,
        "contributions": contributions,
        "forecast": forecast,
        "disclaimer": DISCLAIMER,
    }
    if selected.get("status") == "abstained" or selected.get("probability") is None:
        body["status"] = "abstained"
        body["reason_codes"] = selected.get("reason_codes", ["insufficient_metabolic_input"])
        body["missing_or_stale_data"] = selected.get(
            "missing_or_stale_data", ["metabolic inputs (HbA1c / fasting glucose / CGM summary / BMI)"])
    return body


def _selected_outcome(outcome: str, meta: DiabetesMetaModel, base_heads: Dict[str, LinearHead],
                      features: Dict[str, float], metabolic_obs: Dict[str, float],
                      prob_stack: Dict[str, float], provenance: str) -> dict:
    """Build the selected diabetes/glucose outcome. ALWAYS carries validated_for_clinical_use=False and
    an experimental/simulation evidence status; abstains (probability None) when the required metabolic
    inputs are absent — an abstention no default can overwrite."""
    ev = "experimental" if provenance == "committed" else "simulation"
    if outcome == "glucose_instability":
        head = base_heads["glucose_instability"]
        obs = _observed(features, TARGET_FEATURES["glucose_instability"])
        if not obs:
            return _abstain_outcome(outcome, head.model_version)
        p = head.predict_proba(obs)
        return {"name": outcome, "probability": round(float(p), 4), "risk_band": risk_band(p),
                "confidence": _confidence(len(obs), len(TARGET_FEATURES["glucose_instability"])),
                "model_version": head.model_version, "evidence_status": ev,
                "validated_for_clinical_use": False}
    # diabetes_status / diabetes_complication → stacked meta-model
    if not metabolic_obs:  # hard abstention: cannot be overwritten by a default
        return _abstain_outcome(outcome, meta.head.model_version)
    values = dict(metabolic_obs)
    values.update(prob_stack)  # OOF base-model probabilities (missing ones stay absent → z=0)
    p = meta.predict_proba(values)
    if outcome == "diabetes_complication":
        # an experimental higher-severity view of the same stacked signal (no complication labels exist)
        p = min(1.0, p * 0.85)
    n_obs = len(metabolic_obs) + len(prob_stack)
    return {"name": outcome, "probability": round(float(p), 4), "risk_band": risk_band(p),
            "confidence": _confidence(len(metabolic_obs), len(META_METABOLIC_FEATURES)),
            "model_version": meta.head.model_version, "evidence_status": ev,
            "validated_for_clinical_use": False,
            "note": ("Research-stage stacked estimate on an EXCLUDED task (cgmacros_diabetes); "
                     "this is decision-support, not a diagnosis and not a validated clinical claim.")}


def _abstain_outcome(outcome: str, model_version: str) -> dict:
    return {"name": outcome, "status": "abstained", "probability": None, "risk_band": None,
            "confidence": 0.0, "model_version": model_version, "evidence_status": "abstained",
            "validated_for_clinical_use": False, "action_id": "INSUFFICIENT_DATA",
            "reason_codes": ["insufficient_metabolic_input"],
            "missing_or_stale_data": [
                "metabolic inputs (HbA1c / fasting glucose / CGM summary / BMI)"]}


def _contributions_for(outcome: str, meta: DiabetesMetaModel, base_heads: Dict[str, LinearHead],
                       features: Dict[str, float], metabolic_obs: Dict[str, float],
                       prob_stack: Dict[str, float]) -> List[dict]:
    """Signed contributions for the selected outcome via the shared linear-attribution surface
    (``dvxr.serve.explain.top_feature_attribution``), remapped to the response's contract keys."""
    if outcome == "glucose_instability":
        head = base_heads["glucose_instability"]
        values = _observed(features, TARGET_FEATURES["glucose_instability"])
    else:
        head = meta.head
        values = dict(metabolic_obs)
        values.update(prob_stack)
    if not values:
        return []
    return _linear_attribution(head, values)


def _linear_attribution(head: LinearHead, values: Dict[str, float], k: int = 6) -> List[dict]:
    """Reuse ``explain.top_feature_attribution`` by adapting the JSON head to its (scaler, head)
    duck-type, then remap {feature,contribution,direction} → the response's contract keys."""
    import numpy as np

    from dvxr.serve.explain import top_feature_attribution

    names = [n for n in head.features if n in values]
    if not names:
        return []
    idx = [head.features.index(n) for n in names]
    mean = np.array([head.mean[i] for i in idx], dtype=float)
    scale = np.array([head.scale[i] if head.scale[i] else 1.0 for i in idx], dtype=float)
    coef = np.array([head.coef[i] for i in idx], dtype=float)
    row = np.array([[float(values[n]) for n in names]], dtype=float)

    class _Scaler:
        def transform(self, X):
            return (np.asarray(X, dtype=float) - mean) / scale

    class _Head:
        coef_ = coef.reshape(1, -1)

    class _Shim:
        representation = "tabular_research"
        scaler = _Scaler()
        head = _Head()

    attrs = top_feature_attribution(_Shim(), row, feature_names=names, k=k)
    return [{"factor": a["feature"], "signed_contribution": a["contribution"],
             "direction": a["direction"], "method": "linear"} for a in attrs]
