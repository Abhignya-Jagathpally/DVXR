"""dvxr.prediction.forecast_service — a servable CGM-only continuous glucose forecaster (Gate 3).

The excursion classifier (`CgmOnlyExcursionService`) answers the *binary* "will a threshold be crossed
in (t, t+h]?". This service answers the *continuous* one — "what will glucose be at t+h, with what
calibrated interval?" — as a deployable, fit-offline / predict-at-request-time artifact that mirrors the
classifier exactly:

  * **CGM-only.** Same causal history features (`cgm_history_features`); abstains for any non-CGM request.
  * **Fit offline, never at request time.** `fit()` trains one `GradientBoostingRegressor` per horizon
    plus a split-conformal radius `q` on a subject-held-out calibration slice (reusing
    `dvxr.eval.glucose_forecast`); `predict()` only infers.
  * **Same request-time gates** as the classifier — shared `window_to_anchor`, `assess_history_adequacy`,
    `staleness_minutes`, `ood_from_moments` (spec §5, §9) — so a thin/stale/OOD window abstains, never
    guesses.
  * **Calibrated interval.** Each horizon yields `{point, lower, upper}` = `ŷ ± q` at the target
    coverage, plus **persistence** (`ŷ=cgm_last`) and **linear-extrapolation** (`ŷ=cgm_last+slope·h`)
    baselines the learned forecaster is measured against (held-out, recorded in the artifact manifest).
  * **Portable artifact.** `save`/`load` (joblib + sha256-verified manifest), like the classifier.

Every EEG/fused forecast stays out of scope (no synchronized cohort); this service is single-modality,
single-cohort, and labels itself as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dvxr.eval.clinical_metrics import mae, rmse
from dvxr.eval.glucose_forecast import (
    _conformal_quantile,
    build_forecast_examples,
    build_forecast_matrix,
)
from dvxr.prediction.service import (
    CGM_FEATURE_NAMES,
    AbstainingPredictionService,
    AdequacyConfig,
    PredictionBundle,
    PredictionInputs,
    assess_history_adequacy,
    cgm_history_features,
    ood_from_moments,
    staleness_minutes,
    window_to_anchor,
)
from dvxr.targets import ExcursionThresholds


def _persistence(feats: Dict[str, float], horizon: int) -> float:
    """Naive persistence forecast: glucose stays at its last observed value."""
    return float(feats["cgm_last"])


def _linear_extrapolation(feats: Dict[str, float], horizon: int) -> float:
    """Linear extrapolation from the causal window's fitted slope: ŷ = last + slope·h."""
    return float(feats["cgm_last"] + feats["cgm_slope_per_min"] * float(horizon))


class CgmOnlyGlucoseForecastService:
    """Single-modality CGM glucose forecaster with a split-conformal interval. Trained offline; predicts
    (never trains) at request time; abstains for any request needing a modality beyond CGM."""
    modality_scope = "cgm_only_forecast"
    ARTIFACT_FORMAT = "dvxr-cgm-forecast/1"

    def __init__(self, models: Dict[int, Tuple[object, float]], *, model_version: str,
                 interval_version: str, coverage_target: float, thresholds: ExcursionThresholds,
                 feature_names: Sequence[str] = CGM_FEATURE_NAMES, max_staleness_minutes: float = 30.0,
                 adequacy: AdequacyConfig = AdequacyConfig(),
                 feature_mean: Optional[np.ndarray] = None, feature_std: Optional[np.ndarray] = None,
                 skipped_horizons: Sequence[int] = (), baseline_report: Optional[Dict] = None):
        self._models = models                       # horizon -> (sklearn regressor, conformal radius q)
        self.model_version = model_version
        self.interval_version = interval_version    # e.g. "split-conformal/0.90"
        self.coverage_target = float(coverage_target)
        self._thr = thresholds
        self._features = list(feature_names)
        self._max_staleness = float(max_staleness_minutes)
        self._adeq = adequacy
        self._feat_mean = feature_mean
        self._feat_std = feature_std
        self.skipped_horizons = list(skipped_horizons)
        #: held-out error of the learned forecaster vs. persistence/linear baselines (honesty context).
        self.baseline_report = baseline_report or {}

    # ---- offline training ----
    @classmethod
    def fit(cls, cgm: pd.DataFrame, *, thresholds: ExcursionThresholds = ExcursionThresholds(),
            subject_col: str = "subject_id", model_version: str = "cgm-forecast/pilot-v1",
            alpha: float = 0.1, calibration_frac: float = 0.25, seed: int = 7,
            max_staleness_minutes: float = 30.0, anchors: Optional[Sequence] = None,
            adequacy: AdequacyConfig = AdequacyConfig()) -> "CgmOnlyGlucoseForecastService":
        """Fit one regressor + conformal radius per horizon on a subject-held-out calibration slice. If a
        horizon has too few subjects/rows for an honest calibration split, its head is SKIPPED (recorded)
        rather than calibrated on its own training rows. ``anchors`` (optional) restricts which cutoff
        times seed examples — pass a thinned list on a dense real cohort so training is tractable; None
        uses every observed timestamp (fine for small synthetic cohorts)."""
        from sklearn.ensemble import GradientBoostingRegressor

        from dvxr.eval.splits import subject_holdout_split

        examples = build_forecast_examples(cgm, thresholds=thresholds, anchors=anchors,
                                           subject_col=subject_col)
        X, y, subs, _keys, horizons, names = build_forecast_matrix(cgm, examples, thresholds=thresholds)
        # build_forecast_matrix returns columns in sorted(feature) order; canonicalize to
        # CGM_FEATURE_NAMES so training, the OOD moments, the baselines, and request-time `predict`
        # (which builds x in CGM_FEATURE_NAMES order) all agree — a feature-order mismatch would make
        # every served window look out-of-distribution.
        if len(X) and list(names) != list(CGM_FEATURE_NAMES):
            order = [list(names).index(k) for k in CGM_FEATURE_NAMES]
            X = X[:, order]
        feat_mean = feat_std = None
        if len(X):
            feat_mean = X.mean(axis=0)
            feat_std = X.std(axis=0)
        models: Dict[int, Tuple[object, float]] = {}
        skipped: List[int] = []
        baseline: Dict[int, Dict[str, float]] = {}
        for h in sorted(set(int(x) for x in thresholds.horizons_minutes)):
            m = horizons == h
            if int(m.sum()) < 8 or len(np.unique(subs[m])) < 2:
                skipped.append(h)
                continue
            Xh, yh, sh = X[m], y[m], subs[m]
            tr_i, cal_i = subject_holdout_split(sh, test_frac=calibration_frac, seed=seed)
            if len(tr_i) == 0 or len(cal_i) == 0:
                skipped.append(h)
                continue
            reg = GradientBoostingRegressor(random_state=seed)
            reg.fit(Xh[tr_i], yh[tr_i])
            resid = np.abs(yh[cal_i] - reg.predict(Xh[cal_i]))
            q = _conformal_quantile(resid, alpha)
            models[h] = (reg, float(q))
            baseline[h] = cls._baseline_scores(Xh[cal_i], yh[cal_i], reg, h)
        return cls(models, model_version=model_version,
                   interval_version=f"split-conformal/{1 - alpha:.2f}", coverage_target=1.0 - alpha,
                   thresholds=thresholds, max_staleness_minutes=max_staleness_minutes, adequacy=adequacy,
                   feature_mean=feat_mean, feature_std=feat_std, skipped_horizons=skipped,
                   baseline_report=baseline)

    @staticmethod
    def _baseline_scores(Xcal: np.ndarray, ycal: np.ndarray, reg, horizon: int) -> Dict[str, float]:
        """Held-out RMSE/MAE of the learned forecaster vs persistence and linear extrapolation on the
        calibration subjects — so the manifest records whether the learned model actually beats the
        naive baselines (it should not be trusted if it doesn't)."""
        names = list(CGM_FEATURE_NAMES)
        feats = [{k: float(v) for k, v in zip(names, row)} for row in Xcal]
        persist = np.array([_persistence(f, horizon) for f in feats])
        linear = np.array([_linear_extrapolation(f, horizon) for f in feats])
        learned = reg.predict(Xcal)
        return {
            "learned_rmse": round(rmse(ycal, learned), 3), "learned_mae": round(mae(ycal, learned), 3),
            "persistence_rmse": round(rmse(ycal, persist), 3),
            "persistence_mae": round(mae(ycal, persist), 3),
            "linear_rmse": round(rmse(ycal, linear), 3), "linear_mae": round(mae(ycal, linear), 3),
            "beats_persistence": bool(rmse(ycal, learned) < rmse(ycal, persist)),
            "beats_linear": bool(rmse(ycal, learned) < rmse(ycal, linear)),
            "n_cal": int(len(ycal)),
        }

    # ---- request-time inference ----
    def predict(self, inputs: PredictionInputs) -> PredictionBundle:
        needed = set(inputs.requested_modalities) or {"cgm"}
        if not needed <= {"cgm"}:
            return AbstainingPredictionService(
                f"CGM-only forecaster abstains: request needs {sorted(needed - {'cgm'})}, beyond CGM."
            ).predict(inputs)
        stale = staleness_minutes(inputs.cgm_history, inputs.cutoff, inputs.time_col)
        if stale is not None and stale > self._max_staleness:
            return AbstainingPredictionService(
                f"CGM-only forecaster abstains: CGM feed is stale ({stale:.0f} min > "
                f"{self._max_staleness:.0f} min at the cutoff).").predict(inputs)
        history = window_to_anchor(inputs.cgm_history, inputs.cutoff, time_col=inputs.time_col,
                                   history_minutes=self._thr.history_minutes)
        ok, reason, _span, _n = assess_history_adequacy(
            history, time_col=inputs.time_col, glucose_col=inputs.glucose_col, adequacy=self._adeq)
        if not ok:
            return AbstainingPredictionService(f"CGM-only forecaster abstains: {reason}.").predict(inputs)
        feats = cgm_history_features(history, time_col=inputs.time_col,
                                     glucose_col=inputs.glucose_col, thresholds=self._thr)
        if any(np.isnan(v) for v in feats.values()):
            return AbstainingPredictionService(
                "CGM-only forecaster abstains: no usable CGM history at the cutoff.").predict(inputs)
        x = np.array([[feats[k] for k in CGM_FEATURE_NAMES]], dtype=float)
        ood = ood_from_moments(x, self._feat_mean, self._feat_std, self._adeq.ood_abstain_z)
        if ood >= 1.0:
            return AbstainingPredictionService(
                f"CGM-only forecaster abstains: CGM window is out of the training distribution "
                f"(OOD score {ood:.2f} ≥ 1.0).").predict(inputs)
        forecast: Dict[str, Dict[str, float]] = {}
        for h in inputs.horizons_minutes:
            h = int(h)
            if h not in self._models:
                continue
            reg, q = self._models[h]
            point = float(reg.predict(x)[0])
            lo = float(point - q) if np.isfinite(q) else None
            hi = float(point + q) if np.isfinite(q) else None
            forecast[f"glucose_{h}m"] = {
                "point": round(point, 2),
                "lower": round(lo, 2) if lo is not None else None,
                "upper": round(hi, 2) if hi is not None else None,
                "interval_finite": bool(np.isfinite(q)),
            }
        if not forecast:
            return AbstainingPredictionService(
                "CGM-only forecaster abstains: no fitted head for the requested horizon(s).").predict(inputs)
        reliability = round(1.0 - ood, 6)
        return PredictionBundle(
            risk=None, risk_category=None, confidence=reliability, data_quality="acceptable",
            abstained=False, abstain_reason=None, modality_scope=self.modality_scope,
            model_version=self.model_version, reliability=reliability, ood_score=round(ood, 6),
            forecast=forecast, forecast_model_version=self.model_version,
            forecast_interval_version=self.interval_version,
            forecast_coverage_target=self.coverage_target)

    # ---- artifact persistence ----
    def save(self, path) -> "object":
        import hashlib
        import json
        from pathlib import Path

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {
            "models": self._models, "thresholds": self._thr, "adequacy": self._adeq,
            "feature_names": self._features, "max_staleness_minutes": self._max_staleness,
            "feature_mean": self._feat_mean, "feature_std": self._feat_std,
            "skipped_horizons": self.skipped_horizons, "model_version": self.model_version,
            "interval_version": self.interval_version, "coverage_target": self.coverage_target,
            "baseline_report": self.baseline_report,
        }
        joblib.dump(blob, path / "model.joblib")
        sha = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()
        manifest = {
            "format": self.ARTIFACT_FORMAT, "modality_scope": self.modality_scope,
            "model_version": self.model_version, "interval_version": self.interval_version,
            "coverage_target": self.coverage_target, "threshold_version": self._thr.version,
            "feature_names": list(self._features),
            "horizons_fitted": sorted(int(h) for h in self._models),
            "skipped_horizons": list(self.skipped_horizons),
            "max_staleness_minutes": self._max_staleness,
            "baseline_report": self.baseline_report, "artifact_sha256": sha,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, path) -> "CgmOnlyGlucoseForecastService":
        import hashlib
        import json
        from pathlib import Path

        import joblib

        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        raw = (path / "model.joblib").read_bytes()
        if manifest.get("artifact_sha256") and hashlib.sha256(raw).hexdigest() != manifest["artifact_sha256"]:
            raise ValueError(f"forecast artifact sha256 mismatch at {path} — refusing to load")
        blob = joblib.load(path / "model.joblib")
        return cls(blob["models"], model_version=blob["model_version"],
                   interval_version=blob["interval_version"], coverage_target=blob["coverage_target"],
                   thresholds=blob["thresholds"], feature_names=blob["feature_names"],
                   max_staleness_minutes=blob["max_staleness_minutes"], adequacy=blob["adequacy"],
                   feature_mean=blob["feature_mean"], feature_std=blob["feature_std"],
                   skipped_horizons=blob["skipped_horizons"], baseline_report=blob.get("baseline_report"))
