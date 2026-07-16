"""The RiskPredictionService boundary + an honest CGM-only excursion baseline (Gate 3).

Two implementations sit behind one Protocol:

* ``AbstainingPredictionService`` — the research-stage default for the FUSED product. It never returns
  a number: no synchronized EEG+CGM+wearable dataset exists, so the fused claim is not evaluable.
* ``CgmOnlyExcursionService`` — a single-modality, single-cohort CGM forecaster (Ridge/GBM →
  Platt-calibrated) trained OFFLINE on the prospective excursion target (``dvxr.targets``). At request
  time it only *predicts* (never trains), and it **abstains** whenever the request needs a modality
  beyond CGM, so it can never masquerade as the fused headline.

Feature extraction is deterministic and causal — features come only from the ``[t-history, t]`` slice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import pandas as pd

from dvxr.calibration import BinaryCalibrator, fit_platt_calibrator, risk_band
from dvxr.targets import ExcursionThresholds

CGM_FEATURE_NAMES: Tuple[str, ...] = (
    "cgm_last", "cgm_mean", "cgm_std", "cgm_min", "cgm_max", "cgm_range",
    "cgm_slope_per_min", "cgm_tir", "cgm_frac_hyper", "cgm_frac_hypo", "cgm_n_samples",
)


# --------------------------------------------------------------------------- data contracts
@dataclass
class PredictionInputs:
    """Everything a predictor may read for one request. ``cgm_history`` is the causal `[t-H, t]` slice
    (columns: a timestamp and a glucose column); ``requested_modalities`` is what the report needs."""
    report_type: str
    horizons_minutes: Sequence[int]
    snapshot: object = None
    cgm_history: Optional[pd.DataFrame] = None
    requested_modalities: Sequence[str] = ()
    time_col: str = "timestamp"
    glucose_col: str = "glucose"
    cutoff: object = None                 # request cutoff t; used to gate on CGM freshness


class ScientificValidityError(ValueError):
    """Raised when a model would be fit in a way that is not scientifically valid (e.g. calibrating the
    Platt layer on its own training rows). The honest response is to SKIP the head, never self-calibrate."""


@dataclass(frozen=True)
class AdequacyConfig:
    """When a CGM history is rich enough to support a prediction (spec §9 "abstain on inadequate data").

    Defaults target ~15-min CGM (the CGMacros Libre cadence): ≥90 min of context, ≥6 valid readings, no
    gap wider than ~3 missed readings. A finer-cadence deployment (5-min Dexcom) should raise
    ``minimum_valid_points``. ``ood_abstain_z`` abstains when any feature is this many training-sigmas
    out — a physiologically implausible or off-distribution window gets no number."""
    minimum_history_minutes: float = 90.0
    minimum_valid_points: int = 6
    max_gap_minutes: float = 45.0
    ood_abstain_z: float = 6.0


@dataclass(frozen=True)
class PredictionBundle:
    """A predictor's immutable output. ``risk`` is None iff ``abstained``.

    ``decision_margin`` (distance from the 0.5 threshold, ``2·|p−0.5|``) and ``reliability`` (how much
    the number can be TRUSTED, from data adequacy + in-distribution-ness) are DISTINCT: a probability
    far from 0.5 on thin or off-distribution data has a large margin but low reliability. ``confidence``
    surfaces ``reliability`` (not the margin) to the policy engine, so escalation gates on trust."""
    risk: Optional[Dict[str, float]]
    risk_category: Optional[str]
    confidence: Optional[float]
    data_quality: str
    abstained: bool
    abstain_reason: Optional[str]
    modality_scope: str
    model_version: str
    calibration_version: str = ""
    decision_margin: Optional[float] = None
    reliability: Optional[float] = None
    ood_score: Optional[float] = None
    # optional continuous glucose forecast, per horizon {"glucose_30m": {"point","lower","upper"}, …}
    forecast: Optional[Dict[str, Dict[str, float]]] = None
    forecast_model_version: str = ""
    forecast_interval_version: str = ""
    forecast_coverage_target: Optional[float] = None


@runtime_checkable
class RiskPredictionService(Protocol):
    modality_scope: str
    model_version: str

    def predict(self, inputs: PredictionInputs) -> PredictionBundle:
        ...


# --------------------------------------------------------------------------- features
def cgm_history_features(history: pd.DataFrame, *, time_col: str = "timestamp",
                         glucose_col: str = "glucose",
                         thresholds: ExcursionThresholds = ExcursionThresholds()) -> Dict[str, float]:
    """Deterministic summary features of a causal CGM history window. Empty history → all-NaN row
    (the caller decides to abstain)."""
    if history is None or len(history) == 0:
        return {k: float("nan") for k in CGM_FEATURE_NAMES}
    g_all = pd.to_numeric(history[glucose_col], errors="coerce").to_numpy(dtype=float)
    t_all = pd.to_datetime(history[time_col], errors="coerce").to_numpy()   # datetime64[ns]
    ok = ~np.isnan(g_all)
    g = g_all[ok]
    t = t_all[ok]
    if len(g) == 0:
        return {k: float("nan") for k in CGM_FEATURE_NAMES}
    minutes = (t - t[0]) / np.timedelta64(1, "m")          # numpy, positional
    minutes = np.asarray(minutes, dtype=float)
    slope = float(np.polyfit(minutes, g, 1)[0]) if len(g) >= 2 and np.ptp(minutes) > 0 else 0.0
    return {
        "cgm_last": float(g[-1]),
        "cgm_mean": float(np.mean(g)),
        "cgm_std": float(np.std(g)),
        "cgm_min": float(np.min(g)),
        "cgm_max": float(np.max(g)),
        "cgm_range": float(np.ptp(g)),
        "cgm_slope_per_min": slope,
        "cgm_tir": float(np.mean((g >= thresholds.low_mg_dl) & (g <= thresholds.high_mg_dl))),
        "cgm_frac_hyper": float(np.mean(g > thresholds.high_mg_dl)),
        "cgm_frac_hypo": float(np.mean(g < thresholds.low_mg_dl)),
        "cgm_n_samples": float(len(g)),
    }


def build_cgm_feature_matrix(cgm: pd.DataFrame, examples: pd.DataFrame, *,
                             time_col: str = "timestamp", glucose_col: str = "glucose",
                             subject_col: str = "subject_id",
                             thresholds: ExcursionThresholds = ExcursionThresholds()
                             ) -> pd.DataFrame:
    """For each (reportable) example row, compute the causal CGM history features. Returns a frame with
    the feature columns + subject_id + horizon_minutes + label, aligned to ``examples``' order.

    The cohort is indexed by subject ONCE (sorted by time) and each anchor slices its subject's frame in
    place — never re-sorting the whole cohort per example (which made an offline fit on a dense real
    cohort O(examples × cohort). The causal window ``[anchor − history, anchor]`` is identical to
    ``dvxr.targets.history_slice``."""
    cgm = cgm.copy()
    cgm[time_col] = pd.to_datetime(cgm[time_col], errors="coerce")
    by_subject = {str(sid): g.sort_values(time_col, kind="stable")
                  for sid, g in cgm.groupby(subject_col)}
    hist = pd.Timedelta(minutes=thresholds.history_minutes)
    empty = pd.DataFrame(columns=[time_col, glucose_col])
    rows: List[Dict[str, float]] = []
    for _, ex in examples.iterrows():
        sid = str(ex["subject_id"])
        anchor = pd.Timestamp(ex["anchor_time"])
        g = by_subject.get(sid)
        if g is None:
            sl = empty
        else:
            t = g[time_col]
            sl = g[(t >= anchor - hist) & (t <= anchor)][[time_col, glucose_col]]
        feats = cgm_history_features(sl, time_col=time_col, glucose_col=glucose_col,
                                     thresholds=thresholds)
        feats["subject_id"] = sid
        feats["horizon_minutes"] = int(ex["horizon_minutes"])
        feats["label"] = ex["label"]
        rows.append(feats)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- shared request-time gates
# Free functions so the excursion classifier and the continuous forecaster apply IDENTICAL causal
# windowing, adequacy, freshness, and OOD gates (spec §5, §9) — no drift between the two services.
def window_to_anchor(history: Optional[pd.DataFrame], cutoff, *, time_col: str,
                     history_minutes: float) -> Optional[pd.DataFrame]:
    """Slice a CGM history to the ``[anchor − history_minutes, anchor]`` window training used, so serving
    features match the training distribution. ``anchor`` is the cutoff when known, else the last sample."""
    if history is None or len(history) == 0:
        return history
    # UTC-normalize both sides: windowed history may be tz-naive while a resolved cutoff is tz-aware;
    # comparing them raw raises "Cannot compare tz-naive and tz-aware". utc=True coerces either form.
    t = pd.to_datetime(history[time_col], errors="coerce", utc=True)
    anchor = pd.to_datetime(cutoff, errors="coerce", utc=True) if cutoff is not None else pd.NaT
    if pd.isna(anchor):
        anchor = t.max()
    if pd.isna(anchor):
        return history
    start = anchor - pd.Timedelta(minutes=float(history_minutes))
    return history[(t >= start) & (t <= anchor)]


def assess_history_adequacy(history: Optional[pd.DataFrame], *, time_col: str, glucose_col: str,
                            adequacy: "AdequacyConfig") -> Tuple[bool, Optional[str], float, int]:
    """Return (ok, abstain_reason, span_minutes, n_valid). A single reading or a thin/gappy window is
    NOT enough to forecast — require a minimum span, point count, and cadence."""
    if history is None or len(history) == 0:
        return False, "no usable CGM history at the cutoff", 0.0, 0
    g = pd.to_numeric(history[glucose_col], errors="coerce")
    t = pd.to_datetime(history[time_col], errors="coerce")
    ok_mask = g.notna() & t.notna()
    t = t[ok_mask].sort_values()
    n_valid = int(len(t))
    if n_valid == 0:
        return False, "no usable CGM history at the cutoff", 0.0, 0
    span = float((t.iloc[-1] - t.iloc[0]) / pd.Timedelta(minutes=1)) if n_valid > 1 else 0.0
    a = adequacy
    if n_valid < a.minimum_valid_points:
        return (False, f"inadequate CGM history: {n_valid} readings "
                f"(< {a.minimum_valid_points} required)", span, n_valid)
    if span < a.minimum_history_minutes:
        return (False, f"inadequate CGM history span: {span:.0f} min "
                f"(< {a.minimum_history_minutes:.0f} required)", span, n_valid)
    gaps = t.diff().dropna() / pd.Timedelta(minutes=1)
    if len(gaps) and float(gaps.max()) > a.max_gap_minutes:
        return (False, f"CGM history has a {float(gaps.max()):.0f}-min gap "
                f"(> {a.max_gap_minutes:.0f} allowed) — coverage too sparse", span, n_valid)
    return True, None, span, n_valid


def ood_from_moments(x: np.ndarray, feat_mean, feat_std, ood_abstain_z: float) -> float:
    """Max per-feature |z| against the training moments, normalised by the abstention threshold →
    0 = in-distribution, 1 = at the OOD boundary, >1 = out of distribution. Near-zero-variance features
    carry no distributional signal and are excluded."""
    if feat_mean is None or feat_std is None:
        return 0.0
    std = np.asarray(feat_std, dtype=float)
    informative = std > 1e-6
    if not informative.any():
        return 0.0
    z = np.abs((x[0][informative] - feat_mean[informative]) / std[informative])
    return float(np.max(z) / ood_abstain_z) if len(z) else 0.0


def staleness_minutes(history: Optional[pd.DataFrame], cutoff, time_col: str) -> Optional[float]:
    """Minutes between the newest sample and the cutoff (None if not computable). A large value ⇒ a stale
    feed, reported as such rather than mis-read as 'no history' after windowing."""
    if cutoff is None or history is None or len(history) == 0:
        return None
    # Normalize BOTH sides to UTC-aware before subtracting: the windowed history may be tz-naive
    # (``_cgm_history_from_events`` localizes to None) while a resolved "Generate now" cutoff is
    # tz-aware. Comparing the two raw raises "Cannot subtract tz-naive and tz-aware"; ``utc=True``
    # coerces either form to UTC so staleness is computed instead of crashing the request path.
    last_t = pd.to_datetime(history[time_col], errors="coerce", utc=True).max()
    c = pd.to_datetime(cutoff, errors="coerce", utc=True)
    if pd.isna(last_t) or pd.isna(c):
        return None
    return float((c - last_t) / pd.Timedelta(minutes=1))


# --------------------------------------------------------------------------- services
class AbstainingPredictionService:
    """The research-stage default: always abstains (spec §8.7 safe default)."""
    modality_scope = "fused_gated"
    model_version = "neuroglycemic-sentinel/research-stage"

    def __init__(self, reason: Optional[str] = None):
        self.reason = reason or (
            "A reliable fused glucose-excursion prediction cannot be produced: it requires "
            "synchronized same-subject EEG+wearable+CGM data, which does not exist in this deployment.")

    def predict(self, inputs: PredictionInputs) -> PredictionBundle:
        return PredictionBundle(
            risk=None, risk_category=None, confidence=None, data_quality="unknown",
            abstained=True, abstain_reason=self.reason, modality_scope=self.modality_scope,
            model_version=self.model_version)


class CgmOnlyExcursionService:
    """Single-modality CGM excursion forecaster. Trained offline; predicts (never trains) at request
    time; abstains for any request needing a modality beyond CGM."""
    modality_scope = "cgm_only"

    def __init__(self, models: Dict[int, Tuple[object, BinaryCalibrator]], *,
                 model_version: str, calibration_version: str,
                 thresholds: ExcursionThresholds, feature_names: Sequence[str] = CGM_FEATURE_NAMES,
                 max_staleness_minutes: float = 30.0,
                 adequacy: AdequacyConfig = AdequacyConfig(),
                 feature_mean: Optional[np.ndarray] = None, feature_std: Optional[np.ndarray] = None,
                 skipped_horizons: Sequence[int] = ()):
        self._models = models                       # horizon -> (sklearn clf, calibrator)
        self.model_version = model_version
        self.calibration_version = calibration_version
        self._thr = thresholds
        self._features = list(feature_names)
        self._max_staleness = float(max_staleness_minutes)
        self._adeq = adequacy
        # training feature moments for the OOD gate (per CGM_FEATURE_NAMES); None ⇒ OOD gate is a no-op
        self._feat_mean = feature_mean
        self._feat_std = feature_std
        #: horizons whose head could NOT be fit with an honest held-out calibration slice (skipped, not
        #: self-calibrated) — surfaced so a caller knows exactly which horizons this model won't score.
        self.skipped_horizons = list(skipped_horizons)

    # ---- offline training ----
    @classmethod
    def fit(cls, cgm: pd.DataFrame, examples: pd.DataFrame, *,
            thresholds: ExcursionThresholds = ExcursionThresholds(),
            time_col: str = "timestamp", glucose_col: str = "glucose",
            subject_col: str = "subject_id", model_version: str = "cgm-only/pilot-v1",
            calibration_frac: float = 0.25, seed: int = 7,
            max_staleness_minutes: float = 30.0,
            adequacy: AdequacyConfig = AdequacyConfig()) -> "CgmOnlyExcursionService":
        """Fit one calibrated classifier per horizon on the reportable (uncensored) examples. A
        held-out CALIBRATION slice (by subject) fits the Platt layer — NEVER the training rows. If a
        horizon has no valid held-out calibration slice (too few subjects, or a single-class fold), its
        head is SKIPPED (recorded in ``skipped_horizons``) rather than self-calibrated on training data,
        which would report optimistic, invalid calibration."""
        from sklearn.ensemble import GradientBoostingClassifier

        from dvxr.eval.splits import subject_holdout_split

        rep = examples[examples["censored"] == False].copy()  # noqa: E712
        feat = build_cgm_feature_matrix(cgm, rep, time_col=time_col, glucose_col=glucose_col,
                                        subject_col=subject_col, thresholds=thresholds)
        feat = feat.dropna(subset=list(CGM_FEATURE_NAMES))
        # training feature moments (pooled over fitted rows) drive the request-time OOD gate
        feat_mean = feat_std = None
        if len(feat):
            Xall = feat[list(CGM_FEATURE_NAMES)].to_numpy(dtype=float)
            feat_mean = Xall.mean(axis=0)
            feat_std = Xall.std(axis=0)
        models: Dict[int, Tuple[object, BinaryCalibrator]] = {}
        skipped: List[int] = []
        for h in sorted(set(int(x) for x in thresholds.horizons_minutes)):
            sub = feat[feat["horizon_minutes"] == h]
            y = sub["label"].astype(int).to_numpy()
            if len(sub) < 8 or len(np.unique(y)) < 2:
                skipped.append(h)                   # not enough signal to fit this horizon honestly
                continue
            X = sub[list(CGM_FEATURE_NAMES)].to_numpy(dtype=float)
            # subject-held-out calibration slice — the Platt layer sees ONLY these held-out subjects
            tr, cal = subject_holdout_split(sub["subject_id"].to_numpy(), test_frac=calibration_frac,
                                            seed=seed)
            if len(np.unique(y[tr])) < 2 or len(cal) == 0 or len(np.unique(y[cal])) < 2:
                # no honest calibration slice ⇒ SKIP (never self-calibrate on the training rows)
                skipped.append(h)
                continue
            clf = GradientBoostingClassifier(random_state=seed)
            clf.fit(X[tr], y[tr])
            raw_cal = clf.predict_proba(X[cal])[:, 1]
            calib = fit_platt_calibrator(raw_cal, y[cal])
            models[h] = (clf, calib)
        return cls(models, model_version=model_version,
                   calibration_version=f"platt/{thresholds.version}", thresholds=thresholds,
                   max_staleness_minutes=max_staleness_minutes, adequacy=adequacy,
                   feature_mean=feat_mean, feature_std=feat_std, skipped_horizons=skipped)

    # ---- artifact persistence (fit OFFLINE, save, then load-and-serve; the API never fits) ----
    ARTIFACT_FORMAT = "dvxr-cgm-excursion/1"

    def save(self, path) -> "object":
        """Persist the fitted service as a portable artifact (``model.joblib`` + ``manifest.json``),
        mirroring ``serve.screener.Screener.save``. The joblib blob carries the picklable model state
        (per-horizon classifier+calibrator, thresholds, adequacy config, OOD moments); the manifest
        carries human-readable metadata plus the joblib's sha256 so a loader can verify integrity."""
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
            "skipped_horizons": self.skipped_horizons,
            "model_version": self.model_version, "calibration_version": self.calibration_version,
        }
        joblib.dump(blob, path / "model.joblib")
        sha = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()
        manifest = {
            "format": self.ARTIFACT_FORMAT, "modality_scope": self.modality_scope,
            "model_version": self.model_version, "calibration_version": self.calibration_version,
            "threshold_version": self._thr.version, "feature_names": list(self._features),
            "horizons_fitted": sorted(int(h) for h in self._models),
            "skipped_horizons": list(self.skipped_horizons),
            "max_staleness_minutes": self._max_staleness, "artifact_sha256": sha,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, path) -> "CgmOnlyExcursionService":
        """Load a committed artifact and verify its integrity against the manifest sha256. Raises if the
        joblib bytes do not match (a tampered/corrupt artifact must never silently serve)."""
        import hashlib
        import json
        from pathlib import Path

        import joblib

        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        raw = (path / "model.joblib").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if manifest.get("artifact_sha256") and sha != manifest["artifact_sha256"]:
            raise ValueError(f"CGM artifact sha256 mismatch at {path} — refusing to load a tampered model")
        blob = joblib.load(path / "model.joblib")
        return cls(blob["models"], model_version=blob["model_version"],
                   calibration_version=blob["calibration_version"], thresholds=blob["thresholds"],
                   feature_names=blob["feature_names"],
                   max_staleness_minutes=blob["max_staleness_minutes"], adequacy=blob["adequacy"],
                   feature_mean=blob["feature_mean"], feature_std=blob["feature_std"],
                   skipped_horizons=blob["skipped_horizons"])

    # ---- request-time gates ----
    def _causal_window(self, inputs: PredictionInputs) -> Optional[pd.DataFrame]:
        """Slice the request history to the SAME ``[anchor − history_minutes, anchor]`` window training
        used, so serving features match the training distribution (otherwise a feature like
        ``cgm_n_samples`` — which counts readings in the window — differs purely because the caller
        passed a longer history, and every prediction would look out-of-distribution). ``anchor`` is the
        request cutoff when known, else the last observed sample."""
        return window_to_anchor(inputs.cgm_history, inputs.cutoff, time_col=inputs.time_col,
                                history_minutes=self._thr.history_minutes)

    def _history_adequacy(self, inputs: PredictionInputs) -> Tuple[bool, Optional[str], float, int]:
        """Return (ok, abstain_reason, span_minutes, n_valid) — the shared adequacy gate."""
        return assess_history_adequacy(inputs.cgm_history, time_col=inputs.time_col,
                                       glucose_col=inputs.glucose_col, adequacy=self._adeq)

    def _ood_score(self, x: np.ndarray) -> float:
        """Max per-feature |z| against the training moments (shared OOD gate)."""
        return ood_from_moments(x, self._feat_mean, self._feat_std, self._adeq.ood_abstain_z)

    # ---- request-time inference ----
    def predict(self, inputs: PredictionInputs) -> PredictionBundle:
        import dataclasses
        needed = set(inputs.requested_modalities) or {"cgm"}
        if not needed <= {"cgm"}:
            return AbstainingPredictionService(
                f"CGM-only service abstains: request needs {sorted(needed - {'cgm'})}, which this "
                f"single-modality model does not cover.").predict(inputs)
        # freshness gate FIRST, on the ORIGINAL (pre-window) history: a feed whose newest sample is long
        # before the cutoff is STALE — report that, rather than letting the window drop every stale
        # sample and mis-report it as "no history" (spec §5, §9).
        if inputs.cutoff is not None and inputs.cgm_history is not None and len(inputs.cgm_history):
            last_t = pd.to_datetime(inputs.cgm_history[inputs.time_col], errors="coerce", utc=True).max()
            cutoff = pd.to_datetime(inputs.cutoff, errors="coerce", utc=True)
            if pd.notna(last_t) and pd.notna(cutoff):
                staleness = (cutoff - last_t) / pd.Timedelta(minutes=1)
                if staleness > self._max_staleness:
                    return AbstainingPredictionService(
                        f"CGM-only service abstains: CGM feed is stale "
                        f"({staleness:.0f} min > {self._max_staleness:.0f} min at the cutoff).").predict(inputs)
        # window to the SAME [anchor-H, anchor] slice training used, so features are comparable
        inputs = dataclasses.replace(inputs, cgm_history=self._causal_window(inputs))
        # adequacy gate (spec §9): a single reading or a thin/gappy window earns an abstention, not a guess
        ok, reason, _span, _n = self._history_adequacy(inputs)
        if not ok:
            return AbstainingPredictionService(f"CGM-only service abstains: {reason}.").predict(inputs)
        feats = cgm_history_features(inputs.cgm_history, time_col=inputs.time_col,
                                     glucose_col=inputs.glucose_col, thresholds=self._thr)
        if any(np.isnan(v) for v in feats.values()):
            return AbstainingPredictionService(
                "CGM-only service abstains: no usable CGM history at the cutoff.").predict(inputs)
        x = np.array([[feats[k] for k in CGM_FEATURE_NAMES]], dtype=float)
        # out-of-distribution gate (spec §9): a window unlike anything the model was trained on gets no
        # number — extrapolation is not a calibrated prediction.
        ood = self._ood_score(x)
        if ood >= 1.0:
            return AbstainingPredictionService(
                f"CGM-only service abstains: CGM window is out of the training distribution "
                f"(OOD score {ood:.2f} ≥ 1.0).").predict(inputs)
        risk: Dict[str, float] = {}
        for h in inputs.horizons_minutes:
            h = int(h)
            if h not in self._models:
                continue
            clf, calib = self._models[h]
            p = float(calib.predict(clf.predict_proba(x)[:, 1])[0])
            risk[f"excursion_{h}m"] = round(p, 6)
        if not risk:
            return AbstainingPredictionService(
                "CGM-only service abstains: no fitted head for the requested horizon(s).").predict(inputs)
        pmax = max(risk.values())
        # decision margin (distance from the 0.5 threshold) is DISTINCT from reliability (trust). A
        # confident-looking probability on a near-OOD window has a large margin but low reliability;
        # the policy engine gates on reliability (surfaced as ``confidence``), never on the margin.
        decision_margin = round(abs(pmax - 0.5) * 2, 6)
        reliability = round((1.0 - ood), 6)
        return PredictionBundle(
            risk=risk, risk_category=risk_band(pmax), confidence=reliability,
            data_quality="acceptable", abstained=False, abstain_reason=None,
            modality_scope=self.modality_scope, model_version=self.model_version,
            calibration_version=self.calibration_version,
            decision_margin=decision_margin, reliability=reliability, ood_score=round(ood, 6))
