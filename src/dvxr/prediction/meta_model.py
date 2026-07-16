"""dvxr.prediction.meta_model — the research-stage diabetes META-model (JSON, no pickle).

A small, fully-transparent logistic meta-learner that stacks metabolic covariates (HbA1c, fasting
glucose, BMI, CGM variability, time-above-range) with the OUT-OF-FOLD probabilities of the per-target
base models (stress / anxiety / depression / cognitive workload). It is trained OFFLINE by
``scripts/train_research_meta.py`` on subject-level folds and serialised as plain JSON coefficients so
the artifact is git-committable, human-auditable, and free of any pickle/security surface.

HONESTY: this model targets the ``cgmacros_diabetes`` task, which is in the honesty audit's
``EXCLUDED_TASKS``. It therefore ALWAYS carries ``validated_for_clinical_use = False`` and an
``experimental`` / ``simulation`` evidence status — never a headline AUROC, never a diagnosis. It is a
research illustration of stacking, not a clinical claim.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class LinearHead:
    """A standardise-then-logistic head, stored as plain floats (JSON-serialisable).

    ``predict_proba`` accepts a dict {feature_name: value}; any feature absent from the dict is treated
    as *at the reference mean* (standardised value 0) — it contributes nothing, and the caller decides
    separately whether too-much-missing warrants abstention."""

    features: List[str]
    mean: List[float]
    scale: List[float]
    coef: List[float]
    intercept: float = 0.0
    platt_coef: Optional[float] = None
    platt_intercept: Optional[float] = None
    model_version: str = "research-linear/v0"
    evidence_status: str = "simulation"
    auroc_oof: Optional[float] = None
    validated_for_clinical_use: bool = False

    def standardized(self, values: Dict[str, float]) -> Dict[str, float]:
        z: Dict[str, float] = {}
        for i, name in enumerate(self.features):
            if name in values and values[name] is not None:
                sc = self.scale[i] if self.scale[i] not in (0, None) else 1.0
                z[name] = (float(values[name]) - self.mean[i]) / sc
        return z

    def raw_logit(self, values: Dict[str, float]) -> float:
        z = self.standardized(values)
        logit = float(self.intercept)
        for i, name in enumerate(self.features):
            if name in z:
                logit += float(self.coef[i]) * z[name]
        return logit

    def predict_proba(self, values: Dict[str, float]) -> float:
        p = _sigmoid(self.raw_logit(values))
        if self.platt_coef is not None and self.platt_intercept is not None:
            # Platt recalibration in logit space of the base probability
            p = min(max(p, 1e-6), 1 - 1e-6)
            base_logit = math.log(p / (1 - p))
            p = _sigmoid(self.platt_coef * base_logit + self.platt_intercept)
        return float(p)

    def signed_contributions(self, values: Dict[str, float]) -> List[Dict[str, object]]:
        """Per-feature signed contribution (standardized_value × coefficient) for the OBSERVED
        features only — the honest linear attribution the response surfaces."""
        z = self.standardized(values)
        out: List[Dict[str, object]] = []
        for i, name in enumerate(self.features):
            if name in z:
                c = float(self.coef[i]) * z[name]
                out.append({"factor": name, "signed_contribution": round(c, 4),
                            "direction": "raises" if c > 0 else "lowers", "method": "linear"})
        out.sort(key=lambda d: abs(d["signed_contribution"]), reverse=True)
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LinearHead":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class DiabetesMetaModel:
    """Stacked logistic meta-model: metabolic covariates + base-model OOF probabilities → diabetes
    status. A thin wrapper over :class:`LinearHead` that fixes the honesty invariants."""

    head: LinearHead
    metabolic_features: List[str] = field(default_factory=list)
    prob_features: List[str] = field(default_factory=list)

    def predict_proba(self, values: Dict[str, float]) -> float:
        return self.head.predict_proba(values)

    def signed_contributions(self, values: Dict[str, float]) -> List[Dict[str, object]]:
        return self.head.signed_contributions(values)

    def to_dict(self) -> dict:
        return {"head": self.head.to_dict(), "metabolic_features": self.metabolic_features,
                "prob_features": self.prob_features,
                "validated_for_clinical_use": False, "research_stage": True}

    @classmethod
    def from_dict(cls, d: dict) -> "DiabetesMetaModel":
        return cls(head=LinearHead.from_dict(d["head"]),
                   metabolic_features=list(d.get("metabolic_features", [])),
                   prob_features=list(d.get("prob_features", [])))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "DiabetesMetaModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_linear_head(X: np.ndarray, y: np.ndarray, feature_names: List[str], *,
                    model_version: str, evidence_status: str,
                    auroc_oof: Optional[float] = None) -> LinearHead:
    """Fit a standardise-then-logistic head. Standardisation stats + coefficients are returned as plain
    lists so the head serialises to JSON. One-class ``y`` yields a degenerate (zero-coef) head."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Xz = (X - mean) / scale
    if len(np.unique(y)) < 2:
        return LinearHead(features=list(feature_names), mean=mean.tolist(), scale=scale.tolist(),
                          coef=[0.0] * len(feature_names), intercept=0.0,
                          model_version=model_version, evidence_status=evidence_status,
                          auroc_oof=auroc_oof)
    clf = LogisticRegression(max_iter=2000, random_state=7)
    clf.fit(Xz, y)
    return LinearHead(features=list(feature_names), mean=mean.tolist(), scale=scale.tolist(),
                      coef=clf.coef_.ravel().tolist(), intercept=float(clf.intercept_[0]),
                      model_version=model_version, evidence_status=evidence_status,
                      auroc_oof=auroc_oof)
