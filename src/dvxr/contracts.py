"""dvxr.contracts — the traceability spine of the Generate lifecycle (spec §2, §6).

Immutable dataclasses that flow through the request lifecycle: a `GenerateRequest` comes in, an
immutable `PatientSnapshot` is assembled at an explicit `data_cutoff_at`, the model produces a
`RiskPrediction` (+ `ModelEvidence`), and the policy engine returns an `ActionDecision`. Every object
carries the versions (`model_version`, `feature_version`, `data_cutoff_at`, `schema_version`) needed to
reproduce it, so any displayed number can be traced back to exact source events and code versions.

These are plain dataclasses (`to_dict`/`from_dict`, JSON-round-trippable) — no I/O, no wall-clock
reads — so they are deterministic and storable by any `dvxr.storage` backend. The predictive model
never mutates them after creation (spec §8.1: the LLM receives an immutable prediction object).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "dvxr-contracts/1"


def _stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic id from its content — same inputs ⇒ same id (idempotency, reproducibility)."""
    key = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class GenerateRequest:
    """A user's 'Generate' action (spec §2 step 1)."""
    patient_id: str
    report_type: str = "stress_glucose_risk"
    prediction_horizons_minutes: List[int] = field(default_factory=lambda: [30, 60])
    data_cutoff_at: str = ""            # explicit causal cutoff; "" ⇒ 'now' resolved by orchestrator
    requested_at: str = ""
    user_role: str = "researcher"
    tenant_id: str = "default"          # server-derived (from the authenticated principal), not body
    actor_id: str = ""                  # server-derived actor; audited, NOT part of the request id
    question: Optional[str] = None
    idempotency_key: Optional[str] = None
    request_id: str = ""

    def with_request_id(self) -> "GenerateRequest":
        """Return a copy carrying a deterministic request_id derived from the request content."""
        rid = self.request_id or _stable_id(
            "req", self.tenant_id, self.patient_id, self.report_type,
            tuple(self.prediction_horizons_minutes), self.data_cutoff_at, self.idempotency_key or "")
        return GenerateRequest(**{**asdict(self), "request_id": rid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerateRequest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RiskPrediction:
    """An immutable, calibrated prediction (spec §2 step 6). `abstained=True` ⇒ no risk number; the
    LLM/UI must surface the abstention, never invent a probability."""
    request_id: str
    patient_id: str
    report_type: str
    tenant_id: str = "default"                     # owning tenant (for cross-tenant access control)
    risk: Optional[Dict[str, float]] = None       # {"excursion_30m": 0.58, ...} or None if abstained
    risk_category: Optional[str] = None           # low | elevated | high | None
    confidence: Optional[float] = None
    data_quality: str = "unknown"                 # good | acceptable | poor | unusable | unknown
    missing_modalities: List[str] = field(default_factory=list)
    stale_modalities: List[str] = field(default_factory=list)
    ood_score: Optional[float] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    model_version: str = ""
    feature_version: str = ""
    calibration_version: str = ""
    data_cutoff_at: str = ""
    snapshot_id: str = ""                          # links to the reproducible PatientSnapshot (Gate 2)
    prediction_id: str = ""

    def with_prediction_id(self) -> "RiskPrediction":
        pid = self.prediction_id or _stable_id(
            "pred", self.tenant_id, self.request_id, self.patient_id, self.report_type,
            json.dumps(self.risk, sort_keys=True), self.abstained, self.model_version,
            self.feature_version, self.calibration_version, self.data_cutoff_at, self.snapshot_id)
        return RiskPrediction(**{**asdict(self), "prediction_id": pid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskPrediction":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ModelEvidence:
    """Model-derived contribution evidence (spec §2 step 7) — NOT an LLM guess."""
    prediction_id: str
    contributions: Dict[str, float] = field(default_factory=dict)   # modality -> signed contribution
    modality_quality: Dict[str, float] = field(default_factory=dict)
    missing_data_effects: List[str] = field(default_factory=list)
    uncertainty: Optional[float] = None
    ood_indicators: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelEvidence":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ActionDecision:
    """A protocol-controlled next action chosen by the policy engine (spec §14) — the LLM may explain
    it but never selects a different action."""
    action_id: str
    policy_id: str = ""
    policy_version: str = ""
    reason_codes: List[str] = field(default_factory=list)
    requires_clinician_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionDecision":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PatientSnapshot:
    """An immutable temporally-aligned window ending at the request cutoff (spec §2 step 5). Records
    exactly which events + versions produced the prediction, so it is fully reproducible."""
    patient_id: str
    data_cutoff_at: str
    tenant_id: str = "default"
    event_ids: List[str] = field(default_factory=list)
    event_content_hashes: List[str] = field(default_factory=list)
    modalities_present: List[str] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)
    quality_by_modality: Dict[str, float] = field(default_factory=dict)
    feature_version: str = ""
    schema_version: str = ""
    snapshot_id: str = ""

    def with_snapshot_id(self) -> "PatientSnapshot":
        # the id hashes a canonical manifest: tenant+patient+cutoff, the exact event ids AND their
        # content hashes (so a quality/value change changes the id), plus the versions the model saw.
        sid = self.snapshot_id or _stable_id(
            "snap", self.tenant_id, self.patient_id, self.data_cutoff_at,
            tuple(sorted(self.event_ids)), tuple(sorted(self.event_content_hashes)),
            self.feature_version, self.schema_version)
        return PatientSnapshot(**{**asdict(self), "snapshot_id": sid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
