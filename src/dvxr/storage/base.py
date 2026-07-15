"""dvxr.storage.base — repository Protocols (spec §6 recommended storage split).

Each store owns one class of authoritative state. The vector store is a retrieval index, NOT a source
of truth for exact facts (glucose values, medications, probabilities, permissions, audit logs) — those
live in the relational/time-series/prediction/consent/audit stores (spec §4 "what should NOT be stored
in the vector database"). These are structural Protocols: any object with the right methods satisfies
them, so `dvxr.storage.local` (sqlite/flat-file) and a future postgres impl are interchangeable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class PredictionStore(Protocol):
    """Predictions, confidence, model version, explanation evidence (spec §6)."""
    def put(self, prediction: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> str: ...
    def get(self, prediction_id: str) -> Optional[Dict[str, Any]]: ...
    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]: ...
    def latest_for_patient(self, patient_id: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class AuditStore(Protocol):
    """Access, requests, generated reports, acknowledgements, overrides (spec §6)."""
    def append(self, entry: Dict[str, Any]) -> str: ...
    def for_request(self, request_id: str) -> List[Dict[str, Any]]: ...


@runtime_checkable
class ConsentStore(Protocol):
    """Study- and patient-level permitted-use scopes (spec §6). Exact lookup, never the vector store."""
    def set_scope(self, patient_id: str, scope: Dict[str, Any]) -> None: ...
    def get(self, patient_id: str) -> Optional[Dict[str, Any]]: ...
    def check(self, patient_id: str, purpose: str) -> bool: ...


@runtime_checkable
class ModelRegistry(Protocol):
    """Model-version traceability (spec §6, §10)."""
    def register(self, name: str, version: str, meta: Dict[str, Any], *, active: bool = False) -> str: ...
    def active(self, name: str) -> Optional[Dict[str, Any]]: ...
    def get(self, name: str, version: str) -> Optional[Dict[str, Any]]: ...


# --- Protocols the later PRs fill in; defined now so interfaces exist up front (spec §6). ---
@runtime_checkable
class RawStore(Protocol):
    """Raw EEG / PPG / waveform files, exports (object storage / data lake)."""
    def put(self, key: str, data: bytes, meta: Dict[str, Any]) -> str: ...
    def get(self, key: str) -> Optional[bytes]: ...


@runtime_checkable
class EventStore(Protocol):
    """Normalized physiological + CGM events (time-series). Deterministic range queries, not semantic."""
    def append_events(self, events: Any) -> int: ...
    def window(self, patient_id: str, start: str, end: str) -> Any: ...


@runtime_checkable
class ClinicalStore(Protocol):
    """Medications, diagnoses, encounters, labs, patient metadata (relational)."""
    def facts(self, patient_id: str, as_of: str) -> Dict[str, Any]: ...


@runtime_checkable
class FeatureStore(Protocol):
    """Versioned windows, embeddings, derived features (spec §6, §10)."""
    def put_features(self, key: str, features: Any, version: str) -> str: ...
    def get_features(self, key: str, version: str) -> Any: ...


@runtime_checkable
class VectorStore(Protocol):
    """Text embeddings + semantically searchable documents ONLY (spec §4). Metadata filter BEFORE
    nearest-neighbour search; never the source of truth for exact facts."""
    def index(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> str: ...
    def search(self, embedding: List[float], *, filters: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]: ...
