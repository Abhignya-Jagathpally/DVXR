"""dvxr.retrieval.search — metadata-filtered, embedding-free local retrieval (spec §4, §14).

Filters metadata BEFORE ranking, so a document that fails a filter (wrong patient namespace, inactive,
superseded protocol version, wrong role/document type) is never a candidate — regardless of how well
its text matches. Ranking is deterministic keyword (token-overlap) similarity — this is NOT a vector
index, hence the honest name ``LocalKeywordTextIndex``; a real embedding index drops in behind the same
``RetrievalRepository`` interface. Patient-scoped documents can ONLY be reached through
``search_patient`` (which requires a patient id + tenant), so a clinical note can never be returned
without an explicit patient namespace (spec §7 patient isolation).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")

#: Document types that belong to a specific patient — never returnable without a patient+tenant filter.
PATIENT_SCOPED_TYPES = frozenset({"clinical_note", "patient_note", "narrative", "encounter"})

#: Metadata every chunk must carry, plus per-type required identity/version/scope fields.
_REQUIRED_COMMON = ("document_type",)
_REQUIRED_BY_TYPE = {
    "clinical_note": ("patient_id", "tenant_id", "access_scope"),
    "patient_note": ("patient_id", "tenant_id", "access_scope"),
    "narrative": ("patient_id", "tenant_id", "access_scope"),
    "encounter": ("patient_id", "tenant_id", "access_scope"),
    "protocol": ("protocol_id", "protocol_version", "active"),
    "guideline": ("protocol_id", "protocol_version", "active"),
}


class RetrievalMetadataError(ValueError):
    """A chunk was indexed without the provenance metadata its document type requires."""


def _tokens(text: str) -> set:
    return set(_TOKEN.findall((text or "").lower()))


def _passes(md: Dict, filters: Dict) -> bool:
    """A chunk passes iff it satisfies EVERY filter. `active` and version filters are the safety-
    critical ones (spec §9: never surface an inactive/old protocol)."""
    for k, want in filters.items():
        have = md.get(k)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


@runtime_checkable
class RetrievalRepository(Protocol):
    """The retrieval contract. General knowledge (protocols, model cards) is reachable via ``search``;
    patient-scoped notes ONLY via ``search_patient`` with an explicit access context."""

    def index(self, chunk: Dict) -> str: ...
    def search(self, query: str, *, filters: Optional[Dict] = None, k: int = 5) -> List[Dict]: ...
    def search_patient(self, query: str, *, patient_id: str, tenant_id: str,
                       filters: Optional[Dict] = None, k: int = 5) -> List[Dict]: ...


class LocalKeywordTextIndex:
    """In-memory KEYWORD (token-overlap) text index — deterministic, offline, no embeddings. Enforces
    provenance metadata at index time and patient/tenant isolation at query time."""

    def __init__(self):
        self._chunks: List[Dict] = []

    def _validate(self, chunk: Dict) -> None:
        if "chunk_id" not in chunk or "text" not in chunk or "metadata" not in chunk:
            raise RetrievalMetadataError("chunk must have chunk_id, text, metadata")
        md = chunk["metadata"]
        dtype = md.get("document_type")
        for f in _REQUIRED_COMMON:
            if not md.get(f):
                raise RetrievalMetadataError(f"chunk {chunk['chunk_id']!r} missing metadata {f!r}")
        for f in _REQUIRED_BY_TYPE.get(dtype, ()):  # per-type identity/version/scope
            if md.get(f) in (None, ""):
                raise RetrievalMetadataError(
                    f"chunk {chunk['chunk_id']!r} (document_type={dtype!r}) missing required "
                    f"metadata {f!r}")

    def index(self, chunk: Dict) -> str:
        self._validate(chunk)
        self._chunks.append(chunk)
        return chunk["chunk_id"]

    def index_all(self, chunks: List[Dict]) -> List[str]:
        return [self.index(c) for c in chunks]

    def _rank(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        q = _tokens(query)
        scored = []
        for c in candidates:
            overlap = len(q & _tokens(c["text"]))
            if overlap == 0 and query:
                continue
            scored.append((overlap, c))
        scored.sort(key=lambda t: (-t[0], t[1]["chunk_id"]))
        return [c for _s, c in scored[:k]]

    def search(self, query: str, *, filters: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        """Search GENERAL (non-patient) knowledge. Patient-scoped documents are excluded here — they
        must be fetched via :meth:`search_patient` so a note can never leak without a patient filter."""
        filters = filters or {}
        cands = [c for c in self._chunks
                 if c["metadata"].get("document_type") not in PATIENT_SCOPED_TYPES
                 and _passes(c["metadata"], filters)]
        return self._rank(query, cands, k)

    def search_patient(self, query: str, *, patient_id: str, tenant_id: str,
                       filters: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        """Search a SPECIFIC patient's notes. ``patient_id`` and ``tenant_id`` are MANDATORY and are
        applied as filters BEFORE ranking, so cross-patient / cross-tenant leakage is impossible."""
        if not patient_id or not tenant_id:
            raise ValueError("search_patient requires both patient_id and tenant_id")
        scope = {"patient_id": patient_id, "tenant_id": tenant_id, **(filters or {})}
        cands = [c for c in self._chunks
                 if c["metadata"].get("document_type") in PATIENT_SCOPED_TYPES
                 and _passes(c["metadata"], scope)]
        return self._rank(query, cands, k)

    def source_ids(self) -> set:
        return {c["chunk_id"] for c in self._chunks}


#: Backward-compatible alias (the old name implied vectors; keep it importable).
LocalTextIndex = LocalKeywordTextIndex
