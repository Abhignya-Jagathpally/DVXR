"""dvxr.retrieval — section-aware chunking + metadata-filtered retrieval (spec §4, §6, §14).

The vector store holds TEXT ONLY (de-identified note sections, approved protocols, model cards,
limitations) — never the source of truth for exact facts (spec §4). Retrieval filters metadata
BEFORE similarity (patient namespace, document type, protocol version, active status) so an inactive
or superseded protocol can never be surfaced (spec §9 "vector retrieval finds old protocol"). This
local implementation is embedding-free (deterministic keyword overlap) so it runs offline; swap in a
real vector index behind the same interface at scale.
"""
from dvxr.retrieval.chunking import chunk_note, chunk_protocol  # noqa: F401
from dvxr.retrieval.search import (  # noqa: F401
    LocalKeywordTextIndex,
    LocalTextIndex,               # backward-compatible alias
    RetrievalMetadataError,
    RetrievalRepository,
)
