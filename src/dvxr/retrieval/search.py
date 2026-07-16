"""dvxr.retrieval.search — metadata-filtered, embedding-free local retrieval (spec §4, §14).

Filters metadata BEFORE ranking, so a document that fails a filter (wrong patient namespace, inactive,
superseded protocol version, wrong role/document type) is never a candidate — regardless of how well
its text matches. Ranking is deterministic keyword (token-overlap) similarity, so the index runs
offline with no model; a real vector index drops in behind the same `index`/`search` interface.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_TOKEN = re.compile(r"[a-z0-9]+")


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


class LocalTextIndex:
    """An in-memory text index. Store text chunks with metadata; search filters then ranks."""

    def __init__(self):
        self._chunks: List[Dict] = []

    def index(self, chunk: Dict) -> str:
        self._chunks.append(chunk)
        return chunk["chunk_id"]

    def index_all(self, chunks: List[Dict]) -> List[str]:
        return [self.index(c) for c in chunks]

    def search(self, query: str, *, filters: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        filters = filters or {}
        q = _tokens(query)
        scored = []
        for c in self._chunks:
            if not _passes(c["metadata"], filters):
                continue                                  # metadata filter BEFORE similarity
            overlap = len(q & _tokens(c["text"]))
            if overlap == 0 and query:
                continue
            scored.append((overlap, c))
        # deterministic: sort by score desc, then chunk_id for stable ties
        scored.sort(key=lambda t: (-t[0], t[1]["chunk_id"]))
        return [c for _s, c in scored[:k]]

    def source_ids(self) -> set:
        return {c["chunk_id"] for c in self._chunks}
