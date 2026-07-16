"""dvxr.retrieval.chunking — section-aware chunking (spec §6).

Clinical notes chunk by section then paragraph (never merging unrelated patients/encounters);
protocols chunk by heading / numbered recommendation. Each chunk keeps its parent metadata plus a
deterministic chunk_id, so a retrieved passage always carries its provenance and can be cited.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

_WORD = re.compile(r"\S+")


def _approx_tokens(text: str) -> int:
    return len(_WORD.findall(text))


def _chunk_id(metadata: Dict, idx: int, text: str) -> str:
    key = f"{metadata.get('document_id', '')}|{idx}|{text[:64]}"
    return "chk_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _pack(paragraphs: List[str], target_tokens: int) -> List[str]:
    """Greedily pack paragraphs into chunks up to ~target_tokens (keeps whole paragraphs together)."""
    chunks, cur, cur_tok = [], [], 0
    for p in paragraphs:
        t = _approx_tokens(p)
        if cur and cur_tok + t > target_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_note(text: str, metadata: Dict, target_tokens: int = 400) -> List[Dict]:
    """Chunk a clinical note by section (heading) then paragraph (spec §6: 300-600 tokens).

    ``metadata`` must carry patient/document identity; every chunk inherits it (so a chunk can never be
    confused across patients/encounters) plus a section label and chunk_id."""
    sections = re.split(r"\n(?=#{1,6}\s|[A-Z][A-Za-z ]{2,40}:\s*\n)", text.strip())
    out: List[Dict] = []
    for section in sections:
        header = section.splitlines()[0].strip(" #:") if section.strip() else ""
        paras = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        for chunk_text in _pack(paras, target_tokens):
            md = {**metadata, "section": metadata.get("section", header)}
            out.append({"chunk_id": _chunk_id(md, len(out), chunk_text),
                        "text": chunk_text, "metadata": md})
    return out


def chunk_protocol(text: str, metadata: Dict, target_tokens: int = 600) -> List[Dict]:
    """Chunk a protocol/guideline by heading and numbered recommendation (spec §6: 400-800 tokens).

    ``metadata`` should carry protocol_id / protocol_version / effective_date / active so retrieval can
    version-filter."""
    parts = re.split(r"\n(?=\s*(?:#{1,6}\s|\d+[.)]\s))", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    out: List[Dict] = []
    for chunk_text in _pack(parts, target_tokens):
        out.append({"chunk_id": _chunk_id(metadata, len(out), chunk_text),
                    "text": chunk_text, "metadata": dict(metadata)})
    return out
