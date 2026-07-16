"""dvxr.serve.snapshot — the immutable, cutoff-bound PatientSnapshot (spec §2 step 5, Gate 2).

A prediction must be reproducible: given a ``snapshot_id`` plus the model/feature/schema versions, one
can regenerate exactly what the model saw. This module assembles that snapshot from provenance-enriched
events, enforcing the causal cutoff (only events at or before ``data_cutoff_at`` are admitted — a
future observation can never enter the window). It is deterministic and backend-agnostic: it consumes a
list of event dicts (or an EventStore query result) rather than reaching into a particular database.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from dvxr.contracts import PatientSnapshot
from dvxr.schemas import SCHEMA_VERSION


def _observed_at(ev: Mapping) -> Optional[str]:
    return ev.get("observed_at_utc") or ev.get("timestamp_utc") or ev.get("timestamp")


def _content_hash(ev: Mapping) -> str:
    key = "|".join(str(ev.get(c, "")) for c in
                   ("event_id", "modality", "channel", "value", "observed_at_utc", "quality_score"))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_patient_snapshot(
    events: Iterable[Mapping],
    *,
    patient_id: str,
    data_cutoff_at: str,
    tenant_id: str = "default",
    feature_version: str = "",
    schema_version: str = SCHEMA_VERSION,
    expected_modalities: Sequence[str] = (),
) -> PatientSnapshot:
    """Assemble a reproducible snapshot for ``tenant_id``/``patient_id`` at ``data_cutoff_at``.

    ISOLATION (Gate A): an event is admitted ONLY if it explicitly belongs to this tenant AND patient
    (identity is never inferred — a missing tenant/patient id is rejected, never treated as a match),
    was observed at or before the cutoff, and carries an immutable ``event_id`` (an id-less event is
    quarantined, so it can neither affect modality presence/quality nor be silently dropped from
    ``event_ids``). The snapshot id hashes tenant+patient+cutoff, the event ids AND their content
    hashes, and the feature/schema versions, so any change to what the model saw changes the id.
    """
    cutoff = pd.to_datetime(data_cutoff_at, utc=True) if data_cutoff_at else None
    admitted: List[Mapping] = []
    for ev in events:
        # identity must be PRESENT and MATCH — never inferred from the request (no fail-open)
        if not ev.get("patient_id") or str(ev["patient_id"]) != str(patient_id):
            continue
        if str(ev.get("tenant_id", "")) != str(tenant_id):
            continue
        if not ev.get("event_id"):                   # quarantine id-less events
            continue
        ts = _observed_at(ev)
        if cutoff is not None:
            if ts is None:
                continue
            try:
                if pd.to_datetime(ts, utc=True) > cutoff:
                    continue        # strictly-future event — excluded
            except (ValueError, TypeError):
                continue
        admitted.append(ev)

    # dedup by event_id (a duplicate id is the same event ingested twice — count it once)
    seen = set()
    deduped = []
    for ev in admitted:
        eid = str(ev["event_id"])
        if eid in seen:
            continue
        seen.add(eid)
        deduped.append(ev)
    admitted = deduped

    event_ids = sorted(str(ev["event_id"]) for ev in admitted)
    content_hashes = sorted(_content_hash(ev) for ev in admitted)
    present: List[str] = sorted({str(ev["modality"]) for ev in admitted if ev.get("modality")})
    missing = sorted(m for m in expected_modalities if m not in present)

    quality_by_modality: Dict[str, float] = {}
    for m in present:
        qs = [float(ev["quality_score"]) for ev in admitted
              if ev.get("modality") == m and ev.get("quality_score") is not None]
        if qs:
            quality_by_modality[m] = round(sum(qs) / len(qs), 6)

    return PatientSnapshot(
        patient_id=patient_id,
        data_cutoff_at=data_cutoff_at,
        tenant_id=tenant_id,
        event_ids=event_ids,
        event_content_hashes=content_hashes,
        modalities_present=present,
        missing_modalities=missing,
        quality_by_modality=quality_by_modality,
        feature_version=feature_version,
        schema_version=schema_version,
    ).with_snapshot_id()
