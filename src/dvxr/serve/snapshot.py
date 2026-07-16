"""dvxr.serve.snapshot — the immutable, cutoff-bound PatientSnapshot (spec §2 step 5, Gate 2).

A prediction must be reproducible: given a ``snapshot_id`` plus the model/feature/schema versions, one
can regenerate exactly what the model saw. This module assembles that snapshot from provenance-enriched
events, enforcing the causal cutoff (only events at or before ``data_cutoff_at`` are admitted — a
future observation can never enter the window). It is deterministic and backend-agnostic: it consumes a
list of event dicts (or an EventStore query result) rather than reaching into a particular database.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from dvxr.contracts import PatientSnapshot
from dvxr.schemas import SCHEMA_VERSION


def _observed_at(ev: Mapping) -> Optional[str]:
    return ev.get("observed_at_utc") or ev.get("timestamp_utc") or ev.get("timestamp")


def build_patient_snapshot(
    events: Iterable[Mapping],
    *,
    patient_id: str,
    data_cutoff_at: str,
    feature_version: str = "",
    schema_version: str = SCHEMA_VERSION,
    expected_modalities: Sequence[str] = (),
) -> PatientSnapshot:
    """Assemble a reproducible snapshot for ``patient_id`` at ``data_cutoff_at``.

    Only events belonging to this patient AND observed at or before the cutoff are admitted (causal).
    An event without a parseable ``observed_at_utc`` is treated as undated and excluded (it cannot be
    proven to precede the cutoff). ``expected_modalities`` drives the missing-modality list, so a
    downstream reader knows a modality was expected but not present at the cutoff.
    """
    cutoff = pd.to_datetime(data_cutoff_at, utc=False) if data_cutoff_at else None
    admitted: List[Mapping] = []
    for ev in events:
        if patient_id is not None and str(ev.get("patient_id", patient_id)) != str(patient_id):
            continue
        ts = _observed_at(ev)
        if cutoff is not None:
            if ts is None:
                continue
            try:
                if pd.to_datetime(ts, utc=False) > cutoff:
                    continue        # strictly-future event — excluded
            except (ValueError, TypeError):
                continue
        admitted.append(ev)

    event_ids = sorted(str(ev.get("event_id")) for ev in admitted if ev.get("event_id") is not None)
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
        event_ids=event_ids,
        modalities_present=present,
        missing_modalities=missing,
        quality_by_modality=quality_by_modality,
        feature_version=feature_version,
        schema_version=schema_version,
    ).with_snapshot_id()
