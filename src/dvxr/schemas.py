from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


REQUIRED_EVENT_COLUMNS = [
    "subject_id",
    "session_id",
    "timestamp_utc",
    "source_system",
    "device",
    "modality",
    "channel",
    "value",
    "unit",
    "sampling_rate_hz",
    "quality_flag",
    "label_name",
    "label_value",
]

# ----- production event contract (spec §5) — an ADDITIVE provenance layer on top of the 13-column
#       floor. `validate_events` stays a pure 13-column validator (see test_schema_flexible); the
#       provenance columns are attached by `enrich_provenance` in the ingestion path (PR3), never by
#       every loader. This keeps public component loaders untouched while giving every event that
#       enters the *production* system identity, timing, provenance, quality, and consent metadata.
SCHEMA_VERSION = "dvxr-events/2"

PROVENANCE_COLUMNS = [
    "event_id",              # deterministic content hash — makes ingestion idempotent
    "tenant_id",             # organization / study tenant
    "patient_id",            # clinical-record namespace (NOT a medical-record number)
    "observed_at_utc",       # physiological time (== timestamp_utc unless a converter knows better)
    "ingested_at_utc",       # system arrival time (stamped by ingestion; "" until then)
    "source_record_id",      # id of the source row/record in its origin system
    "source_file_hash",      # hash of the source file (idempotency + lineage)
    "quality_score",         # 0..1 continuous quality
    "quality_status",        # good | acceptable | poor | unusable
    "quality_reasons",       # JSON list of reason codes
    "consent_scope",         # permitted-use scope for this event
    "access_scope",          # who may read it
    "schema_version",        # event-contract version
    "preprocessing_version", # approved preprocessing version that produced derived values
    "converter_version",     # source connector/converter version
]

# quality_flag (free-text legacy) -> (score, status). Unknown flags default to a mid "acceptable".
_QUALITY_FLAG_MAP = {
    "good": (1.0, "good"), "ok": (0.9, "good"), "clean": (1.0, "good"),
    "acceptable": (0.75, "acceptable"), "fair": (0.6, "acceptable"),
    "poor": (0.4, "poor"), "noisy": (0.4, "poor"), "artifact": (0.3, "poor"),
    "bad": (0.1, "unusable"), "unusable": (0.0, "unusable"), "": (0.5, "acceptable"),
}


def _quality_from_flag(flag: str) -> tuple[float, str]:
    return _QUALITY_FLAG_MAP.get(str(flag).strip().lower(), (0.5, "acceptable"))


def _event_id(row: pd.Series) -> str:
    """Deterministic content hash of the identity+value fields — same event ⇒ same id (idempotent)."""
    key = "|".join(str(row[c]) for c in
                   ["subject_id", "session_id", "timestamp_utc", "modality", "channel", "value"])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DataSummary:
    rows: int
    subjects: int
    sessions: int
    modalities: list[str]
    devices: list[str]
    label_values: list[str]


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical event table."""
    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    clean = events.copy()
    # The 13 canonical columns are a required floor, not an exact set: a loader may carry
    # dataset-specific extra columns (e.g. glucose_source=libre|dexcom, meal_photo_path,
    # Fitbit sub-metrics). Keep the required columns first, then preserve any extras as-is.
    extra_cols = [c for c in clean.columns if c not in REQUIRED_EVENT_COLUMNS]
    clean = clean[REQUIRED_EVENT_COLUMNS + extra_cols]
    clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean["sampling_rate_hz"] = pd.to_numeric(clean["sampling_rate_hz"], errors="coerce")

    if clean["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc contains invalid timestamps")
    if clean["value"].isna().any():
        raise ValueError("value contains non-numeric entries")
    if clean["sampling_rate_hz"].isna().any():
        raise ValueError("sampling_rate_hz contains non-numeric entries")

    text_cols = [
        "subject_id",
        "session_id",
        "source_system",
        "device",
        "modality",
        "channel",
        "unit",
        "quality_flag",
        "label_name",
        "label_value",
    ]
    for col in text_cols:
        clean[col] = clean[col].fillna("").astype(str)

    clean = clean.sort_values(["subject_id", "session_id", "timestamp_utc", "modality", "channel"])
    return clean.reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> DataSummary:
    clean = validate_events(events)
    return DataSummary(
        rows=len(clean),
        subjects=clean["subject_id"].nunique(),
        sessions=clean[["subject_id", "session_id"]].drop_duplicates().shape[0],
        modalities=sorted(clean["modality"].unique().tolist()),
        devices=sorted(clean["device"].unique().tolist()),
        label_values=sorted([x for x in clean["label_value"].unique().tolist() if x]),
    )


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return frame


def enrich_provenance(
    events: pd.DataFrame,
    *,
    tenant_id: str = "",
    patient_id_map: Optional[Dict[str, str]] = None,
    consent_scope: str = "unspecified",
    access_scope: str = "unspecified",
    ingested_at_utc: str = "",
    source_record_id_col: Optional[str] = None,
    source_file_hash: str = "",
    preprocessing_version: str = "",
    converter_version: str = "",
) -> pd.DataFrame:
    """Attach the production provenance layer (spec §5) to canonical events.

    Base 13-column validation runs first. Provenance columns are then added *without removing or
    overwriting* any the caller already supplied — a converter that knows a better ``observed_at_utc``
    or a real ``patient_id`` wins. ``event_id`` is a deterministic content hash, so re-ingesting the
    same rows yields the same ids (idempotency). Defaults are deterministic (``ingested_at_utc`` is a
    passed value, not a wall-clock read) so the function is test-reproducible.
    """
    clean = validate_events(events)
    n = len(clean)

    def _fill(col: str, value):
        if col not in clean.columns:
            clean[col] = value

    _fill("event_id", [_event_id(r) for _, r in clean.iterrows()])
    _fill("tenant_id", tenant_id)
    if patient_id_map is not None:
        _fill("patient_id", clean["subject_id"].map(lambda s: patient_id_map.get(s, s)))
    else:
        _fill("patient_id", clean["subject_id"])
    _fill("observed_at_utc", clean["timestamp_utc"])
    _fill("ingested_at_utc", ingested_at_utc)
    if source_record_id_col and source_record_id_col in clean.columns:
        _fill("source_record_id", clean[source_record_id_col].astype(str))
    else:
        _fill("source_record_id", clean["event_id"])
    _fill("source_file_hash", source_file_hash)

    q = clean["quality_flag"].map(_quality_from_flag)
    _fill("quality_score", q.map(lambda t: t[0]))
    _fill("quality_status", q.map(lambda t: t[1]))
    _fill("quality_reasons", "[]")
    _fill("consent_scope", consent_scope)
    _fill("access_scope", access_scope)
    _fill("schema_version", SCHEMA_VERSION)
    _fill("preprocessing_version", preprocessing_version)
    _fill("converter_version", converter_version)

    ordered = REQUIRED_EVENT_COLUMNS + PROVENANCE_COLUMNS
    extras = [c for c in clean.columns if c not in ordered]
    return clean[ordered + extras].reset_index(drop=True)


def validate_provenanced_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the full production event contract: the 13-column floor PLUS every provenance column.

    Use this at production ingestion boundaries; public component loaders keep using ``validate_events``.
    """
    clean = validate_events(events)
    missing = [c for c in PROVENANCE_COLUMNS if c not in clean.columns]
    if missing:
        raise ValueError(f"Missing provenance columns: {missing}")
    clean["quality_score"] = pd.to_numeric(clean["quality_score"], errors="coerce")
    if clean["quality_score"].isna().any():
        raise ValueError("quality_score contains non-numeric entries")
    for col in PROVENANCE_COLUMNS:
        if col != "quality_score":
            clean[col] = clean[col].fillna("").astype(str)
    ordered = REQUIRED_EVENT_COLUMNS + PROVENANCE_COLUMNS
    extras = [c for c in clean.columns if c not in ordered]
    return clean[ordered + extras].reset_index(drop=True)
