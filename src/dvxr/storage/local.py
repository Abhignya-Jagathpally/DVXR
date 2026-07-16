"""dvxr.storage.local — sqlite-backed local implementations of the storage Protocols.

Deliberately small and dependency-free (stdlib `sqlite3` + `json`): enough to run the Generate
lifecycle (PR6) and safety/retrieval (PR7) on one machine for hundreds of participants (spec §10). A
single sqlite file (or `:memory:`) holds predictions, audit, consent, and the model registry. Ordering
is by a monotonic autoincrement rowid — no wall-clock reads — so behaviour is deterministic and
idempotency is exact (a repeated idempotency key returns the first stored prediction).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from dvxr.contracts import _stable_id


def _parse_ts(value):
    """Parse an ISO-8601 instant to a comparable datetime, or None if absent/unparseable. Accepts a
    trailing 'Z'. Used for consent temporal validity — never reads the wall clock."""
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    # normalize to UTC-naive so aware/naive instants compare without a TypeError
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT DEFAULT 'default',
    prediction_id TEXT,
    patient_id TEXT,
    request_id TEXT,
    idempotency_key TEXT,
    payload TEXT NOT NULL,
    UNIQUE(tenant_id, prediction_id),
    UNIQUE(tenant_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id TEXT UNIQUE,
    request_id TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consent (
    tenant_id TEXT DEFAULT 'default',
    patient_id TEXT,
    scope TEXT NOT NULL,
    PRIMARY KEY (tenant_id, patient_id)
);
CREATE TABLE IF NOT EXISTS models (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, version TEXT, active INTEGER DEFAULT 0, meta TEXT NOT NULL,
    UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    event_id TEXT,
    observed_at_utc TEXT,
    payload TEXT NOT NULL,
    UNIQUE(tenant_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_events_scope ON events(tenant_id, patient_id, observed_at_utc);
"""


def _connect(path: str = ":memory:") -> sqlite3.Connection:
    # check_same_thread=False so the single shared connection can serve an async server's threadpool
    # handlers (Starlette/uvicorn). Access stays effectively serialized in the single-process research
    # deployment; a multi-process deployment swaps this local impl for a real DB behind the Protocol.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


class LocalPredictionStore:
    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def put(self, prediction: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> str:
        # idempotency + rows are TENANT-SCOPED so one tenant's id/key never resolves another's payload.
        tenant = str(prediction.get("tenant_id", "default"))
        if idempotency_key:
            existing = self.get_by_idempotency_key(idempotency_key, tenant_id=tenant)
            if existing is not None:
                return existing["prediction_id"]
        pid = prediction.get("prediction_id") or _stable_id(
            "pred", tenant, prediction.get("request_id"), prediction.get("patient_id"),
            json.dumps(prediction.get("risk"), sort_keys=True))
        prediction = {**prediction, "prediction_id": pid}
        self._c.execute(
            "INSERT OR IGNORE INTO predictions(tenant_id, prediction_id, patient_id, request_id, "
            "idempotency_key, payload) VALUES (?,?,?,?,?,?)",
            (tenant, pid, prediction.get("patient_id"), prediction.get("request_id"),
             idempotency_key, json.dumps(prediction)))
        self._c.commit()
        return pid

    def get(self, prediction_id: str, *, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if tenant_id is not None:
            row = self._c.execute(
                "SELECT payload FROM predictions WHERE prediction_id=? AND tenant_id=?",
                (prediction_id, str(tenant_id))).fetchone()
        else:
            row = self._c.execute("SELECT payload FROM predictions WHERE prediction_id=?",
                                  (prediction_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def get_by_idempotency_key(self, idempotency_key: str, *,
                               tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if tenant_id is not None:
            row = self._c.execute(
                "SELECT payload FROM predictions WHERE idempotency_key=? AND tenant_id=?",
                (idempotency_key, str(tenant_id))).fetchone()
        else:
            row = self._c.execute("SELECT payload FROM predictions WHERE idempotency_key=?",
                                  (idempotency_key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def latest_for_patient(self, patient_id: str, *,
                           tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if tenant_id is not None:
            row = self._c.execute(
                "SELECT payload FROM predictions WHERE patient_id=? AND tenant_id=? "
                "ORDER BY seq DESC LIMIT 1", (patient_id, str(tenant_id))).fetchone()
        else:
            row = self._c.execute(
                "SELECT payload FROM predictions WHERE patient_id=? ORDER BY seq DESC LIMIT 1",
                (patient_id,)).fetchone()
        return json.loads(row["payload"]) if row else None


class LocalAuditStore:
    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def append(self, entry: Dict[str, Any]) -> str:
        seq = self._c.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM audit").fetchone()["n"]
        aid = entry.get("audit_id") or _stable_id("aud", entry.get("request_id"), seq,
                                                  json.dumps(entry, sort_keys=True))
        self._c.execute("INSERT INTO audit(audit_id, request_id, payload) VALUES (?,?,?)",
                       (aid, entry.get("request_id"), json.dumps({**entry, "audit_id": aid})))
        self._c.commit()
        return aid

    def for_request(self, request_id: str) -> List[Dict[str, Any]]:
        rows = self._c.execute("SELECT payload FROM audit WHERE request_id=? ORDER BY seq",
                              (request_id,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]


class LocalConsentStore:
    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def set_scope(self, patient_id: str, scope: Dict[str, Any], *, tenant_id: str = "default") -> None:
        # consent is TENANT-SCOPED: tenant B recording consent for a patient id never satisfies a
        # consent check made under tenant A for the same raw patient id.
        self._c.execute("INSERT OR REPLACE INTO consent(tenant_id, patient_id, scope) VALUES (?,?,?)",
                       (str(tenant_id), patient_id, json.dumps(scope)))
        self._c.commit()

    def get(self, patient_id: str, *, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        row = self._c.execute("SELECT scope FROM consent WHERE tenant_id=? AND patient_id=?",
                              (str(tenant_id), patient_id)).fetchone()
        return json.loads(row["scope"]) if row else None

    def check(self, patient_id: str, purpose: str, *, tenant_id: str = "default",
              as_of: Optional[str] = None, modality: Optional[str] = None,
              study_id: Optional[str] = None) -> bool:
        """Fail-closed consent check across every recorded dimension (spec §2, §18). Denies unless: a
        scope exists AND is not ``revoked`` AND covers ``purpose`` AND (when a ``modality``/``study_id``
        is requested and the scope restricts them) covers it AND (when ``as_of`` is given) the scope is
        temporally valid at that instant — inside ``effective_from``..``effective_until``/``expires_at``.
        ``as_of`` is the request's causal cutoff (no wall-clock read). A simple ``{"purposes": [...]}``
        scope keeps working (no temporal/modality/study restriction ⇒ those dimensions are unrestricted)."""
        scope = self.get(patient_id, tenant_id=tenant_id)
        if not scope:
            return False               # no recorded consent ⇒ deny (fail-closed)
        if scope.get("revoked"):
            return False               # explicit revocation ⇒ deny
        purposes = scope.get("purposes", [])
        if not (purpose in purposes or "all" in purposes):
            return False
        mods = scope.get("modalities")
        if modality is not None and mods is not None and not (modality in mods or "all" in mods):
            return False               # consent does not cover the requested modality
        study = scope.get("study_id")
        if study_id is not None and study is not None and str(study) != str(study_id):
            return False               # consent is scoped to a different study
        if as_of:
            t = _parse_ts(as_of)
            ef = _parse_ts(scope.get("effective_from"))
            eu = _parse_ts(scope.get("effective_until") or scope.get("expires_at"))
            if t is None:
                return False           # a temporal check was requested but the instant is unparseable
            if ef is not None and t < ef:
                return False           # not yet in effect
            if eu is not None and t > eu:
                return False           # expired / past the effective window
        return True


class LocalModelRegistry:
    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def register(self, name: str, version: str, meta: Dict[str, Any], *, active: bool = False) -> str:
        if active:
            self._c.execute("UPDATE models SET active=0 WHERE name=?", (name,))
        self._c.execute(
            "INSERT OR REPLACE INTO models(name, version, active, meta) VALUES (?,?,?,?)",
            (name, version, 1 if active else 0, json.dumps(meta)))
        self._c.commit()
        return f"{name}@{version}"

    def active(self, name: str) -> Optional[Dict[str, Any]]:
        row = self._c.execute(
            "SELECT name, version, meta FROM models WHERE name=? AND active=1 ORDER BY seq DESC LIMIT 1",
            (name,)).fetchone()
        if not row:
            return None
        return {"name": row["name"], "version": row["version"], "meta": json.loads(row["meta"])}

    def get(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        row = self._c.execute("SELECT name, version, meta FROM models WHERE name=? AND version=?",
                              (name, version)).fetchone()
        if not row:
            return None
        return {"name": row["name"], "version": row["version"], "meta": json.loads(row["meta"])}


class LocalEventStore:
    """Tenant+patient-scoped normalized events (spec §6 EventStore). ``window`` returns ONLY events for
    the given (tenant, patient) in ``[start, end]`` — the deterministic range query the snapshot builder
    consumes. Identity is stored explicitly and every read is filtered by it (Gate A)."""

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def append_events(self, events: List[Dict[str, Any]]) -> int:
        n = 0
        for ev in events or []:
            tenant = ev.get("tenant_id")
            patient = ev.get("patient_id")
            eid = ev.get("event_id")
            if not tenant or not patient or not eid:
                continue                              # reject identity-less events (never store them)
            obs = ev.get("observed_at_utc") or ev.get("timestamp_utc") or ev.get("timestamp")
            self._c.execute(
                "INSERT OR IGNORE INTO events(tenant_id, patient_id, event_id, observed_at_utc, payload) "
                "VALUES (?,?,?,?,?)",
                (str(tenant), str(patient), str(eid), obs, json.dumps(ev)))
            n += 1
        self._c.commit()
        return n

    def window(self, patient_id: str, start: Optional[str], end: Optional[str], *,
               tenant_id: str = "default") -> List[Dict[str, Any]]:
        """Deterministic range query, TENANT+PATIENT scoped. ``start``/``end`` are ISO strings (or None
        for open-ended). Rows are ordered by observed time then insertion order for reproducibility."""
        clauses = ["tenant_id=?", "patient_id=?"]
        params: List[Any] = [str(tenant_id), str(patient_id)]
        if start is not None:
            clauses.append("(observed_at_utc IS NULL OR observed_at_utc >= ?)")
            params.append(str(start))
        if end is not None:
            clauses.append("(observed_at_utc IS NULL OR observed_at_utc <= ?)")
            params.append(str(end))
        rows = self._c.execute(
            f"SELECT payload FROM events WHERE {' AND '.join(clauses)} ORDER BY observed_at_utc, seq",
            params).fetchall()
        return [json.loads(r["payload"]) for r in rows]


class LocalStores(tuple):
    """Backward-compatible: unpacks as the historical 4-tuple (predictions, audit, consent, models) but
    also exposes ``.events`` (and named fields) for callers that want the full stack."""
    def __new__(cls, predictions, audit, consent, models, events):
        self = super().__new__(cls, (predictions, audit, consent, models))
        self.predictions = predictions
        self.audit = audit
        self.consent = consent
        self.models = models
        self.events = events
        return self


def open_local_stores(path: str = ":memory:") -> "LocalStores":
    """Open all local stores on one sqlite connection (a file path, or in-memory for tests).

    Returns a :class:`LocalStores` that still unpacks as the historical 4-tuple
    ``(predictions, audit, consent, models)`` — existing call sites are unchanged — while also exposing
    ``.events`` (a :class:`LocalEventStore` on the same connection) and named attributes."""
    conn = _connect(path)
    return LocalStores(LocalPredictionStore(conn), LocalAuditStore(conn),
                       LocalConsentStore(conn), LocalModelRegistry(conn), LocalEventStore(conn))
