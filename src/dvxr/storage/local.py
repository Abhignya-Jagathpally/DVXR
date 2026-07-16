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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE,
    patient_id TEXT,
    request_id TEXT,
    idempotency_key TEXT UNIQUE,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id TEXT UNIQUE,
    request_id TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consent (
    patient_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, version TEXT, active INTEGER DEFAULT 0, meta TEXT NOT NULL,
    UNIQUE(name, version)
);
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
        # idempotency: a repeated key returns the ALREADY stored prediction, never a second row.
        if idempotency_key:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing["prediction_id"]
        pid = prediction.get("prediction_id") or _stable_id(
            "pred", prediction.get("request_id"), prediction.get("patient_id"),
            json.dumps(prediction.get("risk"), sort_keys=True))
        prediction = {**prediction, "prediction_id": pid}
        self._c.execute(
            "INSERT OR IGNORE INTO predictions(prediction_id, patient_id, request_id, "
            "idempotency_key, payload) VALUES (?,?,?,?,?)",
            (pid, prediction.get("patient_id"), prediction.get("request_id"),
             idempotency_key, json.dumps(prediction)))
        self._c.commit()
        return pid

    def get(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        row = self._c.execute("SELECT payload FROM predictions WHERE prediction_id=?",
                              (prediction_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        row = self._c.execute("SELECT payload FROM predictions WHERE idempotency_key=?",
                              (idempotency_key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def latest_for_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
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

    def set_scope(self, patient_id: str, scope: Dict[str, Any]) -> None:
        self._c.execute("INSERT OR REPLACE INTO consent(patient_id, scope) VALUES (?,?)",
                       (patient_id, json.dumps(scope)))
        self._c.commit()

    def get(self, patient_id: str) -> Optional[Dict[str, Any]]:
        row = self._c.execute("SELECT scope FROM consent WHERE patient_id=?", (patient_id,)).fetchone()
        return json.loads(row["scope"]) if row else None

    def check(self, patient_id: str, purpose: str) -> bool:
        scope = self.get(patient_id)
        if not scope:
            return False               # no recorded consent ⇒ deny (fail-closed)
        purposes = scope.get("purposes", [])
        return purpose in purposes or "all" in purposes


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


def open_local_stores(path: str = ":memory:") -> Tuple[
        LocalPredictionStore, LocalAuditStore, LocalConsentStore, LocalModelRegistry]:
    """Open all local stores on one sqlite connection (a file path, or in-memory for tests)."""
    conn = _connect(path)
    return (LocalPredictionStore(conn), LocalAuditStore(conn),
            LocalConsentStore(conn), LocalModelRegistry(conn))
