"""dvxr.serve.auth — server-side principal + authorization (Gate 6, spec §2 step 2).

The API must NOT trust ``user_role``/``tenant`` from the request body — a caller could self-assert any
role or reach another patient. Instead the server derives a :class:`Principal` from an API key
(``X-API-Key`` header) it holds in a registry, and authorizes each patient access against that
principal's scope. This is a lightweight local implementation of the Protocol boundary the spec
describes; a production deployment swaps the registry for a real identity provider without touching the
call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional


class AuthError(PermissionError):
    """Authentication failed (no/unknown credential)."""


class AuthorizationError(PermissionError):
    """The principal is authenticated but not permitted for this patient/action."""


@dataclass(frozen=True)
class Principal:
    """An authenticated actor. ``patient_scope`` is the set of patient ids this actor may access, or
    ``"*"`` for all patients within its tenant."""
    actor_id: str
    role: str                                  # researcher | clinician | participant | admin
    tenant_id: str
    patient_scope: object = "*"                # FrozenSet[str] or the literal "*"

    def may_access(self, patient_id: str) -> bool:
        return self.patient_scope == "*" or patient_id in self.patient_scope


#: A default principal for explicitly-unsafe local/demo runs (never used when a registry is provided).
DEV_PRINCIPAL = Principal(actor_id="dev", role="researcher", tenant_id="dev", patient_scope="*")


def authenticate(api_key: Optional[str], principals: Optional[Dict[str, Principal]],
                 *, unsafe_dev: bool = False) -> Principal:
    """Resolve an API key to its Principal. Fail-closed: an unknown/missing key raises AuthError,
    UNLESS ``unsafe_dev`` is set (local demo only) in which case the DEV_PRINCIPAL is returned."""
    if principals:
        if not api_key or api_key not in principals:
            raise AuthError("missing or unknown API key")
        return principals[api_key]
    if unsafe_dev:
        return DEV_PRINCIPAL
    raise AuthError("no principal registry configured and unsafe_dev is off — refusing to serve")


def authorize(principal: Principal, patient_id: str, action: str = "generate_risk_report",
              *, record_tenant: Optional[str] = None) -> None:
    """Raise AuthorizationError unless ``principal`` may perform ``action`` on ``patient_id``. When a
    stored record's tenant is known (``record_tenant``), cross-tenant access is refused outright."""
    if record_tenant is not None and record_tenant != principal.tenant_id:
        raise AuthorizationError(
            f"principal {principal.actor_id!r} (tenant {principal.tenant_id!r}) may not access a "
            f"record in tenant {record_tenant!r}")
    if not principal.may_access(patient_id):
        raise AuthorizationError(
            f"principal {principal.actor_id!r} is not authorized for patient {patient_id!r}")


def build_principals(spec: Dict[str, dict]) -> Dict[str, Principal]:
    """Build an API-key -> Principal registry from a plain dict (e.g. loaded from config/env)."""
    out: Dict[str, Principal] = {}
    for key, p in spec.items():
        scope = p.get("patient_scope", "*")
        scope = "*" if scope == "*" else frozenset(scope)
        out[key] = Principal(actor_id=p["actor_id"], role=p.get("role", "researcher"),
                             tenant_id=p.get("tenant_id", "default"), patient_scope=scope)
    return out
