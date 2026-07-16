"""FastAPI wrapper so the DVXR product API deploys on FastAPI Cloud (``fastapi deploy``).

The product API itself is a Starlette app (:func:`dvxr.sentinel.create_product_api`). FastAPI Cloud
deploys a FastAPI ``app`` object, so this thin module wraps the Starlette product app in a FastAPI
instance — **no routes are re-modeled**, the Starlette app is mounted as-is.

Secure-by-default configuration via environment variables (all optional):

    DVXR_DB_PATH          sqlite path for the persistent stores (default ``dvxr.db`` — NOT ``:memory:``,
                          so predictions/alerts/audit survive restarts).
    DVXR_ARTIFACT_ROOT    committed model-artifact root (default ``artifacts``). CGM report types return
                          real predictions only when an artifact is present here AND registered in
                          DVXR_DB_PATH (run ``scripts/build_cgm_artifact.py`` against the same db + root).
    DVXR_API_KEY          an ``X-API-Key`` that maps to a researcher principal. If unset AND
                          DVXR_UNSAFE_DEV is not ``1``, the ``/v1`` endpoints fail closed (401) — there
                          is no anonymous access to patient endpoints.
    DVXR_UNSAFE_DEV=1     local demo only: disables auth. NEVER set this in a real deployment.
    DVXR_REQUIRE_CONSENT=0  disable consent enforcement (demo only; default is ON).

Honesty: the fused ``stress_glucose_risk`` report **abstains by construction** — there is no
synchronized EEG+CGM data and therefore no fused artifact. Only the single-modality CGM report types can
return a number, and only when a committed CGM artifact is provisioned.
"""
from __future__ import annotations

import os


def _principals():
    """Build the X-API-Key → Principal map from env. Returns None when no key is set (the API then fails
    closed on /v1 unless DVXR_UNSAFE_DEV=1)."""
    key = os.environ.get("DVXR_API_KEY")
    if not key:
        return None
    from dvxr.serve.auth import Principal, Role
    tenant = os.environ.get("DVXR_TENANT", "default")
    return {key: Principal(actor_id="deploy", role=Role.RESEARCHER, tenant_id=tenant, patient_scope="*")}


def build_app():
    """Construct the FastAPI app wrapping the Starlette Sentinel product API."""
    from fastapi import FastAPI

    from dvxr.sentinel import create_product_api

    product = create_product_api(
        db_path=os.environ.get("DVXR_DB_PATH", "dvxr.db"),
        artifact_root=os.environ.get("DVXR_ARTIFACT_ROOT", "artifacts"),
        require_consent=os.environ.get("DVXR_REQUIRE_CONSENT", "1") != "0",
        principals=_principals(),
        unsafe_dev=os.environ.get("DVXR_UNSAFE_DEV") == "1",
    )
    app = FastAPI(
        title="DVXR NeuroGlycemic Sentinel (research-stage)",
        version="0.1.0",
        description=("Research-grade decision-support, not a diagnosis. The fused stress-glucose report "
                     "abstains by construction (no synchronized EEG+CGM data); only single-modality CGM "
                     "report types can return a number."),
    )
    app.mount("/", product)                              # mount the Starlette product app unchanged
    return app


#: the ASGI app FastAPI Cloud deploys: `fastapi deploy` / `uvicorn dvxr.serve.asgi:app`.
app = build_app()
