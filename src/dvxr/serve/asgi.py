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

# Module-level so FastAPI can resolve the stringized `Request` annotation on index() (this file uses
# `from __future__ import annotations`; hints resolve against module globals, not local scope). fastapi
# is a core dependency, so importing it at module load is expected for this wrapper module.
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

#: Machine-readable landing payload (also the source of truth for the HTML page below).
_LANDING = {
    "product": "DVXR NeuroGlycemic Sentinel",
    "stage": "research — decision-support, NOT a diagnosis or medical device",
    "disclaimer": "A raised risk is a prompt to consult a qualified clinician, never a conclusion.",
    "honesty": ("The fused stress_glucose_risk report abstains by construction — there is no "
                "synchronized EEG+CGM data, so no fused model and no fabricated number is ever served. "
                "Only single-modality CGM report types can return a value, and only when a committed CGM "
                "artifact is provisioned."),
    "endpoints": {
        "GET /health": "liveness + disclaimer (no auth)",
        "GET /docs": "interactive API docs",
        "POST /v1/risk-reports": "generate a risk report (X-API-Key required; consent enforced)",
        "GET /v1/predictions/{id}": "retrieve a persisted report",
        "GET|POST /v1/alerts/{id}[/acknowledge|dismiss|escalate]": "alert lifecycle",
    },
    "auth": "Patient endpoints require an X-API-Key; requests without consent fail closed (403).",
}

# Self-contained HTML landing page (charset asserted, so em dashes render correctly instead of
# mojibake). Theme-aware via prefers-color-scheme. No external assets — CSP-safe.
_LANDING_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVXR NeuroGlycemic Sentinel</title>
<style>
  :root{--bg:#f7f8fa;--card:#fff;--fg:#1c2024;--mut:#5b6572;--line:#e4e7ec;--accent:#0b6b5b;--warn:#8a5a00;--warnbg:#fff7e6}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161b22;--fg:#e6edf3;--mut:#9aa4b2;--line:#2a313c;--accent:#3fb9a3;--warn:#e2b341;--warnbg:#241d08}}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:40px 20px 64px}
  .tag{display:inline-block;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);border-radius:999px;padding:3px 10px;margin-bottom:14px}
  h1{font-size:26px;margin:0 0 6px}
  .sub{color:var(--mut);margin:0 0 22px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
  .warn{background:var(--warnbg);border-color:transparent;color:var(--warn)}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:0 0 10px}
  table{width:100%;border-collapse:collapse;font-size:14.5px}
  td{padding:7px 0;border-top:1px solid var(--line);vertical-align:top}
  td.k{white-space:nowrap;padding-right:16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent)}
  tr:first-child td{border-top:none}
  a{color:var(--accent)}
  .foot{color:var(--mut);font-size:13px;margin-top:26px;text-align:center}
</style></head>
<body><div class="wrap">
  <span class="tag">Research stage</span>
  <h1>DVXR NeuroGlycemic Sentinel</h1>
  <p class="sub">Multimodal glucose-excursion early-warning — decision-support, <b>not a diagnosis or medical device</b>.</p>

  <div class="card warn"><b>Disclaimer.</b> A raised risk is a prompt to consult a qualified clinician, never a conclusion.</div>

  <div class="card">
    <h2>Honesty guardrail</h2>
    <p style="margin:0">The fused <code>stress_glucose_risk</code> report <b>abstains by construction</b> — there is no
    synchronized EEG+CGM data, so no fused model and <b>no fabricated number is ever served</b>. Only single-modality
    CGM report types can return a value, and only when a committed CGM artifact is provisioned.</p>
  </div>

  <div class="card">
    <h2>Endpoints</h2>
    <table>
      <tr><td class="k">GET /health</td><td>liveness + disclaimer (no auth) &middot; <a href="/health">open</a></td></tr>
      <tr><td class="k">GET /docs</td><td>interactive API docs &middot; <a href="/docs">open</a></td></tr>
      <tr><td class="k">POST /v1/risk-reports</td><td>generate a risk report (X-API-Key required; consent enforced)</td></tr>
      <tr><td class="k">GET /v1/predictions/{id}</td><td>retrieve a persisted report</td></tr>
      <tr><td class="k">/v1/alerts/{id}</td><td>alert lifecycle: acknowledge &middot; dismiss &middot; escalate</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Access</h2>
    <p style="margin:0">Patient endpoints require an <code>X-API-Key</code>. Requests without recorded consent
    fail closed (<code>403</code>). No key ⇒ <code>401</code>. There is no anonymous access to patient data.</p>
  </div>

  <p class="foot">DVXR — research-grade, not for clinical use. Machine clients: request <code>Accept: application/json</code> for this page as JSON.</p>
</div></body></html>
"""


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

    @app.get("/", include_in_schema=False)
    def index(request: Request):
        """A human-friendly landing so the bare URL isn't a bare 404. Registered BEFORE the mount below
        so it wins for exactly "/"; every other path falls through to the product app. Content-negotiated:
        HTML for browsers (charset asserted → no mojibake), JSON for API clients that ask for it."""
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse(_LANDING)
        return HTMLResponse(_LANDING_HTML)

    app.mount("/", product)                              # mount the Starlette product app unchanged
    return app


#: the ASGI app FastAPI Cloud deploys: `fastapi deploy` / `uvicorn dvxr.serve.asgi:app`.
app = build_app()
