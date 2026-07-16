"""FastAPI Cloud entry point with a user-facing Sentinel web application.

The scientific product API remains the existing, secure-by-default Starlette application from
``dvxr.sentinel.create_product_api``. This wrapper adds only:

* a responsive, same-origin web interface at ``/``;
* static assets at ``/assets``;
* a short-lived HttpOnly UI session that bridges to the existing ``X-API-Key`` contract; and
* developer documentation under ``/developer/docs``.

No prediction, evidence, policy, consent, abstention, or audit behavior is reimplemented here.
The fused ``stress_glucose_risk`` report therefore continues to abstain by construction until a
validated synchronized EEG + wearable + CGM artifact exists.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_UI_COOKIE = "dvxr_ui_session"
_UI_COOKIE_MAX_AGE_SECONDS = 60 * 60


def _principals():
    """Build the existing X-API-Key -> Principal map from deployment environment variables."""
    key = os.environ.get("DVXR_API_KEY")
    if not key:
        return None
    from dvxr.serve.auth import Principal, Role

    tenant = os.environ.get("DVXR_TENANT", "default")
    return {
        key: Principal(
            actor_id=os.environ.get("DVXR_ACTOR_ID", "web-user"),
            role=Role.RESEARCHER,
            tenant_id=tenant,
            patient_scope="*",
        )
    }


def _same_secret(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return hmac.compare_digest(str(left), str(right))


def build_app() -> FastAPI:
    """Construct the FastAPI Cloud application without altering the Sentinel product contract."""
    from dvxr.sentinel import create_product_api

    unsafe_dev = os.environ.get("DVXR_UNSAFE_DEV") == "1"
    api_key = os.environ.get("DVXR_API_KEY")
    web_root = Path(__file__).resolve().parents[1] / "web"
    assets_root = web_root / "assets"

    if not (web_root / "index.html").exists():
        raise RuntimeError(f"Sentinel web assets are missing at {web_root}")

    product = create_product_api(
        db_path=os.environ.get("DVXR_DB_PATH", "dvxr.db"),
        artifact_root=os.environ.get("DVXR_ARTIFACT_ROOT", "artifacts"),
        require_consent=os.environ.get("DVXR_REQUIRE_CONSENT", "1") != "0",
        principals=_principals(),
        unsafe_dev=unsafe_dev,
    )

    app = FastAPI(
        title="DVXR NeuroGlycemic Sentinel",
        version="0.2.0",
        description=(
            "Research-grade multimodal clinical-risk decision support. The predictive service "
            "computes or abstains; the language layer explains verified outputs and never invents risk."
        ),
        docs_url="/developer/docs",
        redoc_url=None,
        openapi_url="/developer/openapi.json",
    )

    # The UI never stores the deployment API key in JavaScript. An authorized user enters the
    # access code once; it is retained only as a secure, HttpOnly, same-site cookie. For /v1 calls,
    # this middleware translates that cookie back into the API's existing X-API-Key contract.
    @app.middleware("http")
    async def ui_session_bridge(request: Request, call_next):
        if request.url.path.startswith("/v1/") and not request.headers.get("X-API-Key"):
            cookie = request.cookies.get(_UI_COOKIE)
            if _same_secret(cookie, api_key):
                headers = list(request.scope.get("headers", []))
                headers.append((b"x-api-key", str(api_key).encode("utf-8")))
                request.scope["headers"] = headers
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response

    app.mount("/assets", StaticFiles(directory=str(assets_root)), name="sentinel-assets")

    @app.get("/", include_in_schema=False)
    async def home():
        response = FileResponse(web_root / "index.html", media_type="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/ui/config", include_in_schema=False)
    async def ui_config():
        return {
            "product": "DVXR NeuroGlycemic Sentinel",
            "research_stage": True,
            "auth_required": bool(api_key) and not unsafe_dev,
            "unsafe_dev": unsafe_dev,
            "consent_required": os.environ.get("DVXR_REQUIRE_CONSENT", "1") != "0",
            "fused_report_status": "abstains_until_synchronized_artifact",
        }

    @app.get("/ui/session", include_in_schema=False)
    async def session_status(request: Request):
        authenticated = unsafe_dev or _same_secret(request.cookies.get(_UI_COOKIE), api_key)
        return {"authenticated": authenticated}

    @app.post("/ui/session", include_in_schema=False)
    async def create_session(request: Request):
        if unsafe_dev:
            return {"authenticated": True, "mode": "unsafe_dev"}
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="The deployment has no DVXR_API_KEY configured; patient endpoints remain closed.",
            )
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Expected a JSON request body.") from exc
        if not _same_secret(str(payload.get("access_code", "")), api_key):
            raise HTTPException(status_code=401, detail="The access code is not valid.")

        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            key=_UI_COOKIE,
            value=api_key,
            max_age=_UI_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        return response

    @app.delete("/ui/session", include_in_schema=False)
    async def delete_session():
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(_UI_COOKIE, path="/", secure=True, httponly=True, samesite="strict")
        return response

    # Mount last so exact UI/static/developer routes above win, while all established product routes
    # remain available at their original paths: /health, /v1/risk-reports, /v1/predictions/..., etc.
    app.mount("/", product)
    return app


app = build_app()
