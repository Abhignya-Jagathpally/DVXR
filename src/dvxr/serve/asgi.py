"""FastAPI Cloud entry point with same-origin UI and cross-origin artifact support.

The product API remains the existing secure-by-default application returned by
``dvxr.sentinel.create_product_api``. This wrapper adds:

* the same-origin Sentinel web application at ``/``;
* developer documentation under ``/developer/docs``;
* a one-hour HttpOnly session for the same-origin UI; and
* a short-lived, origin-bound bearer-token exchange for trusted external artifacts.

Prediction, evidence, policy, consent, calibration, abstention, and audit behavior remain in the
existing product service. The bridge only authenticates the user-facing clients and forwards their
requests into the established ``X-API-Key`` contract.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_UI_COOKIE = "dvxr_ui_session"
_UI_COOKIE_MAX_AGE_SECONDS = 60 * 60
_DEFAULT_ARTIFACT_TOKEN_TTL_SECONDS = 15 * 60
_DEFAULT_ARTIFACT_ORIGIN_REGEX = (
    r"^https://([a-z0-9-]+\.)?(claude\.ai|claudeusercontent\.com)$"
)


def _principals():
    """Build the existing API-key-to-principal map from deployment configuration."""
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


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _artifact_token_ttl() -> int:
    raw = os.environ.get("DVXR_ARTIFACT_TOKEN_TTL_SECONDS", "")
    try:
        value = int(raw) if raw else _DEFAULT_ARTIFACT_TOKEN_TTL_SECONDS
    except ValueError:
        value = _DEFAULT_ARTIFACT_TOKEN_TTL_SECONDS
    return min(max(value, 60), 60 * 60)


def _sign_artifact_token(*, secret: str, origin: str, actor_id: str) -> tuple[str, int]:
    """Create a compact signed token without introducing an additional JWT dependency."""
    ttl = _artifact_token_ttl()
    now = int(time.time())
    payload = {
        "iss": "dvxr-neuroglycemic-sentinel",
        "sub": actor_id,
        "origin": origin,
        "iat": now,
        "exp": now + ttl,
        "nonce": _b64url_encode(os.urandom(12)),
    }
    encoded_payload = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded_payload}.{_b64url_encode(signature)}", ttl


def _verify_artifact_token(*, token: str, secret: str, request_origin: str | None) -> dict[str, Any]:
    """Validate signature, expiry, issuer, and the browser origin bound to the token."""
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected_signature = hmac.new(
            secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256
        ).digest()
        supplied_signature = _b64url_decode(encoded_signature)
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ValueError("signature")
        payload = json.loads(_b64url_decode(encoded_payload).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="The artifact session token is invalid.") from exc

    now = int(time.time())
    if payload.get("iss") != "dvxr-neuroglycemic-sentinel":
        raise HTTPException(status_code=401, detail="The artifact session token has an invalid issuer.")
    if int(payload.get("exp", 0)) <= now:
        raise HTTPException(status_code=401, detail="The artifact session token has expired.")
    if request_origin and payload.get("origin") != request_origin:
        raise HTTPException(status_code=401, detail="The artifact session token is not valid for this origin.")
    return payload


def _request_origin(request: Request) -> str:
    """Return the browser origin used to bind a token; direct clients get a dedicated marker."""
    origin = request.headers.get("origin")
    if origin:
        return origin.rstrip("/")
    return "direct-client"


def build_app() -> FastAPI:
    """Construct the FastAPI Cloud application without changing scientific product semantics."""
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
        title="NeuroGlycemic Sentinel",
        version="0.3.0",
        description=(
            "Research-grade multimodal clinical-risk decision support. The predictive service "
            "calculates or abstains; the language layer explains verified outputs."
        ),
        docs_url="/developer/docs",
        redoc_url=None,
        openapi_url="/developer/openapi.json",
    )

    exact_origins = [
        origin.strip().rstrip("/")
        for origin in os.environ.get("DVXR_UI_ORIGINS", "https://claude.ai").split(",")
        if origin.strip()
    ]
    origin_regex = os.environ.get("DVXR_UI_ORIGIN_REGEX", _DEFAULT_ARTIFACT_ORIGIN_REGEX)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=exact_origins,
        allow_origin_regex=origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
        expose_headers=["X-DVXR-Request-ID"],
        max_age=600,
    )

    @app.middleware("http")
    async def user_interface_auth_bridge(request: Request, call_next):
        """Translate approved UI credentials into the established API-key contract."""
        if request.url.path.startswith("/v1/") and not request.headers.get("X-API-Key"):
            api_credential_valid = False

            cookie = request.cookies.get(_UI_COOKIE)
            if _same_secret(cookie, api_key):
                api_credential_valid = True

            authorization = request.headers.get("Authorization", "")
            if authorization.lower().startswith("bearer ") and api_key:
                token = authorization.split(" ", 1)[1].strip()
                try:
                    _verify_artifact_token(
                        token=token,
                        secret=api_key,
                        request_origin=request.headers.get("origin"),
                    )
                except HTTPException as exc:
                    return JSONResponse(
                        {"detail": exc.detail},
                        status_code=exc.status_code,
                        headers={"Cache-Control": "no-store"},
                    )
                api_credential_valid = True

            if api_credential_valid and api_key:
                headers = list(request.scope.get("headers", []))
                headers.append((b"x-api-key", api_key.encode("utf-8")))
                request.scope["headers"] = headers

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
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
            "product": "NeuroGlycemic Sentinel",
            "research_stage": True,
            "auth_required": bool(api_key) and not unsafe_dev,
            "unsafe_dev": unsafe_dev,
            "consent_required": os.environ.get("DVXR_REQUIRE_CONSENT", "1") != "0",
            "fused_report_status": "abstains_until_synchronized_artifact",
            "artifact_token_exchange": bool(api_key) and not unsafe_dev,
            "artifact_token_ttl_seconds": _artifact_token_ttl(),
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

    @app.post("/ui/token", include_in_schema=False)
    async def create_artifact_token(request: Request):
        """Exchange the deployment access code for a short-lived origin-bound artifact token."""
        if unsafe_dev:
            raise HTTPException(
                status_code=409,
                detail="Artifact tokens are disabled in unsafe development mode.",
            )
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

        origin = _request_origin(request)
        token, ttl = _sign_artifact_token(
            secret=api_key,
            origin=origin,
            actor_id=os.environ.get("DVXR_ACTOR_ID", "artifact-user"),
        )
        response = JSONResponse(
            {
                "access_token": token,
                "token_type": "Bearer",
                "expires_in": ttl,
                "origin": origin,
            }
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    # Mount last so exact UI/static/developer routes win while the established product paths remain
    # unchanged: /health, /v1/risk-reports, /v1/predictions/..., /v1/alerts/..., and so on.
    app.mount("/", product)
    return app


app = build_app()
