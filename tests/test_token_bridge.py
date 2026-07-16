"""Focused tests for the dependency-free, origin-bound artifact session token in the ASGI bridge.

Loads ONLY the helper functions defined above build_app() (splitting the source on ``def build_app()``),
so the token crypto can be exercised without constructing the full DVXR product app.
"""
from __future__ import annotations

import importlib.util
import pathlib
import time

import pytest
from fastapi import HTTPException

# The bridge lives in the installed serving package after the patch is applied.
MODULE_PATH = pathlib.Path(__file__).parents[1] / "src" / "dvxr" / "serve" / "asgi.py"
SPEC = importlib.util.spec_from_file_location("bridge_asgi", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)

SOURCE = MODULE_PATH.read_text(encoding="utf-8")
HELPERS = SOURCE.split("def build_app()", 1)[0]
exec(compile(HELPERS, str(MODULE_PATH), "exec"), MODULE.__dict__)


def test_artifact_token_is_origin_bound(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    monkeypatch.setenv("DVXR_ARTIFACT_TOKEN_TTL_SECONDS", "900")
    token, ttl = MODULE._sign_artifact_token(
        secret="correct-horse-battery-staple",
        origin="https://claude.ai",
        actor_id="researcher-1",
    )
    payload = MODULE._verify_artifact_token(
        token=token,
        secret="correct-horse-battery-staple",
        request_origin="https://claude.ai",
    )
    assert ttl == 900
    assert payload["sub"] == "researcher-1"
    assert payload["origin"] == "https://claude.ai"


def test_artifact_token_rejects_a_different_origin(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    token, _ = MODULE._sign_artifact_token(secret="secret", origin="https://claude.ai", actor_id="r1")
    with pytest.raises(HTTPException) as exc:
        MODULE._verify_artifact_token(token=token, secret="secret",
                                      request_origin="https://untrusted.example")
    assert exc.value.status_code == 401


def test_artifact_token_rejects_tampering(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    token, _ = MODULE._sign_artifact_token(secret="secret", origin="https://claude.ai", actor_id="r1")
    altered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(HTTPException):
        MODULE._verify_artifact_token(token=altered, secret="secret", request_origin="https://claude.ai")


def test_artifact_token_rejects_expired(monkeypatch):
    # mint at t0 with a 60s TTL, verify 61s later -> expired
    monkeypatch.setenv("DVXR_ARTIFACT_TOKEN_TTL_SECONDS", "60")
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    token, _ = MODULE._sign_artifact_token(secret="secret", origin="https://claude.ai", actor_id="r1")
    monkeypatch.setattr(time, "time", lambda: 1_700_000_061)
    with pytest.raises(HTTPException) as exc:
        MODULE._verify_artifact_token(token=token, secret="secret", request_origin="https://claude.ai")
    assert exc.value.status_code == 401


def test_artifact_token_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    token, _ = MODULE._sign_artifact_token(secret="secret", origin="https://claude.ai", actor_id="r1")
    with pytest.raises(HTTPException):
        MODULE._verify_artifact_token(token=token, secret="a-different-secret",
                                      request_origin="https://claude.ai")
