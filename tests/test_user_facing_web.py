"""Contract tests for the user-facing FastAPI wrapper."""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _load_app(monkeypatch, tmp_path, *, key: str | None = None, unsafe_dev: bool = False):
    monkeypatch.setenv("DVXR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DVXR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DVXR_REQUIRE_CONSENT", "1")
    if key:
        monkeypatch.setenv("DVXR_API_KEY", key)
    else:
        monkeypatch.delenv("DVXR_API_KEY", raising=False)
    monkeypatch.setenv("DVXR_UNSAFE_DEV", "1" if unsafe_dev else "0")

    import dvxr.serve.asgi as asgi
    importlib.reload(asgi)
    return asgi.app


def test_home_is_a_user_facing_page(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, unsafe_dev=True))
    response = client.get("/")
    assert response.status_code == 200
    assert "DVXR NeuroGlycemic Sentinel" in response.text
    assert "Generate a risk review" in response.text


def test_static_assets_are_served(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, unsafe_dev=True))
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200


def test_ui_session_does_not_expose_key(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, key="secret-code"))
    config = client.get("/ui/config")
    assert config.status_code == 200
    assert config.json()["auth_required"] is True
    assert "secret-code" not in config.text

    denied = client.post("/ui/session", json={"access_code": "wrong"})
    assert denied.status_code == 401

    accepted = client.post("/ui/session", json={"access_code": "secret-code"})
    assert accepted.status_code == 200
    assert accepted.json()["authenticated"] is True
    assert accepted.cookies.get("dvxr_ui_session") == "secret-code"


def test_existing_product_route_remains_available(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, key="secret-code"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
