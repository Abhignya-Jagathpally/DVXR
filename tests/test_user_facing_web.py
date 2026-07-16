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
    assert "NeuroGlycemic Sentinel" in response.text
    assert "Check. Understand. Act." in response.text          # the product narrative is present
    assert "DVXR" not in response.text                          # anonymized: no brand identifier
    assert "UNT" not in response.text                           # anonymized: no institution identifier


def test_predicted_targets_and_token_exchange_are_exposed(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, key="secret-code"))
    page = client.get("/").text
    for target in ("Stress-associated glucose risk", "CGM excursion risk", "Glucose forecast"):
        assert target in page                                  # every predicted target is named up front
    cfg = client.get("/ui/config").json()
    assert cfg["artifact_token_exchange"] is True              # cross-origin bridge is advertised
    assert cfg["fused_report_status"] == "abstains_until_synchronized_artifact"


def test_static_assets_are_served(monkeypatch, tmp_path):
    client = TestClient(_load_app(monkeypatch, tmp_path, unsafe_dev=True))
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200


def test_index_asset_references_all_resolve(monkeypatch, tmp_path):
    """Every stylesheet/script the served page links must resolve to 200.

    Regression guard for the path mismatch that made the dashboard look "crashed": index.html linked
    href="./styles.css" / src="./app.js" (root) while the app serves them under /assets/, so the CSS and
    JS both 404'd — the page loaded but rendered unstyled and never initialized. Fetching every ref the
    HTML actually points at (not just probing /assets/*) is what catches this.
    """
    import re

    client = TestClient(_load_app(monkeypatch, tmp_path, unsafe_dev=True))
    html = client.get("/").text
    refs = re.findall(r'(?:href|src)="([^"]+?\.(?:css|js)(?:\?[^"]*)?)"', html)
    local = [r for r in refs if not r.startswith(("http://", "https://"))]
    assert local, "index.html should reference at least one local css/js asset"
    for ref in local:
        path = ref[1:] if ref.startswith("./") else ref  # "./assets/app.js" -> "/assets/app.js"
        if not path.startswith("/"):
            path = "/" + path
        path = path.split("?", 1)[0]  # drop any ?v= cache-buster
        assert client.get(path).status_code == 200, f"asset ref {ref!r} -> {path} did not return 200"


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
