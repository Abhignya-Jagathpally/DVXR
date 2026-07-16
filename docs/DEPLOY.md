# Deploying the DVXR product API on FastAPI Cloud

The DVXR product API is a **Starlette** app (`dvxr.sentinel.create_product_api`). FastAPI Cloud deploys a
**FastAPI** `app`, so `dvxr/serve/asgi.py` wraps the Starlette product app in a FastAPI instance
(mounted as-is — no routes are re-modeled). `fastapi`/`uvicorn` are in the core dependencies (the app
imports on the slim runtime alone — no torch/transformers), so the cloud build resolves them from
`pyproject.toml` via `pip install .`.

**Entrypoint.** FastAPI Cloud discovers the app from a root-level `main.py`, which re-exports
`dvxr.serve.asgi:app`. **Build hygiene** is enforced by `.fastapicloudignore`: the heavy research
`requirements.txt` (torch/transformers/momentfm — won't build on 3.12), the screener `Dockerfile`,
`outputs/`, `data/`, `tests/`, and all personal docs/logs are withheld from the upload so the platform
builds the slim FastAPI product and nothing sensitive leaves the machine.

## What deploys

The **full product surface** — the Generate lifecycle plus all report types and the alert lifecycle:

| Route | Purpose |
|---|---|
| `GET /health` | liveness + the "not a diagnosis" disclaimer |
| `POST /v1/risk-reports` | Generate a risk report (all report types) |
| `GET /v1/predictions/{id}` | retrieve the persisted report |
| `GET /v1/alerts/{id}` | alert state |
| `POST /v1/alerts/{id}/acknowledge \| dismiss \| escalate` | alert lifecycle |

**Honesty guardrail (unchangeable):** `report_type=stress_glucose_risk` (the fused EEG+CGM product)
**abstains** — no synchronized same-subject dataset exists, so there is no fused artifact and the live
endpoint will never serve a fabricated number. Only the single-modality CGM report types
(`cgm_glucose_risk`, `cgm_glucose_forecast`) return a real prediction, and only when a committed CGM
artifact is provisioned.

## Configuration (environment variables, secure-by-default)

| Var | Default | Meaning |
|---|---|---|
| `DVXR_DB_PATH` | `dvxr.db` | sqlite path for persistent stores (predictions/alerts/audit survive restarts) |
| `DVXR_ARTIFACT_ROOT` | `artifacts` | committed model-artifact root |
| `DVXR_API_KEY` | — | `X-API-Key` for a researcher principal; **unset ⇒ `/v1` fails closed (401)** |
| `DVXR_UNSAFE_DEV` | — | `1` disables auth — demo only, NEVER in production |
| `DVXR_REQUIRE_CONSENT` | `1` | consent enforcement (default ON) |
| `DVXR_UI_ORIGINS` | `https://claude.ai` | comma-separated exact origins allowed to call the API cross-origin |
| `DVXR_UI_ORIGIN_REGEX` | `^https://([a-z0-9-]+\.)?(claude\.ai\|claudeusercontent\.com)$` | regex for allowed artifact origins (narrow to the exact origin once known) |
| `DVXR_ARTIFACT_TOKEN_TTL_SECONDS` | `900` | lifetime of the origin-bound artifact bearer token (clamped 60–3600) |

### Cross-origin artifact bridge

The served UI at `/` is same-origin (bearer token via `POST /ui/token`, key never in JS). An **external**
Claude-hosted artifact (`docs/claude-artifact.html`) is cross-origin: it exchanges the deployment access
code at `/ui/token` for a short-lived, **origin-bound** bearer token, which the API translates into the
`X-API-Key` contract. CORS uses `allow_credentials=False` (bearer, not cookies). Inspect the real `Origin`
header after the first call and narrow `DVXR_UI_ORIGINS` / `DVXR_UI_ORIGIN_REGEX` to it.

## Provisioning the CGM artifact (to serve real predictions, not just abstain)

A clean deploy has no artifact (`artifacts/` is gitignored) → the CGM paths fail closed to abstention.
To serve real CGM-only predictions, build the artifact **against the same db + root the API uses**:

```bash
python scripts/build_cgm_artifact.py \
    --artifact-root "$DVXR_ARTIFACT_ROOT" \
    --registry-db   "$DVXR_DB_PATH"
```

This fits the incident-onset CGM classifier + the continuous forecaster on CGMacros, saves them under
`$DVXR_ARTIFACT_ROOT`, and registers them ACTIVE in `$DVXR_DB_PATH`. The build needs CGMacros present at
`data/real/cgmacros`. **Data-handling note:** shipping the artifact + wiring how a patient's CGM history
reaches the service is a real PHI/consent decision for a health tool — provision deliberately, keep
`DVXR_REQUIRE_CONSENT=1` and `DVXR_API_KEY` set, never `DVXR_UNSAFE_DEV=1`.

## Deploy

`fastapi login` and `fastapi deploy` are **interactive** (browser login; team picker + first-run
app-creation wizard) — run them yourself in a terminal:

```bash
fastapi login                 # once (opens a browser)
fastapi deploy                # discovers main:app, uploads (respecting .fastapicloudignore), builds
```

The first deploy creates the app and prints its URL. **Set the API key** so `/v1` is reachable (without
it every `/v1` call fails closed with 401 — `/health` still works):

```bash
fastapi cloud env set DVXR_API_KEY <a-strong-random-key>
fastapi cloud env set DVXR_REQUIRE_CONSENT 1     # default; keep consent ON
# do NOT set DVXR_UNSAFE_DEV — auth stays ON in production
fastapi deploy                                    # redeploy so the new env takes effect
```

A clean deploy has no CGM artifact (`artifacts/` is withheld), so **every report abstains** — CGM types
with "no committed artifact", the fused type by construction. That is the honest, safe default; provision
the artifact deliberately (above) to serve real CGM-only numbers.

Smoke-test after deploy:

```bash
curl https://<your-app>/health                                      # -> ok + disclaimer, no auth
curl -H "X-API-Key: $DVXR_API_KEY" -X POST https://<your-app>/v1/risk-reports \
     -d '{"patient_id":"P1","report_type":"stress_glucose_risk"}'   # -> abstains (honest)
```
