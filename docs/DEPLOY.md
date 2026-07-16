# Deploying the DVXR product API on FastAPI Cloud

The DVXR product API is a **Starlette** app (`dvxr.sentinel.create_product_api`). FastAPI Cloud deploys a
**FastAPI** `app`, so `dvxr/serve/asgi.py` wraps the Starlette product app in a FastAPI instance
(mounted as-is — no routes are re-modeled). Install the deploy extra: `pip install -e '.[deploy]'`.

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

```bash
fastapi login                 # once (the user has already signed in)
fastapi deploy                # deploys dvxr.serve.asgi:app
```

Smoke-test after deploy:

```bash
curl https://<your-app>/health
curl -H "X-API-Key: $DVXR_API_KEY" -X POST https://<your-app>/v1/risk-reports \
     -d '{"patient_id":"P1","report_type":"stress_glucose_risk"}'   # -> abstains (honest)
```
