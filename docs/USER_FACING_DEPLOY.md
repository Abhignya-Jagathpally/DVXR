# DVXR NeuroGlycemic Sentinel — user-facing web patch

This patch adds a polished, responsive research interface to the existing `dvxr-wrap-fastapi` branch without changing the predictive, consent, evidence, policy, audit, or abstention logic.

## What changes

- `/` becomes a product landing page and risk-review workspace.
- `/assets/*` serves dependency-free local CSS and JavaScript.
- `/ui/session` creates a short-lived, secure HttpOnly session from `DVXR_API_KEY`; the API key is not embedded in JavaScript.
- Existing endpoints remain unchanged:
  - `GET /health`
  - `POST /v1/risk-reports`
  - `GET /v1/predictions/{prediction_id}`
  - `GET/POST /v1/alerts/...`
- FastAPI developer documentation moves to `/developer/docs`.
- The multimodal `stress_glucose_risk` option remains visibly gated and will abstain until synchronized same-subject data and a validated artifact exist.

## Apply

From the extracted patch directory:

```bash
chmod +x apply_to_repo.sh
./apply_to_repo.sh /path/to/DVXR
cd /path/to/DVXR
pytest -q tests/test_user_facing_web.py
```

Or copy these paths manually:

```text
src/dvxr/serve/asgi.py
src/dvxr/web/__init__.py
src/dvxr/web/index.html
src/dvxr/web/assets/styles.css
src/dvxr/web/assets/app.js
tests/test_user_facing_web.py
```

## FastAPI Cloud environment

Use production-safe values:

```text
DVXR_API_KEY=<strong research access code>
DVXR_DB_PATH=dvxr.db
DVXR_ARTIFACT_ROOT=artifacts
DVXR_REQUIRE_CONSENT=1
DVXR_UNSAFE_DEV=0
DVXR_TENANT=default
DVXR_ACTOR_ID=web-researcher
```

Do **not** expose `DVXR_API_KEY` in frontend JavaScript or commit it to Git.

## Deploy

```bash
fastapi deploy
```

After deployment:

- Open `/` for the product interface.
- Open `/developer/docs` for the API documentation.
- Use “Explore an illustrative report” to present the interface without claiming that a model ran.
- Use “Generate risk review” only for provisioned, authorized, consented patient records.

## Product behavior preserved

The frontend deliberately renders all backend states:

- completed risk estimates and forecast intervals;
- explicit abstention and its reason;
- missing/stale modalities and data quality;
- model-derived evidence;
- grounded explanation;
- policy-selected action;
- alert acknowledge/escalate/dismiss operations;
- model, feature, calibration, policy, snapshot, request, and prediction provenance.

The interface never computes a clinical probability and never substitutes its own recommendation for the policy engine.