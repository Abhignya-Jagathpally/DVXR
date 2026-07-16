"""FastAPI Cloud entrypoint.

`fastapi deploy` / `fastapi run` discover the ASGI app from a root-level ``main.py`` (or ``app.py`` /
``api.py``). The real application is the FastAPI wrapper in :mod:`dvxr.serve.asgi`, which mounts the
Starlette Sentinel product API and is configured entirely through environment variables (see
``docs/DEPLOY.md`` and the module docstring). This file only re-exports it under the discovered name.

Honesty guardrail (unchanged): the fused ``stress_glucose_risk`` report abstains by construction — no
synchronized EEG+CGM data exists, so no fused artifact is ever served. Only single-modality CGM report
types can return a number, and only when a committed CGM artifact is provisioned in the deploy env.
"""
from dvxr.serve.asgi import app  # noqa: F401  (re-exported for app discovery)
