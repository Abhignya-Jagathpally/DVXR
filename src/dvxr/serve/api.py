"""dvxr.serve.api — a thin HTTP API so the product is deployable.

Built on Starlette (no FastAPI dependency; Starlette + uvicorn are the runtime). Wraps the existing
serving functions — nothing new is modeled here:

    GET  /health                      liveness + the mandatory disclaimer
    GET  /tasks                       available screening tasks + their headline AUROCs
    GET  /evidence                    the scoreboard-traced evidence report (text)
    GET  /evidence/{task}             DVXR vs published-SOTA (JSON, protocol-labeled, DOIs)
    POST /screen/subject  {task,sid}  live-score a held-out cohort subject (validated)
    GET  /triage/{task}?top=N         cohort risk ranking (highest first)

Screeners + cohort tasks are loaded lazily and cached per process. Every screening response carries
the research-prototype caveat. Offline / CPU / deterministic; research-grade screening, not diagnosis.

Run:  dvxr serve-api    (or)   uvicorn "dvxr.serve.api:app" --factory
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

DISCLAIMER = ("Research-grade screening / decision-support only — not a diagnosis. "
              "A raised risk is a prompt to consult a qualified clinician, never a conclusion.")
_SCREENER_ROOT = Path("outputs/product/screeners")


def create_app(screener_root: str | Path = _SCREENER_ROOT,
               db_path: str = ":memory:", require_consent: bool = True,
               principals: dict | None = None, unsafe_dev: bool = False):
    """Build the ASGI app. Secure by default: the /v1 patient endpoints require an ``X-API-Key`` that
    resolves to a server-side Principal (role/tenant come from the principal, NOT the request body).
    Pass ``principals`` (api_key -> Principal) for a real deployment, or ``unsafe_dev=True`` for a
    local demo with no auth. ``require_consent`` defaults ON (spec §2, §18)."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    from dvxr.storage import open_local_stores

    root = Path(screener_root)
    _screeners: dict = {}
    _tasks: dict = {}
    # process-local stateful stores (spec §6). ":memory:" for the default single-process research
    # deployment; pass a file path to persist predictions/audit across restarts.
    pred_store, audit_store, consent_store, _model_registry = open_local_stores(db_path)

    def _get_screener(task: str):
        """Load a COMMITTED screener. Generate/serving never trains during a request (spec §2): a
        missing artifact raises rather than fitting on the fly."""
        if task not in _screeners:
            from dvxr.serve.screener import Screener
            d = root / task
            if not (d / "manifest.json").exists():
                raise RuntimeError(
                    f"no committed screener for task {task!r} at {d} — serving never trains during a "
                    f"request; fit and commit the screener first (`dvxr fit --task {task}`).")
            _screeners[task] = Screener.load(d)
        return _screeners[task]

    def _get_task(task_name: str, representation: str):
        if task_name not in _tasks:
            from dvxr.bench.tasks import TASK_BUILDERS
            t = TASK_BUILDERS[task_name]()
            t.name = task_name
            t.extra["_representation"] = representation
            _tasks[task_name] = t
        return _tasks[task_name]

    async def health(request):
        return JSONResponse({"status": "ok", "product": "DVXR Screen",
                             "disclaimer": DISCLAIMER})

    async def tasks(request):
        from dvxr.serve.evidence import product_numbers
        pn = product_numbers()
        return JSONResponse({"tasks": [
            {"task": t, "label": v["label"], "auroc_window": v["auroc"], "encoder": v["encoder"]}
            for t, v in pn.items()], "disclaimer": DISCLAIMER})

    async def evidence(request):
        from dvxr.serve.evidence import render_report
        return PlainTextResponse(render_report())

    async def evidence_task(request):
        from dvxr.serve.evidence import external_comparison, OUR_METRICS
        task = request.path_params["task"]
        if task not in OUR_METRICS:
            return JSONResponse({"error": f"unknown task {task!r}"}, status_code=404)
        return JSONResponse(external_comparison(task))

    async def screen_subject(request):
        from dvxr.serve.live import run_screening_live
        body = await request.json()
        task = body.get("task")
        sid = body.get("subject")
        if not task:
            return JSONResponse({"error": "body must include 'task'"}, status_code=400)
        try:
            screener = _get_screener(task)
            t = _get_task(task, screener.representation)
            import numpy as np
            subjects = np.asarray(t.subject_ids)
            if sid is None:
                sid = list(dict.fromkeys(subjects.tolist()))[-1]
            elif sid not in subjects:
                return JSONResponse({"error": f"subject {sid!r} not in cohort {task}"},
                                    status_code=404)
            out = run_screening_live(screener, t, sid, validated=True, source="cohort")
        except Exception as e:  # noqa: BLE001 — surface the reason
            return JSONResponse({"error": str(e)}, status_code=500)
        res = out["result"]
        return JSONResponse({
            "task": task, "subject": str(sid), "probability": res["probability"],
            "risk_band": res["risk_band"], "interval": res["interval"],
            "n_windows": res["n_windows"], "heldout_auroc": res.get("heldout_auroc"),
            "heldout_auroc_subject": res.get("heldout_auroc_subject"),
            "window_probs": out["window_probs"], "validated": out["validated"],
            "disclaimer": DISCLAIMER})

    async def triage(request):
        from dvxr.serve.batch import triage_cohort
        task = request.path_params["task"]
        top = int(request.query_params.get("top", 20))
        try:
            df = triage_cohort(_get_screener(task), task).head(top)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse({"task": task, "n": int(len(df)),
                             "ranking": df.to_dict(orient="records"), "disclaimer": DISCLAIMER})

    def _principal(request):
        from dvxr.serve.auth import authenticate
        return authenticate(request.headers.get("X-API-Key"), principals, unsafe_dev=unsafe_dev)

    async def risk_reports(request):
        """POST /v1/risk-reports — the Generate lifecycle (spec §2). Never trains; persists a
        reproducible, audited request; idempotent by (tenant, patient, report_type, key). The actor's
        role + tenant are taken from the authenticated principal, NEVER from the request body."""
        import dataclasses

        from dvxr.contracts import GenerateRequest
        from dvxr.serve.auth import AuthError, AuthorizationError, authorize
        from dvxr.serve.orchestrate import ConsentError, generate_risk_report
        try:
            principal = _principal(request)
        except AuthError as e:
            return JSONResponse({"error": str(e)}, status_code=401)
        body = await request.json()
        if not body.get("patient_id"):
            return JSONResponse({"error": "body must include 'patient_id'"}, status_code=400)
        try:
            authorize(principal, str(body["patient_id"]), "generate_risk_report")
        except AuthorizationError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        # server-derived identity overrides anything the caller tried to self-assert in the body
        req = dataclasses.replace(GenerateRequest.from_dict(body),
                                  user_role=principal.role, tenant_id=principal.tenant_id)
        try:
            out = generate_risk_report(req, prediction_store=pred_store, audit_store=audit_store,
                                       consent_store=consent_store, require_consent=require_consent)
        except ConsentError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        except Exception:  # noqa: BLE001 — never leak internals to the client
            return JSONResponse({"error": "internal error generating report"}, status_code=500)
        out["disclaimer"] = DISCLAIMER
        return JSONResponse(out)

    async def get_risk_report(request):
        from dvxr.serve.auth import AuthError, AuthorizationError, authorize
        try:
            principal = _principal(request)
        except AuthError as e:
            return JSONResponse({"error": str(e)}, status_code=401)
        pid = request.path_params["prediction_id"]
        # fetch is TENANT-SCOPED to the caller's principal: a prediction_id that exists only under a
        # different tenant is a 404 here (it never leaves storage), so a cross-tenant id collision can
        # never surface another tenant's record. authorize() below is defense-in-depth on patient scope.
        rec = pred_store.get(pid, tenant_id=principal.tenant_id)
        if rec is None:
            return JSONResponse({"error": f"unknown prediction {pid!r}"}, status_code=404)
        try:
            authorize(principal, str(rec.get("patient_id")), "read_prediction",
                      record_tenant=rec.get("tenant_id"))
        except AuthorizationError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        return JSONResponse({"prediction": rec, "disclaimer": DISCLAIMER})

    return Starlette(routes=[
        Route("/health", health),
        Route("/tasks", tasks),
        Route("/evidence", evidence),
        Route("/evidence/{task}", evidence_task),
        Route("/screen/subject", screen_subject, methods=["POST"]),
        Route("/triage/{task}", triage),
        Route("/v1/risk-reports", risk_reports, methods=["POST"]),
        Route("/v1/predictions/{prediction_id}", get_risk_report),
    ])


def app():
    """ASGI factory for `uvicorn dvxr.serve.api:app --factory`."""
    return create_app()


def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)
