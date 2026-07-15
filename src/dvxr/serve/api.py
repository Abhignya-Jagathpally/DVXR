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


def create_app(screener_root: str | Path = _SCREENER_ROOT):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    root = Path(screener_root)
    _screeners: dict = {}
    _tasks: dict = {}

    def _get_screener(task: str):
        if task not in _screeners:
            from dvxr.serve.screener import Screener, fit_screener
            d = root / task
            _screeners[task] = (Screener.load(d) if (d / "manifest.json").exists()
                                else fit_screener(task))
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

    return Starlette(routes=[
        Route("/health", health),
        Route("/tasks", tasks),
        Route("/evidence", evidence),
        Route("/evidence/{task}", evidence_task),
        Route("/screen/subject", screen_subject, methods=["POST"]),
        Route("/triage/{task}", triage),
    ])


def app():
    """ASGI factory for `uvicorn dvxr.serve.api:app --factory`."""
    return create_app()


def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)
