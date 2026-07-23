"""Multi-agent orchestration of the research-prediction pipeline.

``run_agentic_prediction`` returns the *same* numeric response as
``dvxr.serve.research_predict.run_research_prediction`` (a byte-identical body), plus an
additive ``trace`` (per-node audit) and a grounded ``explanation``. The graph orchestrates;
it never computes a new prediction number.

LangGraph is an optional dependency (``dvxr[agents]``). Import stays lazy so the base
package and the torch-free honesty audit are unaffected when it is absent.
"""

from __future__ import annotations

from typing import Any, Dict


def run_agentic_prediction(payload: dict, *, screener_root: Any = None) -> Dict[str, Any]:
    """Score a research-prediction request through the orchestration graph.

    Raises ``research_predict.ValidationError`` on bad input (same contract as the direct
    path, so the HTTP handler maps it to 400)."""
    from dvxr.serve.agents.graph import compiled_graph

    initial = {"payload": payload, "screener_root": screener_root, "trace": []}
    final = compiled_graph().invoke(initial)
    body = dict(final["body"])
    body["explanation"] = final.get("explanation")
    body["trace"] = final.get("trace", [])
    body["orchestration"] = "langgraph-v1"
    return body


def agentic_available() -> bool:
    """True when LangGraph is importable (the ``agents`` extra is installed)."""
    try:
        import langgraph  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["run_agentic_prediction", "agentic_available"]
