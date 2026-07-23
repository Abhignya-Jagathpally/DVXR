"""Orchestration-graph nodes: thin adapters over ``research_predict`` helpers.

Each node reuses the committed scoring functions rather than reimplementing them, so the
numeric body the graph assembles is byte-identical to ``run_research_prediction``. The
only additions are a grounded explanation and a per-node audit ``trace``.

Honesty invariants expressed as node boundaries:
  * modality scoping — each per-target prediction reads only its modality's feature slice
    (``_observed(features, TARGET_FEATURES[t])``); a neural target cannot see metabolic input;
  * fail-closed abstention — the ``calibration_gate`` node is the single place that can set
    the response status to ``abstained``; the explanation branch never revives a number;
  * LLM explains, never predicts — the explanation node only restates values already frozen
    in the body; it introduces no new number.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from dvxr.serve import research_predict as rp
from dvxr.serve.agents.state import PipelineState


def _trace(state: PipelineState, node: str, **detail: Any) -> List[Dict[str, Any]]:
    record = {"node": node, **detail}
    return [*state.get("trace", []), record]


def ingestion_availability(state: PipelineState) -> Dict[str, Any]:
    """Validate the request, load committed artifacts, derive available modalities."""
    req = rp.FeaturePredictionRequest.from_payload(state["payload"])
    base_heads, meta, provenance = rp.load_research_models(
        rp.resolve_models_root(state.get("screener_root"))
    )
    prediction_id = "res-" + hashlib.sha256(
        json.dumps({"f": req.features, "o": req.outcome}, sort_keys=True).encode()
    ).hexdigest()[:16]
    quality = rp._input_quality(req.features, req.warnings)
    present = sorted(
        {rp.CANONICAL_FEATURES[name][0] for name in req.features if name in rp.CANONICAL_FEATURES}
    )
    return {
        "req": req,
        "heads": base_heads,
        "meta": meta,
        "prediction_id": prediction_id,
        "provenance": provenance,
        "input_quality": quality,
        "available_modalities": present,
        "trace": _trace(state, "ingestion_availability",
                         modalities=present, provenance=provenance,
                         overall_quality=quality.get("overall")),
    }


def encode_predict(state: PipelineState) -> Dict[str, Any]:
    """Per-target base predictions — each reads ONLY its modality's feature slice."""
    req = state["req"]
    base_heads = state["heads"]
    provenance = state["provenance"]
    target_predictions: Dict[str, Any] = {}
    prob_stack: Dict[str, float] = {}
    for t in rp.TARGETS:
        head = base_heads[t]
        obs = rp._observed(req.features, rp.TARGET_FEATURES[t])
        if not obs:
            target_predictions[t] = {
                "probability": None, "risk_band": None, "confidence": 0.0,
                "model_version": head.model_version, "evidence_status": "abstained",
                "reason_codes": ["no_modality_input"]}
            continue
        p = head.predict_proba(obs)
        ev = head.evidence_status if provenance == "committed" else "simulation"
        target_predictions[t] = {
            "probability": round(float(p), 4), "risk_band": rp.risk_band(p),
            "confidence": rp._confidence(len(obs), len(rp.TARGET_FEATURES[t])),
            "model_version": head.model_version, "evidence_status": ev}
        prob_stack[f"prob_{t}"] = p
    return {
        "target_predictions": target_predictions,
        "prob_stack": prob_stack,
        "trace": _trace(state, "encode_predict",
                        predicted=[t for t, v in target_predictions.items()
                                   if v.get("probability") is not None],
                        abstained_targets=[t for t, v in target_predictions.items()
                                           if v.get("probability") is None]),
    }


def fuse_select(state: PipelineState) -> Dict[str, Any]:
    """Selected diabetes/glucose outcome (meta stack) + honest linear contributions."""
    req = state["req"]
    base_heads = state["heads"]
    meta = state["meta"]
    metabolic_obs = rp._observed(req.features, rp.META_METABOLIC_FEATURES)
    selected = rp._selected_outcome(
        req.outcome, meta, base_heads, req.features, metabolic_obs,
        state["prob_stack"], state["provenance"])
    contributions = rp._contributions_for(
        req.outcome, meta, base_heads, req.features, metabolic_obs, state["prob_stack"])
    return {
        "selected_outcome": selected,
        "contributions": contributions,
        "trace": _trace(state, "fuse_select", outcome=req.outcome,
                        selected_status=selected.get("status", "ok")),
    }


def forecast(state: PipelineState) -> Dict[str, Any]:
    """Glucose forecast (abstains without a committed CGM artifact — the resolver decides)."""
    req = state["req"]
    fc = rp._forecast(req.features, req.horizons_minutes,
                      screener_root=state.get("screener_root"))
    return {
        "forecast": fc,
        "trace": _trace(state, "forecast",
                        evidence_status=fc.get("evidence_status")),
    }


def calibration_gate(state: PipelineState) -> Dict[str, Any]:
    """Assemble the response body and set the fail-closed abstention status.

    This is the single node that can flip the overall status to ``abstained``. The body
    is exactly what ``run_research_prediction`` returns."""
    from dvxr.serve.api import DISCLAIMER

    selected = state["selected_outcome"]
    body: Dict[str, Any] = {
        "prediction_id": state["prediction_id"],
        "status": "ok",
        "research_stage": True,
        "evidence_provenance": state["provenance"],
        "input_quality": state["input_quality"],
        "target_predictions": state["target_predictions"],
        "selected_outcome": selected,
        "contributions": state["contributions"],
        "forecast": state["forecast"],
        "disclaimer": DISCLAIMER,
    }
    abstained = selected.get("status") == "abstained" or selected.get("probability") is None
    if abstained:
        body["status"] = "abstained"
        body["reason_codes"] = selected.get("reason_codes", ["insufficient_metabolic_input"])
        body["missing_or_stale_data"] = selected.get(
            "missing_or_stale_data",
            ["metabolic inputs (HbA1c / fasting glucose / CGM summary / BMI)"])
    return {
        "body": body,
        "abstained": bool(abstained),
        "trace": _trace(state, "calibration_gate", status=body["status"]),
    }


def _grounded_explanation(body: Dict[str, Any]) -> Dict[str, Any]:
    """A grounded narrative that only restates numbers already in ``body``.

    This is the ``HealthAgent`` seat: an LLM can replace this deterministic renderer, but
    it must remain explanation-only — it may never introduce a value not already present.
    """
    selected = body.get("selected_outcome", {})
    name = selected.get("name", "the selected outcome")
    if body.get("status") == "abstained" or selected.get("probability") is None:
        missing = body.get("missing_or_stale_data") or selected.get("missing_or_stale_data") or []
        text = (
            f"The model abstained on {name}: the required inputs were not present, so no "
            f"probability was produced. Provide {', '.join(missing) if missing else 'the missing inputs'} "
            "to obtain a research-stage estimate. This is decision-support, not a diagnosis."
        )
        drivers: List[Dict[str, Any]] = []
    else:
        prob = selected.get("probability")
        band = selected.get("risk_band")
        drivers = [
            {"feature": c.get("feature"), "direction": c.get("direction"),
             "contribution": c.get("contribution")}
            for c in body.get("contributions", [])[:3]
        ]
        driver_txt = "; ".join(
            f"{d['feature']} ({d['direction']})" for d in drivers if d.get("feature")
        )
        text = (
            f"Research-stage estimate for {name}: probability {prob} (risk band {band}). "
            + (f"Top contributors: {driver_txt}. " if driver_txt else "")
            + "Not validated for clinical use; this explains the model output and makes no "
              "independent prediction."
        )
    return {
        "text": text,
        "grounded_on": {
            "status": body.get("status"),
            "selected_probability": selected.get("probability"),
            "risk_band": selected.get("risk_band"),
            "evidence_status": selected.get("evidence_status"),
            "validated_for_clinical_use": selected.get("validated_for_clinical_use", False),
        },
        "top_contributions": drivers,
        "predicts": False,
    }


def explain_prediction(state: PipelineState) -> Dict[str, Any]:
    from dvxr.serve.llm_explainer import explain
    return {
        "explanation": explain(state["body"]),
        "trace": _trace(state, "explain_prediction"),
    }


def explain_abstention(state: PipelineState) -> Dict[str, Any]:
    from dvxr.serve.llm_explainer import explain
    return {
        "explanation": explain(state["body"]),
        "trace": _trace(state, "explain_abstention"),
    }


def route_after_gate(state: PipelineState) -> str:
    """Conditional edge: fail-closed abstention goes to the abstention explainer."""
    return "explain_abstention" if state.get("abstained") else "explain_prediction"
