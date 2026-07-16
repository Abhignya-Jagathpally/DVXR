"""dvxr.llm.grounded — structured, validated explanation over an immutable prediction (spec §9, §15).

This is the product's language layer boundary: it receives an IMMUTABLE prediction + model evidence +
the policy's action + retrieved source passages, and emits a fixed-schema structured explanation. It
does NOT compute risk, infer missing values, invent a diagnosis, or select a different action. The
assembled text is checked by `dvxr.safety.validators` before it is returned; on any grounding failure
the caller shows the deterministic template / abstention instead of ungrounded prose.

The default path is deterministic (no network): every sentence is built from the input objects, so it
passes the validators by construction. An LLM/SLM may later author `risk_summary`/`action_explanation`
prose, but it is subjected to the SAME validators — it can only phrase the verified facts.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from dvxr.safety.policy import action_text
from dvxr.safety.validators import (
    GroundingError,
    validate_action_id,
    validate_citations,
    validate_no_diagnosis_language,
    validate_numbers,
)

_DISCLAIMER = "This is research-grade decision-support, not a diagnosis."


def grounded_explanation(prediction: Dict, evidence: Optional[Dict], action: Dict,
                         sources: Optional[List[Dict]] = None) -> Dict:
    """Return a fixed-schema, validated explanation of ``prediction`` (spec §15).

    ``prediction`` / ``evidence`` / ``action`` are the ``to_dict()`` of the immutable contracts;
    ``sources`` are retrieved chunks ({chunk_id, text, metadata}). Raises GroundingError if the
    assembled explanation fails a numeric/citation/action/diagnosis check (a bug — the deterministic
    builder should always pass; the guard exists so an LLM-authored variant can be rejected)."""
    sources = sources or []
    evidence = evidence or {}
    horizons = prediction.get("prediction_horizons_minutes") or []
    abstained = bool(prediction.get("abstained"))

    # each supporting factor cites the immutable evidence_id of its contribution (spec §8.6). The
    # evidence records ARE the source for model-derived factors; a contribution with no matching record
    # yields source_id=None, which the citation validator rejects (never a source-free claim).
    evidence_records = evidence.get("evidence_records") or []
    ev_by_feature = {r["feature"]: r["evidence_id"] for r in evidence_records}
    supporting: List[Dict] = []
    for m, contrib in (evidence.get("contributions") or {}).items():
        supporting.append({
            "statement": f"{m} contributed to the prediction",
            "source_type": "model_evidence", "source_id": ev_by_feature.get(m),
            "value": float(contrib)})

    citations = [{"source_id": s["chunk_id"],
                  "document_type": s["metadata"].get("document_type"),
                  "section": s["metadata"].get("section")} for s in sources]

    if abstained:
        risk_summary = (prediction.get("abstain_reason")
                        or "A reliable prediction cannot be produced from the available data.")
        uncertainty = "No calibrated probability is available; the system abstained."
    else:
        risk = prediction.get("risk") or {}
        parts = [f"{hz}-minute risk {risk.get(f'excursion_{hz}m'):.2f}"
                 for hz in horizons if risk.get(f"excursion_{hz}m") is not None]
        risk_summary = ("Estimated glucose-excursion risk — " + ", ".join(parts)) if parts \
            else "A calibrated risk estimate is available."
        uncertainty = (f"Confidence {prediction['confidence']:.2f}."
                       if prediction.get("confidence") is not None else "Confidence not reported.")

    action_id = action.get("action_id", "")
    explanation = {
        "risk_summary": risk_summary,
        "prediction_horizon_minutes": horizons,
        "supporting_factors": supporting,
        "missing_or_stale_data": list(prediction.get("missing_modalities") or []),
        "uncertainty_statement": uncertainty,
        "action_id": action_id,
        "action_explanation": action_text(action_id) if action_id else "",
        "citations": citations,
        "limitations": [_DISCLAIMER],
    }

    # validate the assembled text against the immutable sources (spec §8)
    text_blob = " ".join([risk_summary, uncertainty, explanation["action_explanation"],
                          *(s["statement"] for s in supporting)])
    validate_numbers(text_blob, prediction, evidence)
    # valid sources = the immutable evidence records AND the retrieved chunks
    valid_sources = {r["evidence_id"] for r in evidence_records} | {s["chunk_id"] for s in sources}
    validate_citations(supporting + citations, valid_sources)
    validate_action_id(action_id, action.get("action_id", ""))
    validate_no_diagnosis_language(text_blob + " " + _DISCLAIMER)
    return explanation
