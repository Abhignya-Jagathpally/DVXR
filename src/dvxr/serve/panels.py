"""dvxr.serve.panels — the six-panel report assembly for the dashboard (spec §11, §13, §20).

The dashboard must visibly SEPARATE data → prediction → explanation → action. This module runs the
Generate lifecycle (`dvxr.serve.orchestrate`, which never trains) and shapes its output into six
panels the UI renders. Because it returns plain structured data (no Streamlit), the panel content is
unit-testable and the same assembly can back the web app, an EHR module, or the mobile view.

  1. context           — patient, role, cutoff, horizons, model/policy version
  2. data_readiness    — modality freshness/quality, missing/stale modalities, abstention flag
  3. prediction        — risk (or abstention), category, uncertainty
  4. why               — model-derived supporting factors (never an LLM guess)
  5. next_action       — policy action id + approved text, reason codes, clinician-review flag
  6. evidence_provenance — request/prediction ids, model/feature versions, cutoff, citations, limits
"""
from __future__ import annotations

from typing import Dict

from dvxr.contracts import GenerateRequest
from dvxr.safety.policy import action_text
from dvxr.serve.orchestrate import generate_risk_report


def build_report_panels(
    request: GenerateRequest,
    *,
    prediction_store,
    audit_store,
    consent_store=None,
    require_consent: bool = False,
) -> Dict:
    """Run Generate and return the six dashboard panels (spec §13). Never trains a model."""
    report = generate_risk_report(request, prediction_store=prediction_store,
                                  audit_store=audit_store, consent_store=consent_store,
                                  require_consent=require_consent)
    pred = report["prediction"]
    action = report["action"]
    explanation = report.get("explanation", {})
    abstained = bool(pred.get("abstained"))

    context = {
        "patient_id": pred.get("patient_id"),
        "user_role": request.user_role,
        "data_cutoff_at": pred.get("data_cutoff_at"),
        "prediction_horizons_minutes": pred.get("prediction_horizons_minutes")
        or request.prediction_horizons_minutes,
        "model_version": report.get("model_version"),
        "policy": {"id": action.get("policy_id"), "version": action.get("policy_version")},
    }
    data_readiness = {
        "missing_modalities": pred.get("missing_modalities", []),
        "stale_modalities": pred.get("stale_modalities", []),
        "data_quality": pred.get("data_quality", "unknown"),
        "abstained": abstained,
    }
    prediction_panel = {
        "abstained": abstained,
        "risk": pred.get("risk"),
        "risk_category": pred.get("risk_category"),
        "confidence": pred.get("confidence"),
        "uncertainty_statement": explanation.get("uncertainty_statement"),
        "summary": explanation.get("risk_summary"),
    }
    why = {"supporting_factors": explanation.get("supporting_factors", [])}
    next_action = {
        "action_id": action.get("action_id"),
        "action_text": action_text(action["action_id"]) if action.get("action_id") else "",
        "reason_codes": action.get("reason_codes", []),
        "requires_clinician_review": action.get("requires_clinician_review", False),
        # acknowledge/dismiss/escalate are UI controls; the ids the UI can POST back
        "controls": ["acknowledge", "dismiss", "escalate"],
    }
    evidence_provenance = {
        "request_id": report.get("request_id"),
        "prediction_id": report.get("prediction_id"),
        "model_version": report.get("model_version"),
        "feature_version": report.get("feature_version"),
        "data_cutoff_at": pred.get("data_cutoff_at"),
        "citations": explanation.get("citations", []),
        "limitations": explanation.get("limitations", []),
        "disclaimer": report.get("disclaimer"),
    }
    return {
        "status": report.get("status"),
        "reused": report.get("reused", False),
        "panels": {
            "context": context,
            "data_readiness": data_readiness,
            "prediction": prediction_panel,
            "why": why,
            "next_action": next_action,
            "evidence_provenance": evidence_provenance,
        },
    }
