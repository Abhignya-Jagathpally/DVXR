"""dvxr.serve.orchestrate — the Generate request lifecycle (spec §2).

Clicking "Generate" must NOT train a model or recompute hours of signal processing; it produces a
validated, persisted, reproducible report from the latest prediction (spec §2 intro). This module is
the orchestration spine that ties the pieces built in earlier PRs together:

  register request + audit  →  consent check (fail-closed)  →  idempotency reuse  →
  produce prediction WITHOUT training (the research-stage glucose product ABSTAINS)  →
  policy action  →  persist prediction  →  audit completion.

It depends only on the storage Protocols (`dvxr.storage`) and the immutable contracts
(`dvxr.contracts`), so it is backend-agnostic and deterministic. The predictive model is never
trained here — a missing artifact yields an abstention, never an on-the-fly fit.
"""
from __future__ import annotations

from typing import Optional

from dvxr.contracts import ActionDecision, GenerateRequest, RiskPrediction
from dvxr.serve.vision import ABSTAIN_ACTION_ID, glucose_risk_report

#: user role -> the consent purpose it exercises (spec §2 step 2).
_ROLE_PURPOSE = {"researcher": "research", "clinician": "clinical", "participant": "participant"}

MODEL_VERSION = "neuroglycemic-sentinel/research-stage"
FEATURE_VERSION = "features-2.0.0"
POLICY_ID = "DVXR-PILOT-ACTION-V1"
POLICY_VERSION = "1.0"


class ConsentError(PermissionError):
    """Raised when the patient has not consented to the requested purpose (fail-closed)."""


def _abstaining_prediction(req: GenerateRequest) -> RiskPrediction:
    """The research-stage glucose product's prediction: an honest abstention, never a trained score."""
    report = glucose_risk_report(patient_id=req.patient_id,
                                 horizons_minutes=req.prediction_horizons_minutes)
    return RiskPrediction(
        request_id=req.request_id,
        patient_id=req.patient_id,
        report_type=req.report_type,
        risk=None,
        abstained=True,
        abstain_reason=report["risk_summary"],
        data_quality="unknown",
        missing_modalities=list(report["missing_or_stale_data"]),
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        data_cutoff_at=req.data_cutoff_at,
    ).with_prediction_id()


def _action_for(prediction: RiskPrediction) -> ActionDecision:
    """Protocol-controlled next action (spec §14). Abstention ⇒ INSUFFICIENT_DATA; a real prediction
    path (PR7 policy engine) will map risk×confidence×quality to a versioned action id."""
    if prediction.abstained:
        return ActionDecision(action_id=ABSTAIN_ACTION_ID, policy_id=POLICY_ID,
                              policy_version=POLICY_VERSION,
                              reason_codes=["no_synchronized_cohort", "fusion_claim_not_permitted"])
    return ActionDecision(action_id="CONTINUE_MONITORING", policy_id=POLICY_ID,
                          policy_version=POLICY_VERSION, reason_codes=["research_stage"])


def generate_risk_report(
    request: GenerateRequest,
    *,
    prediction_store,
    audit_store,
    consent_store=None,
    require_consent: bool = True,
) -> dict:
    """Run the Generate lifecycle and return the persisted report. Never trains a model.

    Idempotent: a repeated ``idempotency_key`` returns the already-stored prediction. Consent is
    fail-closed when ``require_consent`` (default) — an unknown/insufficient scope raises ConsentError
    and is audited. Every step appends to the audit store under the request id.
    """
    req = request.with_request_id()
    audit_store.append({"request_id": req.request_id, "event": "generate.requested",
                        "patient_id": req.patient_id, "report_type": req.report_type,
                        "user_role": req.user_role})

    if require_consent:
        purpose = _ROLE_PURPOSE.get(req.user_role, req.user_role)
        ok = consent_store is not None and consent_store.check(req.patient_id, purpose)
        if not ok:
            audit_store.append({"request_id": req.request_id, "event": "generate.denied.consent",
                                "patient_id": req.patient_id, "purpose": purpose})
            raise ConsentError(f"patient {req.patient_id!r} has no consent for purpose {purpose!r}")

    # idempotency — reuse the already-stored prediction for a repeated key (no recompute)
    if req.idempotency_key:
        existing = prediction_store.get_by_idempotency_key(req.idempotency_key)
        if existing is not None:
            audit_store.append({"request_id": req.request_id, "event": "generate.reused",
                                "prediction_id": existing.get("prediction_id")})
            return {"request_id": req.request_id, "prediction": existing, "reused": True,
                    "status": existing.get("abstained") and "abstained" or "completed"}

    # produce the prediction — NEVER trains; the research-stage glucose product abstains
    prediction = _abstaining_prediction(req)
    action = _action_for(prediction)
    pid = prediction_store.put(prediction.to_dict(), idempotency_key=req.idempotency_key)
    audit_store.append({"request_id": req.request_id, "event": "generate.completed",
                        "prediction_id": pid, "abstained": prediction.abstained,
                        "action_id": action.action_id})
    return {
        "request_id": req.request_id,
        "prediction_id": pid,
        "status": "abstained" if prediction.abstained else "completed",
        "prediction": prediction.to_dict(),
        "action": action.to_dict(),
        "model_version": prediction.model_version,
        "feature_version": prediction.feature_version,
        "reused": False,
        "disclaimer": "Research-grade decision-support, not a diagnosis.",
    }
