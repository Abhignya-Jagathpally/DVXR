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

from dvxr.cohort import GLUCOSE_FUSION_MODALITIES
from dvxr.contracts import GenerateRequest, RiskPrediction
from dvxr.llm.grounded import grounded_explanation
from dvxr.safety.policy import select_action
from dvxr.serve.snapshot import build_patient_snapshot
from dvxr.serve.vision import glucose_risk_report

#: user role -> the consent purpose it exercises (spec §2 step 2).
_ROLE_PURPOSE = {"researcher": "research", "clinician": "clinical", "participant": "participant"}

MODEL_VERSION = "neuroglycemic-sentinel/research-stage"
FEATURE_VERSION = "features-2.0.0"


class ConsentError(PermissionError):
    """Raised when the patient has not consented to the requested purpose (fail-closed)."""


def _abstaining_prediction(req: GenerateRequest, snapshot_id: str = "") -> RiskPrediction:
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
        snapshot_id=snapshot_id,
    ).with_prediction_id()


def _action_for(prediction: RiskPrediction, role: str):
    """Protocol-controlled next action from the versioned policy engine (spec §14)."""
    return select_action(abstained=prediction.abstained, risk_category=prediction.risk_category,
                         confidence=prediction.confidence, data_quality=prediction.data_quality,
                         role=role)


def generate_risk_report(
    request: GenerateRequest,
    *,
    prediction_store,
    audit_store,
    consent_store=None,
    require_consent: bool = True,
    events=None,
) -> dict:
    """Run the Generate lifecycle and return the persisted report. Never trains a model.

    Idempotent: a repeated ``idempotency_key`` returns the already-stored prediction. Consent is
    fail-closed when ``require_consent`` (default) — an unknown/insufficient scope raises ConsentError
    and is audited. Every step appends to the audit store under the request id.

    ``events`` (optional) is the provenance-enriched event stream the snapshot is built from; when
    omitted an empty (but still reproducible) snapshot is recorded, tying the prediction to its cutoff.
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
            # Reconstruct the (deterministic) action + explanation from the stored prediction so a
            # reused report has the SAME shape as a fresh one. The prediction itself is not recomputed.
            return _assemble_report(req, RiskPrediction.from_dict(existing),
                                    existing.get("prediction_id"), reused=True)

    # assemble the reproducible cutoff-bound snapshot (Gate 2) — only events <= cutoff are admitted
    snapshot = build_patient_snapshot(
        events or [], patient_id=req.patient_id, data_cutoff_at=req.data_cutoff_at,
        feature_version=FEATURE_VERSION, expected_modalities=GLUCOSE_FUSION_MODALITIES)
    audit_store.append({"request_id": req.request_id, "event": "snapshot.created",
                        "snapshot": snapshot.to_dict()})

    # produce the prediction — NEVER trains; the research-stage glucose product abstains
    prediction = _abstaining_prediction(req, snapshot_id=snapshot.snapshot_id)
    pid = prediction_store.put(prediction.to_dict(), idempotency_key=req.idempotency_key)
    report = _assemble_report(req, prediction, pid, reused=False)
    audit_store.append({"request_id": req.request_id, "event": "generate.completed",
                        "prediction_id": pid, "abstained": prediction.abstained,
                        "action_id": report["action"]["action_id"]})
    return report


def _assemble_report(req: GenerateRequest, prediction: RiskPrediction, prediction_id,
                     *, reused: bool) -> dict:
    """Build the complete report dict from a prediction. Deterministic (action + explanation are a
    pure function of the immutable prediction), so the fresh and idempotent-reuse paths return the
    identical shape — no key is present in one path and absent in the other (spec §2, §8)."""
    action = _action_for(prediction, req.user_role)
    explanation = grounded_explanation(prediction.to_dict(), evidence=None, action=action.to_dict(),
                                       sources=[])
    return {
        "request_id": req.request_id,
        "prediction_id": prediction_id,
        "status": "abstained" if prediction.abstained else "completed",
        "prediction": prediction.to_dict(),
        "action": action.to_dict(),
        "explanation": explanation,
        "model_version": prediction.model_version,
        "feature_version": prediction.feature_version,
        "reused": reused,
        "disclaimer": "Research-grade decision-support, not a diagnosis.",
    }
