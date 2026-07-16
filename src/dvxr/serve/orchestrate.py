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
from dvxr.safety.validators import GroundingError
from dvxr.prediction import AbstainingPredictionService, PredictionInputs, build_model_evidence
from dvxr.safety.policy import select_action
from dvxr.serve.snapshot import build_patient_snapshot

#: user role -> the consent purpose it exercises (spec §2 step 2).
_ROLE_PURPOSE = {"researcher": "research", "clinician": "clinical", "participant": "participant"}

MODEL_VERSION = "neuroglycemic-sentinel/research-stage"
FEATURE_VERSION = "features-2.0.0"


class ConsentError(PermissionError):
    """Raised when the patient has not consented to the requested purpose (fail-closed)."""


class IdempotencyConflict(ValueError):
    """Raised when a caller reuses an idempotency key with a DIFFERENT semantic request. The key maps
    to an already-stored prediction whose request fingerprint does not match — returning it would be
    wrong, so we reject (HTTP 409) rather than silently serve a mismatched result (spec §18)."""


#: report_type -> the modalities a valid report of that type needs. Anything spanning >1 modality
#: (the fused headline) needs synchronized data that does not exist ⇒ the default predictor abstains.
_REPORT_MODALITIES = {
    "stress_glucose_risk": tuple(sorted(GLUCOSE_FUSION_MODALITIES)),
    "glucose_risk": ("cgm",),
    "cgm_glucose_risk": ("cgm",),
}


def _cgm_history_from_events(events, *, tenant_id: str, patient_id: str, cutoff: str):
    """Assemble a causal CGM history frame (timestamp, glucose) from provenance events.

    ISOLATION (Gate A, spec §7): an event is admitted ONLY if it belongs to this exact
    ``tenant_id`` AND ``patient_id`` AND was observed at or before ``cutoff``. An event whose tenant
    or patient identity is MISSING is rejected (never inferred from the request), so a mixed event list
    can never let patient A's prediction read patient B's glucose. Returns None when no admissible CGM
    values remain (⇒ the predictor abstains)."""
    if not events:
        return None
    import pandas as pd
    rows = []
    for ev in events:
        if str(ev.get("modality")) != "cgm":
            continue
        ev_tenant = ev.get("tenant_id")
        ev_patient = ev.get("patient_id")
        if not ev_tenant or not ev_patient:          # missing identity ⇒ reject, never infer
            continue
        if str(ev_tenant) != str(tenant_id) or str(ev_patient) != str(patient_id):
            continue                                 # wrong tenant/patient ⇒ never admit
        val = ev.get("value", ev.get("glucose"))
        ts = ev.get("observed_at_utc") or ev.get("timestamp_utc") or ev.get("timestamp")
        if val is None or ts is None:
            continue
        rows.append({"timestamp": ts, "glucose": val})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if cutoff:
        df = df[df["timestamp"] <= pd.to_datetime(cutoff, errors="coerce")]
    return df.sort_values("timestamp").reset_index(drop=True) if len(df) else None


def _prediction_from_bundle(req: GenerateRequest, bundle, snapshot) -> RiskPrediction:
    """Map a predictor's immutable PredictionBundle onto the persisted RiskPrediction contract."""
    return RiskPrediction(
        request_id=req.request_id,
        patient_id=req.patient_id,
        report_type=req.report_type,
        tenant_id=req.tenant_id,
        risk=bundle.risk,
        risk_category=bundle.risk_category,
        confidence=bundle.confidence,               # == reliability (trust), NOT the decision margin
        ood_score=bundle.ood_score,
        data_quality=bundle.data_quality,
        missing_modalities=list(snapshot.missing_modalities),
        abstained=bundle.abstained,
        abstain_reason=bundle.abstain_reason,
        model_version=bundle.model_version or MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        calibration_version=bundle.calibration_version,
        data_cutoff_at=req.data_cutoff_at,
        snapshot_id=snapshot.snapshot_id,
    ).with_prediction_id()


def _scoped_idempotency_key(req: GenerateRequest):
    """Namespace a caller's idempotency key by tenant+patient+report_type so the same raw key issued
    for a different patient (or tenant) never returns the wrong patient's prediction (spec §18)."""
    if not req.idempotency_key:
        return None
    return f"{req.tenant_id}|{req.patient_id}|{req.report_type}|{req.idempotency_key}"


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
    predictor=None,
    event_repository=None,
    model_registry=None,
    retrieval=None,
) -> dict:
    """Run the Generate lifecycle and return the persisted report. Never trains a model.

    Idempotent: a repeated ``idempotency_key`` returns the already-stored prediction. Consent is
    fail-closed when ``require_consent`` (default) — an unknown/insufficient scope raises ConsentError
    and is audited. Every step appends to the audit store under the request id.

    The reproducible snapshot is built from ``events``; when ``events`` is omitted and an
    ``event_repository`` is supplied, the orchestrator FETCHES this (tenant, patient)'s events at/before
    the cutoff from the repository (so the API path produces a real snapshot, not an empty one). A
    ``model_registry`` (optional) records the active model version; ``retrieval`` (optional) supplies
    grounded citation sources for the explanation. Model-derived ``ModelEvidence`` is assembled and
    persisted with the prediction, so the retrieved report carries the same evidence a fresh one does.
    """
    req = request.with_request_id()
    audit_store.append({"request_id": req.request_id, "event": "generate.requested",
                        "patient_id": req.patient_id, "report_type": req.report_type,
                        "user_role": req.user_role, "actor_id": req.actor_id,
                        "tenant_id": req.tenant_id})

    if require_consent:
        purpose = _ROLE_PURPOSE.get(req.user_role, req.user_role)
        ok = consent_store is not None and consent_store.check(
            req.patient_id, purpose, tenant_id=req.tenant_id)
        if not ok:
            audit_store.append({"request_id": req.request_id, "event": "generate.denied.consent",
                                "patient_id": req.patient_id, "purpose": purpose})
            raise ConsentError(f"patient {req.patient_id!r} has no consent for purpose {purpose!r}")

    # idempotency — reuse the already-stored prediction for a repeated key (no recompute). The key is
    # SCOPED by tenant+patient+report_type so the same raw key cannot collide across patients/tenants
    # (spec §18): patient A's prediction can never be returned for patient B's request.
    scoped_key = _scoped_idempotency_key(req)
    if scoped_key:
        existing = prediction_store.get_by_idempotency_key(scoped_key, tenant_id=req.tenant_id)
        if existing is not None:
            # canonical-fingerprint guard: the stored prediction's request_id (a hash of tenant,
            # patient, report_type, horizons, cutoff, key) must match this request. A repeated key with
            # a DIFFERENT semantic request is a conflict — reject, never serve a mismatched result.
            stored_rid = existing.get("request_id")
            if stored_rid and stored_rid != req.request_id:
                audit_store.append({"request_id": req.request_id, "event": "generate.conflict",
                                    "actor_id": req.actor_id, "idempotency_key": req.idempotency_key,
                                    "stored_request_id": stored_rid})
                raise IdempotencyConflict(
                    f"idempotency key {req.idempotency_key!r} was already used for a different request")
            audit_store.append({"request_id": req.request_id, "event": "generate.reused",
                                "actor_id": req.actor_id,
                                "prediction_id": existing.get("prediction_id")})
            # Reconstruct the (deterministic) action + explanation from the stored prediction so a
            # reused report has the SAME shape as a fresh one. The prediction itself is not recomputed;
            # the persisted ModelEvidence rides along so the reused report is byte-for-byte comparable.
            return _assemble_report(req, RiskPrediction.from_dict(existing),
                                    existing.get("prediction_id"), reused=True,
                                    evidence=existing.get("evidence"))

    # fetch this (tenant, patient)'s events at/before the cutoff from the repository when the caller did
    # not pass them inline — this is what makes the API's snapshot real rather than always empty.
    if events is None and event_repository is not None:
        events = event_repository.window(req.patient_id, None, req.data_cutoff_at or None,
                                         tenant_id=req.tenant_id)

    # assemble the reproducible cutoff-bound snapshot (Gate 2) — only events <= cutoff are admitted
    snapshot = build_patient_snapshot(
        events or [], patient_id=req.patient_id, data_cutoff_at=req.data_cutoff_at,
        tenant_id=req.tenant_id, feature_version=FEATURE_VERSION,
        expected_modalities=GLUCOSE_FUSION_MODALITIES)
    audit_store.append({"request_id": req.request_id, "event": "snapshot.created",
                        "snapshot": snapshot.to_dict()})

    # produce the prediction via the injected service (Gate 3) — NEVER trains. The default is the
    # abstaining service; the fused headline always abstains (no synchronized data). A CGM-only
    # service returns a number only for a single-modality CGM request with real history.
    service = predictor if predictor is not None else AbstainingPredictionService()
    inputs = PredictionInputs(
        report_type=req.report_type,
        horizons_minutes=req.prediction_horizons_minutes,
        snapshot=snapshot,
        cgm_history=_cgm_history_from_events(events, tenant_id=req.tenant_id,
                                             patient_id=req.patient_id, cutoff=req.data_cutoff_at),
        requested_modalities=_REPORT_MODALITIES.get(req.report_type, tuple(sorted(GLUCOSE_FUSION_MODALITIES))),
        cutoff=req.data_cutoff_at or None,
    )
    bundle = service.predict(inputs)
    prediction = _prediction_from_bundle(req, bundle, snapshot)

    # model-derived evidence (spec §2 step 7) — from the snapshot + predictor signals, never the LLM
    evidence = build_model_evidence(prediction, snapshot, bundle)

    # record the active model version for traceability (spec §6, §10), when a registry is provided
    if model_registry is not None:
        active = model_registry.active(prediction.model_version.split("/")[0])
        if active:
            audit_store.append({"request_id": req.request_id, "event": "model.resolved",
                                "name": active.get("name"), "version": active.get("version")})

    # persist the prediction WITH its evidence so the retrieved report carries the same evidence
    persisted = {**prediction.to_dict(), "evidence": evidence.to_dict()}
    pid = prediction_store.put(persisted, idempotency_key=scoped_key)

    sources = _retrieve_sources(req, retrieval)
    report = _assemble_report(req, prediction, pid, reused=False,
                              evidence=evidence.to_dict(), sources=sources)
    audit_store.append({"request_id": req.request_id, "event": "generate.completed",
                        "actor_id": req.actor_id, "prediction_id": pid,
                        "abstained": prediction.abstained,
                        "action_id": report["action"]["action_id"]})
    return report


def _retrieve_sources(req: GenerateRequest, retrieval) -> list:
    """Fetch grounded citation sources for the explanation, tenant+patient scoped (never cross-patient).
    Returns [] when no retrieval backend or no question is supplied."""
    if retrieval is None or not req.question:
        return []
    try:
        return retrieval.search_patient(req.question, patient_id=req.patient_id,
                                        tenant_id=req.tenant_id, k=3)
    except Exception:  # noqa: BLE001 — retrieval is best-effort; a failure must not break Generate
        return []


def assemble_persisted_report(rec: dict, *, user_role: str) -> dict:
    """Rebuild the COMPLETE report (prediction + evidence + action + grounded explanation) from a stored
    prediction record, so GET /v1/predictions/{id} returns the same shape POST produced — not a bare
    prediction row. Deterministic: action + explanation are a pure function of the immutable prediction
    and its persisted evidence."""
    prediction = RiskPrediction.from_dict(rec)
    req = GenerateRequest(patient_id=prediction.patient_id, report_type=prediction.report_type,
                          tenant_id=prediction.tenant_id, user_role=user_role,
                          data_cutoff_at=prediction.data_cutoff_at, request_id=prediction.request_id)
    return _assemble_report(req, prediction, prediction.prediction_id, reused=True,
                            evidence=rec.get("evidence"))


def _assemble_report(req: GenerateRequest, prediction: RiskPrediction, prediction_id,
                     *, reused: bool, evidence=None, sources=None) -> dict:
    """Build the complete report dict from a prediction. Deterministic (action + explanation are a
    pure function of the immutable prediction + persisted evidence), so the fresh and idempotent-reuse
    paths return the identical shape — no key is present in one path and absent in the other."""
    action = _action_for(prediction, req.user_role)
    # a grounding failure must NEVER surface ungrounded prose or a 500 — fall back to a deterministic
    # safe explanation that shows no risk narrative and flags that grounding failed (spec §8).
    try:
        explanation = grounded_explanation(prediction.to_dict(), evidence=evidence,
                                           action=action.to_dict(), sources=sources or [])
        grounding_ok = True
    except GroundingError:
        explanation = _safe_fallback_explanation(action.to_dict())
        grounding_ok = False
    return {
        "request_id": req.request_id,
        "prediction_id": prediction_id,
        "status": "abstained" if prediction.abstained else "completed",
        "prediction": prediction.to_dict(),
        "evidence": evidence,
        "action": action.to_dict(),
        "explanation": explanation,
        "grounding_complete": grounding_ok,
        "model_version": prediction.model_version,
        "feature_version": prediction.feature_version,
        "reused": reused,
        "disclaimer": "Research-grade decision-support, not a diagnosis.",
    }


def _safe_fallback_explanation(action: dict) -> dict:
    """A minimal, fully-grounded explanation used when `grounded_explanation` raises. It surfaces the
    (policy-chosen) action and an abstention-style note — no risk numbers, no ungrounded prose."""
    from dvxr.safety.policy import action_text
    action_id = action.get("action_id", "")
    return {
        "risk_summary": "A grounded explanation could not be produced; no risk narrative is shown.",
        "prediction_horizon_minutes": [],
        "supporting_factors": [],
        "missing_or_stale_data": [],
        "uncertainty_statement": "The explanation layer abstained (grounding check failed).",
        "action_id": action_id,
        "action_explanation": action_text(action_id) if action_id else "",
        "citations": [],
        "limitations": ["This is research-grade decision-support, not a diagnosis."],
    }
