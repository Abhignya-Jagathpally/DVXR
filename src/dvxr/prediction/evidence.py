"""dvxr.prediction.evidence — assemble ModelEvidence from a prediction + snapshot (spec §2 step 7).

Evidence is MODEL-DERIVED, never LLM-authored: modality quality and missing-data effects come from the
reproducible snapshot; uncertainty and OOD indicators come from the predictor's own reliability signals.
The grounded explanation layer may only phrase these values — it cannot invent contributions."""
from __future__ import annotations

from typing import Optional

from dvxr.contracts import ModelEvidence, PatientSnapshot, RiskPrediction, _stable_id
from dvxr.prediction.service import PredictionBundle


def build_model_evidence(prediction: RiskPrediction, snapshot: PatientSnapshot,
                         bundle: Optional[PredictionBundle] = None) -> ModelEvidence:
    """Derive the evidence object that accompanies a prediction.

    * ``modality_quality`` — the snapshot's per-modality quality scores (what the model actually saw).
    * ``missing_data_effects`` — a human-readable note per modality the report needed but lacked.
    * ``uncertainty`` — ``1 − reliability`` when a calibrated number was produced (higher ⇒ less certain).
    * ``ood_indicators`` — the predictor's out-of-distribution score, when available.
    * ``contributions`` — signed per-modality contributions; only populated for a real (non-abstained)
      single-modality prediction, since the fused product abstains and has no attributable contributions.
    """
    modality_quality = dict(snapshot.quality_by_modality or {})
    missing = list(prediction.missing_modalities or snapshot.missing_modalities or [])
    missing_effects = [f"{m} unavailable at cutoff — not contributing to this prediction" for m in missing]

    ood_indicators = {}
    uncertainty = None
    contributions = {}
    method = "none"
    if bundle is not None:
        if bundle.ood_score is not None:
            ood_indicators["cgm_window"] = float(bundle.ood_score)
        if bundle.reliability is not None:
            uncertainty = round(1.0 - float(bundle.reliability), 6)
        # a real single-modality prediction attributes to the modality it actually used
        if not prediction.abstained and bundle.modality_scope == "cgm_only":
            contributions = {"cgm": round(float(bundle.decision_margin or 0.0), 6)}
            method = "decision_margin"

    # every contribution gets an immutable evidence_id so an explanation's supporting factor cites it
    # (never source-free). The id hashes the prediction + feature, so it is reproducible.
    evidence_records = [
        {
            "evidence_id": _stable_id("ev", prediction.prediction_id, feature),
            "evidence_type": "model_evidence",
            "feature": feature,
            "value": float(value),
            "method": method,
            "model_version": prediction.model_version,
            "snapshot_id": prediction.snapshot_id,
        }
        for feature, value in contributions.items()
    ]

    return ModelEvidence(
        prediction_id=prediction.prediction_id,
        contributions=contributions,
        modality_quality=modality_quality,
        missing_data_effects=missing_effects,
        uncertainty=uncertainty,
        ood_indicators=ood_indicators,
        evidence_records=evidence_records,
    )
