"""dvxr.safety.policy — versioned, deterministic next-action policy engine (spec §14, §17).

The action after a prediction is chosen by rules over (risk_category, confidence, data_quality, role),
returning an APPROVED action id from a versioned registry — never free medical prose generated in a
Python conditional. The LLM later explains the selected action; it cannot pick a different one. Action
*text* lives in the registry metadata (data), not in control flow, so a clinician can review/version it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dvxr.contracts import ActionDecision

POLICY_ID = "DVXR-PILOT-ACTION-V1"
POLICY_VERSION = "1.0"


class PolicyError(RuntimeError):
    """Raised when a requested action id is not in the approved registry, or a role may not use it."""


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    title: str
    permitted_roles: frozenset
    requires_clinician_review: bool


#: The approved action registry (spec §17). Text/titles are data, reviewable and versioned.
ACTION_REGISTRY: Dict[str, ActionSpec] = {
    "CONTINUE_MONITORING": ActionSpec(
        "CONTINUE_MONITORING", "Continue monitoring",
        frozenset({"researcher", "clinician", "participant"}), False),
    "VERIFY_SENSOR_AND_CGM": ActionSpec(
        "VERIFY_SENSOR_AND_CGM", "Verify sensor quality and CGM feed",
        frozenset({"researcher", "clinician", "participant"}), False),
    "REVIEW_ELEVATED_RISK": ActionSpec(
        "REVIEW_ELEVATED_RISK", "Review elevated risk per approved monitoring protocol",
        frozenset({"researcher", "clinician"}), True),
    "ESCALATE_PER_APPROVED_PROTOCOL": ActionSpec(
        "ESCALATE_PER_APPROVED_PROTOCOL", "Escalate per the clinician-approved workflow",
        frozenset({"clinician"}), True),
    "INSUFFICIENT_DATA": ActionSpec(
        "INSUFFICIENT_DATA", "Abstain — request the missing inputs",
        frozenset({"researcher", "clinician", "participant"}), False),
}


def action_text(action_id: str) -> str:
    """Approved human-readable text for an action id (the app renders this, not LLM-authored prose)."""
    spec = ACTION_REGISTRY.get(action_id)
    if spec is None:
        raise PolicyError(f"unknown action id {action_id!r}")
    return spec.title


def select_action(
    *,
    abstained: bool = False,
    risk_category: Optional[str] = None,
    confidence: Optional[float] = None,
    data_quality: str = "unknown",
    role: str = "researcher",
) -> ActionDecision:
    """Deterministically choose an approved action (spec §14 table). Abstention and poor data are
    handled first (never present a definitive warning on bad data)."""
    reasons: List[str] = []

    if abstained:
        return _decision("INSUFFICIENT_DATA", ["abstained"], role)

    poor = data_quality in ("poor", "unusable", "unknown")
    high = risk_category == "high"
    elevated = risk_category in ("elevated", "high")
    confident = confidence is not None and confidence >= 0.7

    if poor and elevated:
        # high risk on poor data ⇒ request verification rather than a definitive warning (spec §9)
        return _decision("VERIFY_SENSOR_AND_CGM", ["elevated_risk", "poor_data_quality"], role)
    if high and confident:
        return _decision("ESCALATE_PER_APPROVED_PROTOCOL", ["high_risk", "high_confidence"], role)
    if elevated:
        return _decision("REVIEW_ELEVATED_RISK", ["elevated_predicted_risk"], role)
    if poor:
        return _decision("VERIFY_SENSOR_AND_CGM", ["poor_data_quality"], role)
    return _decision("CONTINUE_MONITORING", ["low_risk", "good_data"], role)


def _decision(action_id: str, reasons: List[str], role: str) -> ActionDecision:
    spec = ACTION_REGISTRY[action_id]
    if role not in spec.permitted_roles:
        # fall back to the safest role-permitted action rather than exposing a disallowed one
        reasons = reasons + [f"role_{role}_not_permitted_for_{action_id}"]
        action_id = "VERIFY_SENSOR_AND_CGM" if role != "participant" else "CONTINUE_MONITORING"
        spec = ACTION_REGISTRY[action_id]
    return ActionDecision(action_id=spec.action_id, policy_id=POLICY_ID, policy_version=POLICY_VERSION,
                          reason_codes=reasons, requires_clinician_review=spec.requires_clinician_review)
