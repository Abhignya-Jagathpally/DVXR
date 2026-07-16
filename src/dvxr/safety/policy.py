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
    # participant-/researcher-safe URGENT actions: shown when a clinician-only action is chosen, so
    # urgency is preserved (never downgraded to "continue monitoring"). Both flag clinician review.
    "CONTACT_APPROVED_CARE_CHANNEL": ActionSpec(
        "CONTACT_APPROVED_CARE_CHANNEL", "Contact your approved care channel now",
        frozenset({"researcher", "clinician", "participant"}), True),
    "AWAIT_CLINICIAN_REVIEW": ActionSpec(
        "AWAIT_CLINICIAN_REVIEW", "Awaiting clinician review — follow your existing safety plan",
        frozenset({"researcher", "clinician", "participant"}), True),
}

#: When a role may not receive the chosen (urgent) action, route to a role-permitted action that
#: PRESERVES urgency — never a benign monitoring action. The original action still fires internally as
#: the decision's ``system_action_id``.
_ROLE_SAFE_FALLBACK = {
    "ESCALATE_PER_APPROVED_PROTOCOL": "CONTACT_APPROVED_CARE_CHANNEL",
    "REVIEW_ELEVATED_RISK": "AWAIT_CLINICIAN_REVIEW",
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
    original = ACTION_REGISTRY[action_id]
    system_action_id = action_id                      # what the system does internally
    view_id = action_id                               # what this role's viewer is shown
    requires_review = original.requires_clinician_review
    if role not in original.permitted_roles:
        # DO NOT downgrade urgency. Route to a role-permitted action that preserves urgency and keep the
        # clinician-review flag from the ORIGINAL action; the original still fires as system_action_id.
        reasons = reasons + [f"role_{role}_not_permitted_for_{action_id}"]
        view_id = _ROLE_SAFE_FALLBACK.get(action_id, "AWAIT_CLINICIAN_REVIEW")
        requires_review = requires_review or ACTION_REGISTRY[view_id].requires_clinician_review
    spec = ACTION_REGISTRY[view_id]
    return ActionDecision(action_id=spec.action_id, policy_id=POLICY_ID, policy_version=POLICY_VERSION,
                          reason_codes=reasons, requires_clinician_review=requires_review,
                          system_action_id=system_action_id)
