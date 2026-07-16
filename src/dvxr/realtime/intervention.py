"""dvxr.realtime.intervention — transparent, rule-based adaptive interventions (JITAI).

Rules are DECLARATIVE and unit-testable; the deterministic rule output is the source of truth. Each
rule maps a physiological condition to an APPROVED policy action id (`dvxr.safety.policy`) — NOT to a
free-form medical instruction (spec §17: no hard-coded dosing/treatment prose in conditionals). The
message is a neutral, protocol-pointing description; specific clinical actions come from the versioned
policy registry, reviewable by a clinician. The Stage-8 LLM layer may rephrase, never originate/override.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class InterventionRule:
    name: str
    priority: int                      # higher = more urgent
    condition: Callable[[Dict], bool]
    recommendation: str
    action_id: str = "CONTINUE_MONITORING"   # approved policy action id (dvxr.safety.policy)


@dataclass(frozen=True)
class Recommendation:
    rule: str
    priority: int
    message: str
    action_id: str = "CONTINUE_MONITORING"


def _get(state: Dict, key: str, default=None):
    v = state.get(key, default)
    return default if v is None else v


# Declarative rule set. Each condition reads the monitor's per-update state dict. Messages are neutral
# and point to the approved protocol; the action_id is the authoritative, versioned next step.
RULES: List[InterventionRule] = [
    InterventionRule(
        "hypoglycemia_risk", 100,
        lambda s: _get(s, "glucose_now", 999) < 70
        or _get(s, "glucose_forecast", 999) < 70,
        "Glucose is low or trending low — verify the CGM reading and follow the approved low-glucose "
        "protocol.", action_id="ESCALATE_PER_APPROVED_PROTOCOL"),
    InterventionRule(
        "hyperglycemia_risk", 90,
        lambda s: _get(s, "glucose_now", 0) > 180
        or _get(s, "glucose_forecast", 0) > 180,
        "Glucose is elevated — review per the approved monitoring protocol and recheck soon.",
        action_id="REVIEW_ELEVATED_RISK"),
    InterventionRule(
        "glucose_dropping_fast", 80,
        lambda s: _get(s, "glucose_trend", 0) < -2.0 and _get(s, "glucose_now", 999) < 100,
        "Glucose is dropping quickly — verify the sensor and recheck within a few minutes.",
        action_id="VERIFY_SENSOR_AND_CGM"),
    InterventionRule(
        "high_stress", 70,
        lambda s: _get(s, "stress_band", "") in ("elevated", "high")
        or _get(s, "stress_probability", 0.0) >= 0.7,
        "Stress is elevated — a brief pause or paced-breathing may help; continue monitoring.",
        action_id="CONTINUE_MONITORING"),
    InterventionRule(
        "high_cognitive_load", 50,
        lambda s: _get(s, "cognitive_workload_risk", 0.0) >= 0.7,
        "High cognitive load detected — consider a brief break; continue monitoring.",
        action_id="CONTINUE_MONITORING"),
]


def evaluate_interventions(state: Dict, rules: List[InterventionRule] = RULES) -> List[Recommendation]:
    """Return the fired recommendations, most urgent first (deterministic)."""
    fired = []
    for r in rules:
        try:
            if r.condition(state):
                fired.append(Recommendation(r.name, r.priority, r.recommendation, r.action_id))
        except Exception:
            continue
    fired.sort(key=lambda x: x.priority, reverse=True)
    return fired
