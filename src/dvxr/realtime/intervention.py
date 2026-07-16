"""dvxr.realtime.intervention — transparent, rule-based adaptive interventions (JITAI).

Rules are DECLARATIVE and unit-testable; the deterministic rule output is the source of truth. Each
rule maps a physiological condition to an APPROVED policy action id (`dvxr.safety.policy`) — NOT to a
free-form medical instruction (spec §17: no hard-coded dosing/treatment prose in conditionals). The
message is a neutral, protocol-pointing description; specific clinical actions come from the versioned
policy registry, reviewable by a clinician. The Stage-8 LLM layer may rephrase, never originate/override.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from dvxr.safety.policy import select_action


@dataclass(frozen=True)
class InterventionRule:
    name: str
    priority: int                      # higher = more urgent
    condition: Callable[[Dict], bool]
    recommendation: str
    risk_category: str = "low"         # low | watch | elevated | high — fed to the policy engine
    confidence: float = 0.8            # how confident this signal is (drives escalation vs review)


@dataclass(frozen=True)
class Recommendation:
    rule: str
    priority: int
    message: str
    action_id: str = "CONTINUE_MONITORING"    # AUTHORITATIVE — chosen by dvxr.safety.policy.select_action
    reason_codes: tuple = ()
    requires_clinician_review: bool = False


def _get(state: Dict, key: str, default=None):
    v = state.get(key, default)
    return default if v is None else v


# Declarative rule set. Each condition reads the monitor's per-update state dict and maps to a RISK
# CATEGORY (never a hard-coded action id) — the versioned policy engine (dvxr.safety.policy) is the
# single authority that turns risk + role + data quality into the approved next action (spec §14, §17).
RULES: List[InterventionRule] = [
    InterventionRule(
        "hypoglycemia_risk", 100,
        lambda s: _get(s, "glucose_now", 999) < 70 or _get(s, "glucose_forecast", 999) < 70,
        "Glucose is low or trending low — verify the CGM reading and follow the approved protocol.",
        risk_category="high", confidence=0.9),
    InterventionRule(
        "hyperglycemia_risk", 90,
        lambda s: _get(s, "glucose_now", 0) > 180 or _get(s, "glucose_forecast", 0) > 180,
        "Glucose is elevated — review per the approved monitoring protocol and recheck soon.",
        risk_category="elevated", confidence=0.6),
    InterventionRule(
        "glucose_dropping_fast", 80,
        lambda s: _get(s, "glucose_trend", 0) < -2.0 and _get(s, "glucose_now", 999) < 100,
        "Glucose is dropping quickly — verify the sensor and recheck within a few minutes.",
        risk_category="elevated", confidence=0.6),
    InterventionRule(
        "high_stress", 70,
        lambda s: _get(s, "stress_band", "") in ("elevated", "high")
        or _get(s, "stress_probability", 0.0) >= 0.7,
        "Stress is elevated — a brief pause or paced-breathing may help; continue monitoring.",
        risk_category="watch", confidence=0.5),
    InterventionRule(
        "high_cognitive_load", 50,
        lambda s: _get(s, "cognitive_workload_risk", 0.0) >= 0.7,
        "High cognitive load detected — consider a brief break; continue monitoring.",
        risk_category="watch", confidence=0.5),
]


def evaluate_interventions(state: Dict, rules: List[InterventionRule] = RULES, *,
                           role: str = "clinician",
                           data_quality: Optional[str] = None) -> List[Recommendation]:
    """Return the fired recommendations, most urgent first. The action id is NOT taken from the rule —
    it is chosen by the central policy engine (``select_action``) from the rule's risk category, the
    caller's role, and the data quality, so realtime and Generate share one action authority."""
    dq = data_quality if data_quality is not None else _get(state, "data_quality", "acceptable")
    fired = []
    for r in rules:
        try:
            if r.condition(state):
                decision = select_action(risk_category=r.risk_category, confidence=r.confidence,
                                         data_quality=dq, role=role)
                fired.append(Recommendation(
                    r.name, r.priority, r.recommendation, action_id=decision.action_id,
                    reason_codes=tuple(decision.reason_codes),
                    requires_clinician_review=decision.requires_clinician_review))
        except Exception:
            continue
    fired.sort(key=lambda x: x.priority, reverse=True)
    return fired
