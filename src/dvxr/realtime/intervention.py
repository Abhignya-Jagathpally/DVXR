"""dvxr.realtime.intervention — transparent, rule-based adaptive interventions (JITAI).

Rules are DECLARATIVE and unit-testable; the deterministic rule output is the source
of truth. The Stage-8 LLM layer may later rephrase these recommendations, but never
originates or overrides them.
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


@dataclass(frozen=True)
class Recommendation:
    rule: str
    priority: int
    message: str


def _get(state: Dict, key: str, default=None):
    v = state.get(key, default)
    return default if v is None else v


# Declarative rule set. Each condition reads the monitor's per-update state dict.
RULES: List[InterventionRule] = [
    InterventionRule(
        "hypoglycemia_risk", 100,
        lambda s: _get(s, "glucose_now", 999) < 70
        or _get(s, "glucose_forecast", 999) < 70,
        "Glucose is low or trending low — check CGM and take fast-acting carbs if needed."),
    InterventionRule(
        "hyperglycemia_risk", 90,
        lambda s: _get(s, "glucose_now", 0) > 180
        or _get(s, "glucose_forecast", 0) > 180,
        "Glucose is elevated — hydrate and follow your insulin plan; recheck soon."),
    InterventionRule(
        "glucose_dropping_fast", 80,
        lambda s: _get(s, "glucose_trend", 0) < -2.0 and _get(s, "glucose_now", 999) < 100,
        "Glucose is dropping quickly — recheck within a few minutes."),
    InterventionRule(
        "high_stress", 70,
        lambda s: _get(s, "stress_band", "") in ("elevated", "high")
        or _get(s, "stress_probability", 0.0) >= 0.7,
        "Stress is elevated — try a 2-minute paced-breathing or a short pause."),
    InterventionRule(
        "high_cognitive_load", 50,
        lambda s: _get(s, "cognitive_workload_risk", 0.0) >= 0.7,
        "High cognitive load detected — consider reducing task complexity or a brief break."),
]


def evaluate_interventions(state: Dict, rules: List[InterventionRule] = RULES) -> List[Recommendation]:
    """Return the fired recommendations, most urgent first (deterministic)."""
    fired = []
    for r in rules:
        try:
            if r.condition(state):
                fired.append(Recommendation(r.name, r.priority, r.recommendation))
        except Exception:
            continue
    fired.sort(key=lambda x: x.priority, reverse=True)
    return fired
