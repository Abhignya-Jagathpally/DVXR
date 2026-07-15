"""dvxr.serve.vision — the research-stage glucose product surface (honest abstention).

The product headline is the **NeuroGlycemic Sentinel** glucose-excursion early-warning system, but
it is research-stage: the fused 30/60-minute claim requires synchronized same-subject
EEG+wearable+CGM pilot data that does not yet exist (`evidence.PRODUCT_VISION`). Rather than emit a
fabricated score, the default ``stress_glucose_risk`` report **abstains** — the spec's safe default
(§8.7, §16). This module produces that abstention as a structured object + a human-readable render, so
the CLI and the dashboard have a real, honest glucose surface today. A genuine prediction path lands in
PR5/PR6 once synchronized data + causal heads exist and the synchronized-same-subject gate is cleared.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from dvxr.serve.evidence import PRODUCT_VISION

#: The action the policy engine returns when required inputs are absent (spec §14). A formal,
#: versioned policy registry replaces this literal in PR7.
ABSTAIN_ACTION_ID = "INSUFFICIENT_DATA"


def glucose_risk_report(patient_id: Optional[str] = None,
                        horizons_minutes: Optional[List[int]] = None) -> Dict:
    """The default ``stress_glucose_risk`` report — an honest abstention while the product is
    research-stage. Returns a structured object (never a fabricated risk number)."""
    horizons = list(horizons_minutes or PRODUCT_VISION.horizons_minutes)
    return {
        "report_type": "stress_glucose_risk",
        "patient_id": patient_id,
        "status": "abstained",
        "research_stage": True,
        "prediction_horizons_minutes": horizons,
        "risk": None,                          # no fabricated probability
        "action_id": ABSTAIN_ACTION_ID,
        "reason_codes": ["no_synchronized_cohort", "fusion_claim_not_permitted"],
        "missing_or_stale_data": [
            "synchronized same-subject EEG+wearable+CGM pilot data",
        ],
        "risk_summary": ("A reliable glucose-excursion prediction cannot be produced yet: the fused "
                         "model requires synchronized same-subject EEG+wearable+CGM data, which does "
                         "not exist in this deployment. Fusion on unrelated public cohorts is blocked "
                         "by the synchronized-same-subject gate."),
        "uncertainty_statement": ("The product is research-stage and not yet validated; no calibrated "
                                  "glucose-excursion probability is available."),
        "model_version": "neuroglycemic-sentinel/research-stage",
        "disclaimer": "Research-grade decision-support, not a diagnosis.",
    }


def render_glucose_report(report: Optional[Dict] = None) -> str:
    """Human-readable render of the abstaining glucose report for the CLI/app."""
    r = report or glucose_risk_report()
    v = PRODUCT_VISION
    lines = [
        f"{v.name} — glucose-excursion early-warning",
        "=" * 60,
        v.tagline,
        f"Status: RESEARCH-STAGE — NOT YET VALIDATED   Horizons: {r['prediction_horizons_minutes']} min",
        "",
        f"Report: {r['report_type']}   →   {r['status'].upper()}  (action: {r['action_id']})",
        f"  {r['risk_summary']}",
        f"  Missing: {', '.join(r['missing_or_stale_data'])}",
        f"  Uncertainty: {r['uncertainty_statement']}",
        "",
        "Built from the validated components (see `dvxr report`): "
        + ", ".join(v.components),
        r["disclaimer"],
    ]
    return "\n".join(lines)
