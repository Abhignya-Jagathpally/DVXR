"""dvxr.llm.insight — grounded health insight over CACMF outputs (Stage 8).

Produces a plain-language personal insight and a clinician-facing summary. Every
sentence is grounded in the provided numbers; the layer EXPLAINS — it never
overrides or originates a calibrated prediction. A mandatory caveat line is always
present. Runs fully offline via the deterministic fallback.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Optional

from dvxr.llm.client import LLMClient
from dvxr.llm.prompts import (
    CLINICIAN_SYSTEM,
    CLINICIAN_USER,
    PERSONAL_SYSTEM,
    PERSONAL_USER,
)

CAVEAT = (
    "Caveat: this is a research prototype. Values are calibrated-model estimates and "
    "documented proxy signals, not a diagnosis; consult a qualified clinician for any "
    "medical decision."
)


def build_grounded_facts(bundle: Dict) -> str:
    """Deterministic bullet list of ONLY the numbers present in the bundle."""
    lines = []
    for task, info in bundle.get("tasks", {}).items():
        p = info.get("probability")
        band = info.get("band", "")
        if p is not None:
            lines.append(f"- {task}: probability {p:.2f}, band {band}")
    g = bundle.get("glucose")
    if g and g.get("now") is not None:
        seg = f"- glucose now {g['now']:.1f} mg/dL"
        if g.get("forecast") is not None:
            seg += (f", short-horizon forecast {g['forecast']:.1f}"
                    f" [{g.get('lower', float('nan')):.1f}, {g.get('upper', float('nan')):.1f}]")
        lines.append(seg)
    for name, val in bundle.get("biomarkers", {}).items():
        if val is not None:
            lines.append(f"- biomarker {name}: {val:.2f}")
    if bundle.get("top_modality"):
        lines.append(f"- most influential modality: {bundle['top_modality']}")
    for rec in bundle.get("interventions", []):
        lines.append(f"- suggested action: {rec}")
    return "\n".join(lines) if lines else "- no quantitative signals available"


def _ensure_caveat(text: str) -> str:
    return text if CAVEAT in text else text.rstrip() + "\n\n" + CAVEAT


def _generate(bundle: Dict, client: Optional[LLMClient], system: str, user_tmpl: str,
              offline_header: str) -> str:
    client = client or LLMClient()
    facts = build_grounded_facts(bundle)
    if client.is_offline:
        return _ensure_caveat(offline_header + "\n" + facts)
    text = client.complete([{"role": "user", "content": user_tmpl.format(facts=facts)}],
                           system=system)
    if not text.strip():
        text = offline_header + "\n" + facts
    return _ensure_caveat(text)


def personal_insight(bundle: Dict, client: Optional[LLMClient] = None) -> str:
    return _generate(bundle, client, PERSONAL_SYSTEM, PERSONAL_USER,
                     "Personal health summary (grounded):")


def clinician_summary(bundle: Dict, client: Optional[LLMClient] = None) -> str:
    return _generate(bundle, client, CLINICIAN_SYSTEM, CLINICIAN_USER,
                     "Clinician-facing summary (grounded):")


def write_insight_report(bundle: Dict, out_path: str | pathlib.Path = "outputs/insight_example.md",
                         client: Optional[LLMClient] = None) -> dict:
    client = client or LLMClient()
    personal = personal_insight(bundle, client)
    clinician = clinician_summary(bundle, client)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# CACMF Insight (backend: {client.backend_name})\n\n"
        f"## Personal insight\n\n{personal}\n\n## Clinician summary\n\n{clinician}\n")
    return {"personal": personal, "clinician": clinician,
            "backend": client.backend_name, "path": str(out)}
