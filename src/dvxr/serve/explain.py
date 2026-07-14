"""dvxr.serve.explain — grounded, caveated explanation for a screening result.

Reuses the validated insight surface (dvxr.llm.insight) which EXPLAINS a calibrated prediction
and never originates one, always appends a research-prototype caveat, and runs fully offline via
a deterministic fallback. This just adapts one-or-more Screener results into the bundle that
surface consumes, plus a compact linear-attribution over the input dimensions for band-power
screeners (LaBraM latent dims are not human-named, so attribution is reported as latent-dim
influence, honestly labeled).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def screener_bundle(results: List[dict]) -> Dict:
    """Assemble the insight bundle from Screener.score_subject() results."""
    tasks = {}
    for r in results:
        tasks[r.get("label", r["task"])] = {"probability": float(r["probability"]),
                                            "band": r.get("risk_band", "")}
    return {"tasks": tasks}


def top_feature_attribution(screener, emb: np.ndarray, feature_names: Optional[List[str]] = None,
                            k: int = 5) -> List[dict]:
    """Linear attribution: standardized-input × head-coefficient, averaged over the subject's
    windows. For band-power screeners feature_names give human labels; for LaBraM these are
    latent dims (labeled as such, not over-interpreted)."""
    emb = np.asarray(emb, dtype=float)
    try:
        z = screener.scaler.transform(emb).mean(axis=0)
        coef = np.asarray(screener.head.coef_).ravel()
        contrib = z * coef
    except Exception:
        return []
    order = np.argsort(-np.abs(contrib))[:k]
    labeled = feature_names if (feature_names and len(feature_names) == len(contrib)) else None
    out = []
    for i in order:
        name = labeled[i] if labeled else (f"latent[{i}]" if screener.representation == "labram_eeg"
                                           else f"feature[{i}]")
        out.append({"feature": name, "contribution": round(float(contrib[i]), 4),
                    "direction": "raises" if contrib[i] > 0 else "lowers"})
    return out


def explain(results: List[dict], client=None) -> dict:
    """Grounded personal + clinician narrative for the screening result(s)."""
    from dvxr.llm.insight import clinician_summary, personal_insight, build_grounded_facts
    bundle = screener_bundle(results)
    return {"facts": build_grounded_facts(bundle),
            "personal": personal_insight(bundle, client),
            "clinician": clinician_summary(bundle, client)}
