"""dvxr.safety.validators — post-generation grounding checks (spec §8, §16).

A generated explanation is only allowed to surface facts that trace to the immutable prediction, the
model evidence, retrieved source passages, or the policy decision. These validators run AFTER the
language model and reject an explanation that: invents a number that does not match the prediction;
makes a clinical/protocol claim with no cited source; names an action id the policy engine did not
choose; or introduces unsupported diagnostic language. A failed validation ⇒ fall back to the
deterministic template / abstain, never show the ungrounded text.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

_NUM = re.compile(r"\d+(?:\.\d+)?%?")
# diagnostic / prescriptive language the explanation must never introduce (spec §8.8)
_FORBIDDEN_PHRASES = [
    "you have", "diagnos", "prescrib", "increase your insulin", "decrease your insulin",
    "take insulin", "adjust your dose", "you are diabetic", "confirmed", "definitely",
]


class GroundingError(ValueError):
    """Raised when a generated explanation fails a grounding check."""


def _numbers(text: str) -> List[str]:
    return _NUM.findall(text or "")


def allowed_number_strings(prediction: Dict, evidence: Optional[Dict] = None) -> set:
    """The set of number-strings an explanation may contain — derived from the prediction/evidence."""
    allowed = set()

    def _add(x):
        if isinstance(x, (int, float)):
            allowed.add(f"{x:.2f}".rstrip("0").rstrip("."))
            allowed.add(str(round(float(x) * 100)))            # percent form
            allowed.add(f"{round(float(x) * 100)}%")
            allowed.add(str(x))
    for v in (prediction.get("risk") or {}).values():
        _add(v)
    for k in ("confidence", "ood_score"):
        if prediction.get(k) is not None:
            _add(prediction[k])
    for hz in prediction.get("prediction_horizons_minutes", []) or []:
        allowed.add(str(hz))
    if evidence:
        for v in (evidence.get("contributions") or {}).values():
            _add(v)
    return allowed


def validate_numbers(text: str, prediction: Dict, evidence: Optional[Dict] = None) -> None:
    """Every number in the narrative must match a value from the prediction/evidence (spec §8.4)."""
    allowed = allowed_number_strings(prediction, evidence)
    for tok in _numbers(text):
        norm = tok.rstrip("%")
        norm_stripped = norm.rstrip("0").rstrip(".") if "." in norm else norm
        if tok not in allowed and norm not in allowed and norm_stripped not in allowed:
            raise GroundingError(f"ungrounded number in explanation: {tok!r}")


def validate_citations(claims: Iterable[Dict], source_ids: Iterable[str]) -> None:
    """Every claim must resolve to an existing evidence/retrieved source (spec §8.6).

    A claim with a MISSING or None ``source_id`` is rejected outright — a supporting factor that cites
    nothing is not grounded. A claim carrying a source_id that is not in the valid set is also rejected."""
    valid = set(source_ids)
    for c in claims:
        sid = c.get("source_id")
        if not sid:
            raise GroundingError(
                f"ungrounded claim (no source_id): {c.get('statement', c)!r}")
        if sid not in valid:
            raise GroundingError(f"claim cites a non-existent source: {sid!r}")


def validate_action_id(explained_action_id: str, policy_action_id: str) -> None:
    """The explanation's action id must equal the policy engine's choice (spec §8.5, §14)."""
    if explained_action_id != policy_action_id:
        raise GroundingError(
            f"explanation action {explained_action_id!r} != policy action {policy_action_id!r}")


def validate_no_diagnosis_language(text: str) -> None:
    """Reject unsupported diagnostic/prescriptive language (spec §8.8).

    The negated disclaimer ("not a diagnosis") is allowed; a positive diagnostic/prescriptive claim
    is not. If the only "diagnos" occurrence is inside the disclaimer, it passes."""
    low = (text or "").lower()
    negated_diagnosis = any(p in low for p in ("not a diagnos", "never a diagnos", "not a diagnostic"))
    for bad in _FORBIDDEN_PHRASES:
        if bad not in low:
            continue
        if bad == "diagnos" and negated_diagnosis:
            continue
        raise GroundingError(f"unsupported diagnostic/prescriptive phrase: {bad!r}")
