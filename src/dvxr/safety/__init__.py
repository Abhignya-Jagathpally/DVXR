"""dvxr.safety — the deterministic policy engine + grounding validators (spec §8, §14, §16).

The next action after a prediction is chosen by a versioned POLICY engine, not generated freely by
the language model (spec §14); the LLM may only explain the selected action. Every number and claim in
a generated explanation is checked against the immutable prediction/evidence/source objects
(spec §8 hallucination prevention). Abstention is a first-class, safe outcome (spec §16).
"""
from dvxr.safety.policy import (  # noqa: F401
    ACTION_REGISTRY,
    POLICY_ID,
    POLICY_VERSION,
    PolicyError,
    select_action,
)
from dvxr.safety.validators import (  # noqa: F401
    GroundingError,
    validate_action_id,
    validate_citations,
    validate_no_diagnosis_language,
    validate_numbers,
)
