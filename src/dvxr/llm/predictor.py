"""dvxr.llm.predictor — DEPRECATED shim.

The LLM-in-the-predictive-path probe moved to ``dvxr.experiments.llm_representation_probe`` to
make explicit that it is EXPERIMENTAL and NOT part of the product path (in the product the LLM is
explanation-only, ``dvxr.llm.insight``). This shim re-exports the probe so existing bench and
experiment importers keep working; new code should import from ``dvxr.experiments`` directly.
"""
from __future__ import annotations

import warnings

from dvxr.experiments.llm_representation_probe import *  # noqa: F401,F403
from dvxr.experiments.llm_representation_probe import (  # noqa: F401  explicit re-exports
    EXPERIMENTAL_ONLY,
    NOT_FOR_CLINICAL_INFERENCE,
    resolve_model_id,
)

warnings.warn(
    "dvxr.llm.predictor is deprecated; import the experimental probe from "
    "dvxr.experiments.llm_representation_probe (it is EXPERIMENTAL / not a clinical predictor).",
    DeprecationWarning,
    stacklevel=2,
)
