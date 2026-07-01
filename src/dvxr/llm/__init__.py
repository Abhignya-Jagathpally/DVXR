"""dvxr.llm — provider-agnostic LLM insight layer (offline-safe)."""
from .client import LLMClient, OfflineLLM  # noqa: F401
from .insight import (  # noqa: F401
    CAVEAT,
    build_grounded_facts,
    clinician_summary,
    personal_insight,
    write_insight_report,
)
