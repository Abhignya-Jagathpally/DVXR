"""dvxr.llm.client — provider-agnostic LLM client (ARCHITECTURE §A1 Stage 8).

Uniform ``.complete(messages, system=...)``. Default provider = Anthropic Claude
(model via env ``DVXR_LLM_MODEL``); OpenAI is pluggable. A mandatory ``OfflineLLM``
fallback composes a deterministic response when no API key / network / SDK is
available, so the layer NEVER hard-fails. API keys are read from the environment
only and never logged.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from dvxr.config import LLM_INSIGHT


class OfflineLLM:
    """Deterministic, no-network fallback. Echoes the user content verbatim so the
    caller (insight layer) controls exactly what — and which numbers — appear."""
    name = "offline-template"

    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None,
                 **opts) -> str:
        return "\n".join(m["content"] for m in messages if m.get("role") == "user")


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.provider = (provider or os.environ.get("DVXR_LLM_PROVIDER")
                         or LLM_INSIGHT["provider"])
        self.model = (model or os.environ.get("DVXR_LLM_MODEL")
                      or LLM_INSIGHT["model"])
        self._offline = OfflineLLM()
        self._backend = self._resolve(api_key)

    def _resolve(self, api_key: Optional[str]):
        if self.provider == "offline":
            return None
        try:
            if self.provider == "anthropic":
                key = api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    return None
                import anthropic
                return ("anthropic", anthropic.Anthropic(api_key=key))
            if self.provider == "openai":
                key = api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    return None
                import openai
                return ("openai", openai.OpenAI(api_key=key))
        except Exception:
            return None
        return None

    @property
    def is_offline(self) -> bool:
        return self._backend is None

    @property
    def backend_name(self) -> str:
        return self._offline.name if self.is_offline else f"{self.provider}:{self.model}"

    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None,
                 max_tokens: int = 700, **opts) -> str:
        """Return the model's text. Falls back to OfflineLLM on any error/missing key."""
        if self._backend is None:
            return self._offline.complete(messages, system=system)
        kind, client = self._backend
        try:
            if kind == "anthropic":
                resp = client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    system=system or "", messages=messages)
                return "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", "") == "text")
            if kind == "openai":
                msgs = ([{"role": "system", "content": system}] if system else []) + messages
                resp = client.chat.completions.create(
                    model=self.model, messages=msgs, max_tokens=max_tokens)
                return resp.choices[0].message.content or ""
        except Exception:
            return self._offline.complete(messages, system=system)
        return self._offline.complete(messages, system=system)
