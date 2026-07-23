"""LLM-generated, hallucination-guarded explanations via the Claude API.

Every explanation is produced by Claude when ``ANTHROPIC_API_KEY`` is configured, then
**hard-validated**: the shipped text may contain no number that is not already in the
prediction body. If Claude emits an ungrounded number, the call is retried with the
offending values flagged; if it still fails (or the API is unavailable, times out, or
errors), a deterministic grounded renderer ships instead. So the response is *always*
grounded and the service never blocks on the LLM.

Latency is bounded three ways: (1) an in-process cache keyed by the grounded facts — an
identical prediction reuses its explanation in ~0 ms; (2) a fast model + small token
budget; (3) a hard timeout with deterministic fallback. The numeric prediction is computed
before this runs and is never changed here — the LLM phrases, it never predicts.

Config (env):
  ANTHROPIC_API_KEY       enables the Claude path
  DVXR_EXPLAINER          auto (default) | llm | deterministic
  DVXR_EXPLAINER_MODEL    model id (default: claude-haiku-4-5 — fast, low latency)
  DVXR_EXPLAINER_TIMEOUT  seconds (default: 8)
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Optional

_NUMBER = re.compile(r"(?<![A-Za-z0-9])-?\d+\.?\d*(?![A-Za-z0-9])")
_CACHE: "dict[str, Dict[str, Any]]" = {}
_CACHE_LOCK = threading.Lock()
_CACHE_CAP = 512

_SYSTEM = (
    "You explain a research-stage clinical risk model's output to a clinician. You are given "
    "the EXACT numbers the model produced. Restate them in 2-3 plain sentences.\n"
    "HARD RULES:\n"
    "- Use ONLY the numbers provided. Never compute, estimate, round to a new value, or "
    "introduce any number that is not in the facts.\n"
    "- You are explaining, not predicting: make no independent judgement and give no medical "
    "advice or treatment recommendation.\n"
    "- State that it is research-stage and not validated for clinical use.\n"
    "- Plain prose only. No preamble, no markdown, no bullet points."
)


def _module_ok(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


_LOCAL_PIPE: dict[str, Any] = {}


def _local_pipeline(model_id: str):
    """Cache a transformers text-generation pipeline for the local open-source backend."""
    if model_id not in _LOCAL_PIPE:
        from transformers import pipeline  # lazy — heavy import

        _LOCAL_PIPE[model_id] = pipeline("text-generation", model=model_id)
    return _LOCAL_PIPE[model_id]


# --------------------------------------------------------------------------- grounding
def _numbers_in(text: str) -> set[str]:
    out: set[str] = set()
    for token in _NUMBER.findall(text or ""):
        try:
            out.add(f"{float(token):.4g}")
        except ValueError:
            continue
    return out


def allowed_numbers(body: Dict[str, Any]) -> set[str]:
    """Every number the explanation may mention: any body value + its percent form."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            found.add(f"{float(value):.4g}")
            found.add(f"{float(value) * 100:.4g}")
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(body)
    return found


def is_grounded(text: str, allowed: set[str]) -> tuple[bool, set[str]]:
    ungrounded = {n for n in _numbers_in(text) if n not in allowed}
    return (not ungrounded), ungrounded


# --------------------------------------------------------------- deterministic fallback
def deterministic_explanation(body: Dict[str, Any]) -> Dict[str, Any]:
    """A grounded narrative that only restates numbers already in ``body`` (no LLM)."""
    selected = body.get("selected_outcome", {})
    name = selected.get("name", "the selected outcome")
    if body.get("status") == "abstained" or selected.get("probability") is None:
        missing = body.get("missing_or_stale_data") or selected.get("missing_or_stale_data") or []
        text = (
            f"The model abstained on {name}: the required inputs were not present, so no "
            f"probability was produced. Provide {', '.join(missing) if missing else 'the missing inputs'} "
            "to obtain a research-stage estimate. This is decision-support, not a diagnosis."
        )
        drivers: List[Dict[str, Any]] = []
    else:
        prob, band = selected.get("probability"), selected.get("risk_band")
        drivers = [
            {"feature": c.get("factor") or c.get("feature"), "direction": c.get("direction")}
            for c in body.get("contributions", [])[:3]
        ]
        driver_txt = "; ".join(f"{d['feature']} ({d['direction']})" for d in drivers if d.get("feature"))
        text = (
            f"Research-stage estimate for {name}: probability {prob} (risk band {band}). "
            + (f"Top contributors: {driver_txt}. " if driver_txt else "")
            + "Not validated for clinical use; this explains the model output and makes no "
              "independent prediction."
        )
    return _envelope(body, text, drivers, source="deterministic", model=None, grounded=True,
                     latency_ms=0.0)


def _envelope(body, text, drivers, *, source, model, grounded, latency_ms) -> Dict[str, Any]:
    selected = body.get("selected_outcome", {})
    return {
        "text": text,
        "grounded_on": {
            "status": body.get("status"),
            "selected_probability": selected.get("probability"),
            "risk_band": selected.get("risk_band"),
            "evidence_status": selected.get("evidence_status"),
            "validated_for_clinical_use": selected.get("validated_for_clinical_use", False),
        },
        "top_contributions": drivers,
        "predicts": False,
        "source": source,
        "model": model,
        "grounded": grounded,
        "latency_ms": round(float(latency_ms), 1),
    }


# --------------------------------------------------------------------------- LLM facts
def _facts(body: Dict[str, Any]) -> Dict[str, Any]:
    """The compact, safe set of facts handed to Claude (numbers it may restate)."""
    selected = body.get("selected_outcome", {})
    facts: Dict[str, Any] = {
        "outcome": selected.get("name"),
        "status": body.get("status"),
        "research_stage": True,
        "validated_for_clinical_use": selected.get("validated_for_clinical_use", False),
    }
    if body.get("status") == "abstained" or selected.get("probability") is None:
        facts["abstained"] = True
        facts["missing_inputs"] = body.get("missing_or_stale_data") or selected.get("missing_or_stale_data")
    else:
        facts["probability"] = selected.get("probability")
        facts["risk_band"] = selected.get("risk_band")
        facts["top_contributors"] = [
            {"factor": c.get("factor") or c.get("feature"), "direction": c.get("direction")}
            for c in body.get("contributions", [])[:3]
        ]
    return facts


def _signature(facts: Dict[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()[:24]


# ------------------------------------------------------------------------ Claude client
class LLMExplainer:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: int = 220,
        retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("DVXR_EXPLAINER_MODEL", "claude-haiku-4-5")
        self.timeout = float(timeout or os.environ.get("DVXR_EXPLAINER_TIMEOUT", "8"))
        self.max_tokens = max_tokens
        self.retries = retries

    def provider(self) -> str:
        """Which backend will generate text: 'claude', 'local', or 'none'.

        Explicit override via DVXR_EXPLAINER_PROVIDER; otherwise prefer a funded Claude key,
        then a local open-source model, then none (deterministic)."""
        forced = os.environ.get("DVXR_EXPLAINER_PROVIDER")
        if forced:
            return forced
        if self.api_key and _module_ok("anthropic"):
            return "claude"
        if os.environ.get("DVXR_LOCAL_MODEL") and _module_ok("transformers"):
            return "local"
        return "none"

    def available(self) -> bool:
        return self.provider() in {"claude", "local"}

    def _client(self):
        import anthropic

        return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

    def _prompt(self, facts: Dict[str, Any], flagged: Optional[set[str]]) -> str:
        import json

        guard = ""
        if flagged:
            guard = ("\nYour previous reply used these numbers that are NOT in the facts and are "
                     f"therefore forbidden: {sorted(flagged)}. Rewrite using only facts numbers.")
        return f"Facts (JSON):\n{json.dumps(facts)}\n\nWrite the explanation.{guard}"

    def _call(self, facts: Dict[str, Any], flagged: Optional[set[str]] = None) -> str:
        prov = self.provider()
        if prov == "local":
            return self._call_local(facts, flagged)
        message = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": self._prompt(facts, flagged)}],
        )
        return "".join(getattr(block, "text", "") for block in message.content).strip()

    def _call_local(self, facts: Dict[str, Any], flagged: Optional[set[str]] = None) -> str:
        """Open-source local model via transformers (e.g. Qwen2.5-Instruct, Llama-3.2-1B).

        Set DVXR_LOCAL_MODEL to a HF instruct model id. Same grounding guard applies; a
        GGUF/llama.cpp or Ollama backend can replace this generator without touching explain().
        """
        pipe = _local_pipeline(os.environ["DVXR_LOCAL_MODEL"])
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": self._prompt(facts, flagged)}]
        out = pipe(messages, max_new_tokens=self.max_tokens, do_sample=False,
                   return_full_text=False)
        return (out[0]["generated_text"] if out else "").strip()

    def explain(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Claude-generated, grounding-validated explanation with deterministic fallback."""
        import time

        mode = os.environ.get("DVXR_EXPLAINER", "auto")
        if mode == "deterministic" or (mode == "auto" and not self.available()):
            return deterministic_explanation(body)
        if mode == "llm" and not self.available():
            # explicitly requested LLM but unavailable: fall back, but label it clearly
            out = deterministic_explanation(body)
            out["source"] = "deterministic_fallback_llm_unavailable"
            return out

        facts = _facts(body)
        sig = _signature(facts)
        with _CACHE_LOCK:
            cached = _CACHE.get(sig)
        if cached is not None:
            return {**cached, "latency_ms": 0.0, "source": cached["source"] + "+cache"}

        allowed = allowed_numbers(body)
        drivers = facts.get("top_contributors", [])
        start = time.perf_counter()
        flagged: Optional[set[str]] = None
        for _ in range(self.retries + 1):
            try:
                text = self._call(facts, flagged=flagged)
            except Exception:  # noqa: BLE001 — API error/timeout: fall back, stay grounded
                out = deterministic_explanation(body)
                out["source"] = "deterministic_fallback_api_error"
                return out
            ok, ungrounded = is_grounded(text, allowed)
            if ok:
                latency = (time.perf_counter() - start) * 1000.0
                prov = self.provider()
                model_id = os.environ.get("DVXR_LOCAL_MODEL") if prov == "local" else self.model
                result = _envelope(body, text, drivers, source=prov,
                                   model=model_id, grounded=True, latency_ms=latency)
                with _CACHE_LOCK:
                    if len(_CACHE) >= _CACHE_CAP:
                        _CACHE.pop(next(iter(_CACHE)))
                    _CACHE[sig] = result
                return result
            flagged = ungrounded  # retry with the offending numbers flagged

        # Claude kept hallucinating numbers after retries → guaranteed-safe fallback.
        out = deterministic_explanation(body)
        out["source"] = "deterministic_fallback_ungrounded_llm"
        return out


_DEFAULT = LLMExplainer()


def explain(body: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level entry: LLM when configured, deterministic otherwise; always grounded."""
    return _DEFAULT.explain(body)
