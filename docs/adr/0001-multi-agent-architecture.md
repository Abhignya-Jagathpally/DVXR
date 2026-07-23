# ADR 0001 — Multi-agent orchestration for the research-prediction pipeline

Status: Accepted (2026-07-23)

## Context

Goals 1–2 of the POW call for a multimodal pipeline with an explainable, availability-aware
fusion path. The scoring logic already exists in `dvxr.serve.research_predict`
(`run_research_prediction`): per-target modality-scoped heads, a stacked selected outcome,
honest linear contributions, a forecast that abstains without a committed CGM artifact, and
fail-closed abstention. We want a genuine multi-agent architecture without weakening any of
those honesty invariants, and without re-implementing (and thus risking divergence from) the
committed scoring code.

## Decision

Add a **LangGraph** `StateGraph` as an *orchestration spine* that wraps the existing
functions as nodes:

    ingestion_availability -> encode_predict -> fuse_select -> forecast
        -> calibration_gate --(abstained?)--> explain_abstention -> END
                             \--(ok)---------> explain_prediction -> END

- Each node delegates to a `research_predict` helper; the numeric body the graph assembles
  is **byte-identical** to `run_research_prediction` (enforced by `tests/test_agentic_parity.py`).
- Modality scoping is expressed as node boundaries: a per-target node reads only its
  modality's feature slice (`_observed(features, TARGET_FEATURES[t])`).
- `calibration_gate` is the single fail-closed node that can set `status = "abstained"`;
  the conditional edge routes abstaining requests to the abstention explainer, which never
  revives a number.
- The explanation node ("LLM explains, never predicts") restates only values already frozen
  in the body (`predicts: False`). It is the seat where the sentinel `HealthAgent`/an LLM can
  be plugged in later; it is kept in-process and deterministic here to avoid importing the
  `src`-as-package sentinel tree into the dvxr interpreter.
- Exposed at `POST /v1/research/predict/agentic` (sibling route; the direct route is
  unchanged) and returns the identical body plus an additive per-node `trace` and
  `explanation`.

### Framework choices

| Framework | Decision | Rationale |
|---|---|---|
| langgraph | **Add** (optional `dvxr[agents]` extra) | Orchestration spine for conditional routing; replaces bespoke control flow, not any model. |
| langchain-core | **Reuse** (already pinned in sentinel) | Confined to the explanation seat. |
| huggingface / transformers | **Reuse** (already `dvxr[eeg]`) | LaBraM/encoder weights; no new dependency warranted. |
| langchain (full) | **Reject** | Only `langchain-core` primitives are used; the full meta-package adds autonomous tool-calling that conflicts with "agents never compute a number." |
| tensorflow | **Reject** | The entire stack is torch; adding TF would double the footprint and split the encoder ecosystem for zero benefit. |

### Dev-time agents & hooks

Project-scoped Claude Code subagents under `.claude/agents/` (`honesty-auditor`,
`benchmark-runner`, `figure-generator`, `dataset-ingestion`) and a committed
`.claude/settings.json` PostToolUse honesty gate (`.claude/hooks/honesty_gate.py`) that runs
the torch-free `tests.test_honesty_audit` after edits under `src/**` or
`neuroglycemic-sentinel/src/**` and blocks on regression.

## Consequences

- `langgraph` becomes an optional dependency; the base package and the torch-free
  `make audit` are unaffected (all agent imports are lazy). When the extra is absent, the
  agentic route falls back to the direct path.
- Installing `langgraph` downgraded `websockets` 16.1 → 15.0.1 (compatible with Starlette/uvicorn).
- We deliberately did **not** add a `Stop` hook running `make audit`: an end-of-turn blocking
  gate risks a loop when the tree is legitimately mid-refactor. The scoped PostToolUse gate is
  the invariant guard; `make audit` remains the CI gate (`.github/workflows/audit.yml`).
