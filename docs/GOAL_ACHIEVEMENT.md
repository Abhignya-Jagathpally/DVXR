# POW goal achievement — what was built, and how honestly

Maps the Proof-of-Work goals to concrete, verifiable artifacts. **Goal 4 (IEEE paper) is
intentionally out of scope.** Every number traces to a committed scoreboard or an out-of-repo
run; the honesty gate (`make audit`) stays green throughout.

## Goal 1 — LLM pipeline that ingests wearable, BCI, EHR & diabetes data

A device-agnostic ingestion + representation stack that predicts from the DVXR lab's own
streams — **EEG (Galea/EMOTIV) + PHR (wearable) + PPG (pulse) + CGM/meals**:

- **Canonical event schema + per-modality encoders**: real **LaBraM** (EEG), CGM-history
  features, **Bio_ClinicalBERT** (notes), physiology features (PHR/PPG). `src/dvxr/encoders/`,
  `src/dvxr/schemas.py`.
- **Interoperability with `neuroglycemic-sentinel`**: out-of-process bridge
  `dvxr.integrations.GlucoseForecastingBridge` (no pin/layout conflicts). `docs/GLUCOSE_FORECASTING.md`.
- **Device ingestion**: CGMacros builder (`prepare-cgmacros`, CGM + wearable/pulse HR + meals),
  DiaTrend/BIG-IDEAS builders; **LSL** stream contract (`eeg + wearable + reference_glucose`)
  for live Galea/EMOTIV/watch/pulse capture. `docs/GLUCOSE_FROM_DEVICES.md`, `docs/FRAMEWORK_OVERVIEW.md`.

## Goal 2 — multimodal integration framework

- **Fusion**: availability-aware mixture-of-experts glucose model (per-device experts →
  quality-gated fusion → residual over persistence → calibrated interval; abstains when no
  device usable) and the CACMF cross-modal transformer. `docs/MODEL_ARCHITECTURE.md`,
  `docs/MODEL_FLOW.md`, diagram `outputs/_r2/model_architecture.png`.
- **Multi-agent orchestration**: a **LangGraph** runtime graph wraps the pipeline (ingestion →
  encoders → fusion → fail-closed calibration gate → explanation) at
  `POST /v1/research/predict/agentic`, numeric-identical to the direct path.
  `src/dvxr/serve/agents/`, ADR `docs/adr/0001-multi-agent-architecture.md`.
- **Real-time + explanation**: RT-Demo streaming (`rt-demo-v1`), web react-three-fiber twin,
  Unity digital-twin scene; a **grounded LLM explainer** (Claude API or local open-source via
  transformers) that **explains, never predicts** — hard anti-hallucination guard + deterministic
  fallback (`src/dvxr/serve/llm_explainer.py`, `tests/test_llm_explainer.py`).

## Goal 3 — single-vs-multimodal ablation

- **Model ladder** (same patient-disjoint split): persistence → linear → tree → RF → **GBM** →
  MLP → deep net. Honest finding: gradient boosting ties/wins on point accuracy — the causal
  representation is the win, not depth. `docs/MODEL_JUSTIFICATION.md`, `outputs/_r2/glucose_model_ladder.csv`.
- **Per-device leave-one-out**: CGM dominates, meals +~0.4, wearable/pulse +0.01→0.44 (grows
  with horizon) and provides graceful degradation. `docs/GLUCOSE_FROM_DEVICES.md`.
- **Heads vs SOTA**: depression 0.961, WESAD stress 0.955 (SOTA-competitive, subject-held-out);
  DEAP anxiety honest negative (at chance even at full rate). `docs/HEADS_SOTA.md`,
  `outputs/benchmark_scoreboard.md`.

## Headline results (real, honest)

- **Glucose forecast** (CGMacros, 45 subjects, patient-disjoint, gate passed): RMSE
  **12.8 / 21.9 / 26.6 / 29.1 mg/dL** @30/60/90/120 min — beats persistence at every horizon
  under patient-clustered 95% CIs; PI-95 coverage ~0.91. Research-only (one blocker:
  prospective external validation).
- **Model-in-detail**: architecture + framework diagrams, model-flow mermaid, beautiful HTML
  figure (`docs/model_flow_diagram.html`).

## Multi-agent architecture — frameworks used, per requirement (not gratuitously)

| Piece | Used | Where |
|---|---|---|
| subagents | ✅ | `.claude/agents/{honesty-auditor,benchmark-runner,figure-generator,dataset-ingestion}.md` |
| hooks | ✅ | `.claude/settings.json` PostToolUse honesty gate |
| skills | ✅ | shipped skill library (artifact-design used for the HTML figure) |
| langgraph | ✅ | runtime orchestration graph (`dvxr[agents]`) |
| langchain-core | ✅ | sentinel `HealthAgent` explanation seat |
| huggingface | ✅ | LaBraM + Bio_ClinicalBERT via `transformers`/`huggingface_hub` |
| connectors | ✅ | Anthropic (Claude API) explainer; HF Hub weights |
| tools | ✅ | serve API endpoints, the sentinel bridge |
| **tensorflow** | ❌ **rejected** | the stack is torch — TF would double the footprint for zero benefit (honest call, ADR 0001) |

## Honesty posture (non-negotiable)

Temporally causal (no leakage; `docs/CAUSAL.md`); `validated_for_clinical_use=False` on every
output; LLM explains never predicts (guarded, `docs/EXPLAINABILITY.md`); glucose-from-EEG is a
wired, honest slot awaiting co-registered data — not faked; simple-baseline wins reported, not
hidden. `make audit` green.

## Verify
```
make audit
python -m pytest tests/ -q
python -m pytest neuroglycemic-sentinel/tests/ -q
cd web/signal && npm run build
# Unity: open Assets/Scenes/DVXR_RT_Demo.unity in the Editor (headless-authored)
```
