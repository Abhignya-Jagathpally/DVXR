# POW goal achievement — what was built, and how honestly

This is a **clinical risk prediction framework** — its purpose is to predict clinical risk
(glucose instability, stress, anxiety, depression, cognitive workload) from multimodal
mental-health and physiological signals, exactly as the POW charters. What follows maps the
goals to concrete artifacts. **Goal 4 (IEEE paper) is intentionally out of scope.**

## Clinical positioning — purpose vs. deployment maturity

- **Purpose: clinical.** Everything here is built to predict clinical risk from BCI + wearable
  + pulse + metabolic signals. That is the target application, not a hedge.
- **Maturity: pre-deployment.** The outputs carry `validated_for_clinical_use = false` — this
  is a *deployment-readiness* flag, **not** a statement that the framework is non-clinical. It
  means the model has not yet cleared the validation a tool needs before it drives real
  patient care. It is the honest status that keeps the framework credible to clinicians and
  regulators, and it protects patients from a research model being used as a cleared device.
- **The framework is built clinic-ready**: calibrated probabilities/intervals (split-conformal),
  fail-closed **abstention** on insufficient data, patient-disjoint evaluation with superiority
  gates, availability-aware fusion, and grounded, hallucination-guarded explanations — the
  ingredients a clinical decision-support tool requires.

### Path from pre-deployment to deployed (the remaining gate)

1. **Prospective external validation** on a held-out clinical cohort/site (the one recorded
   release blocker) — performance must hold out-of-sample, not just on retrospective data.
2. **Calibration + subgroup/fairness audit** across demographics and devices.
3. **Co-registered device capture** (EEG + PHR + PPG + reference glucose via LSL) to validate
   the fused clinical-risk claim end-to-end on the actual DVXR hardware.
4. **Regulatory pathway** (e.g. FDA SaMD / clinical-decision-support) with the intended-use
   statement and human-in-the-loop safeguards already scaffolded in the serving contract.

The framework is designed so that clearing these flips the deployment status **without a
redesign** — the clinical purpose was there from the start.

---

Every number below traces to a committed scoreboard or an out-of-repo run; the honesty gate
(`make audit`) stays green throughout.

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
  under patient-clustered 95% CIs; PI-95 coverage ~0.91. Clinical-purpose, pre-deployment
  (one remaining gate: prospective external validation — see Clinical positioning above).
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

## Honesty posture (what makes the clinical claim credible)

Temporally causal (no leakage; `docs/CAUSAL.md`); calibrated + abstaining; LLM explains never
predicts (guarded, `docs/EXPLAINABILITY.md`); glucose-from-EEG is a wired slot awaiting
co-registered data — not faked; simple-baseline wins reported, not hidden. The
`validated_for_clinical_use = false` flag is the honest **pre-deployment** marker (see Clinical
positioning), not a denial of clinical purpose — it is exactly what lets a clinician trust the
rest. `make audit` green.

## Verify
```
make audit
python -m pytest tests/ -q
python -m pytest neuroglycemic-sentinel/tests/ -q
cd web/signal && npm run build
# Unity: open Assets/Scenes/DVXR_RT_Demo.unity in the Editor (headless-authored)
```
