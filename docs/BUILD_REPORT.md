# DVXR build report — POW Goals 1–3

This report ties the POW's first three goals to concrete, verifiable artifacts produced in
this build. **Goal 4 (IEEE paper) is intentionally out of scope** — the intent was to build
the system, not write the paper.

Every number below traces to a committed scoreboard or an out-of-repo run directory; the
honesty posture is enforced by `make audit` (torch-free CI) and the invariants in
`.github/workflows/audit.yml`.

## Round 2 — best-in-class, empirically-justified models (headline)

The second round replaced the weak glucose result and questioned every modeling decision:

- **Glucose (the headline):** CGM-autoregressive forecasting on the **real CGMacros cohort**
  (45 subjects) — **RMSE 12.99 mg/dL @30 min** (persistence 17.40), 22.18/26.90/29.05 @60/90/120,
  PI-95 coverage ~0.91. Superiority gate **passed** (beats persistence at every horizon under
  patient-clustered 95% CIs); deterministic reproduction confirmed. A real jump from the last
  loop's ~31 and competitive with published CGM SOTA. Research-only (one blocker: prospective
  external validation). Details: `docs/MODEL_JUSTIFICATION.md`.
- **Why this model? (honest):** the same-split model ladder shows **gradient boosting is the
  best point forecaster at every horizon** (12.48 @30) — the deep net does not earn its
  complexity on point accuracy; it earns it via calibration + abstention + fusion. Every model
  beats persistence ~25% → the **representation is the win, not the architecture**.
- **Heads:** depression 0.961/0.986, WESAD stress 0.955, stress 0.892 — SOTA-competitive,
  protocol-labeled (`docs/HEADS_SOTA.md`). DEAP anxiety honest negative: at chance even at full
  128 Hz (`outputs/_r2/deap_fullrate_probe.md`) — decimation hypothesis refuted.
- **Trust:** explainability (3 layers), latency (<3 ms all paths), hallucination test, and
  device/EHR interoperability with interpretation payloads — `docs/{EXPLAINABILITY,INTEROPERABILITY}.md`,
  `outputs/latency_report.md`. Every question answered in `docs/DECISIONS_QA.md`.
- **Demo:** a DVXR-lab-aligned **BCI digital-twin skill experience** (web react-three-fiber +
  Unity scaffold) — decoded commands become avatar skills; the twin's glucose ring abstains
  when data is insufficient. `web/signal/src/components/rtdemo/`, `docs/UNITY_RT_DEMO.md`.

## Goal 1 — ingestion / pipeline for wearable, BCI, EHR, and diabetes data

A unified ingestion + encoder stack already lands heterogeneous modalities on the canonical
13-column event schema and real foundation-model encoders:

- Canonical schema + adapters: `src/dvxr/schemas.py`, `src/dvxr/encoders/` (EEG/CGM/EHR/
  notes/omics/wearable), real pretrained **LaBraM** via `encoders/labram_real.py`.
- Datasets wired: WESAD, DEAP, EEGMAT, Mumtaz-MDD (EEG); PhysioNet Non-EEG, CGMacros,
  Shanghai-CGM, BIG-IDEAS (physiology/CGM); MIMIC-IV demo + MTSamples (EHR/notes).
- **Glucose forecasting is now reachable from `dvxr`** via the out-of-process bridge
  `dvxr.integrations.GlucoseForecastingBridge` → the `neuroglycemic-sentinel` causal
  forecaster (`docs/GLUCOSE_FORECASTING.md`). Data/checkpoints stay out-of-repo in
  `neuroglycemic-runtime/`.
- Headline EEG screener result (committed, traceable): **depression MDD-vs-healthy AUROC
  0.961** (0.942–0.976) with the real LaBraM encoder — see `outputs/benchmark_scoreboard.md`
  and `BENCHMARK_FINDINGS.md`.

## Goal 2 — multimodal integration framework

- **Fusion**: CACMF cross-modal transformer + VQ codebook with five strategies
  (early/intermediate/late-weighted/attention/cross-modal) and three aggregators —
  `src/dvxr/fusion/`, `src/dvxr/config.py`. Absent modalities get a learned "absent" token
  (no silent imputation).
- **Availability-aware, abstaining product path**: `src/dvxr/serve/research_predict.py`
  scopes each target to its modality and abstains when inputs are missing;
  `dvxr.sentinel` fused reports abstain by construction (no synchronized per-subject
  EEG+CGM data exists).
- **Orchestration (new, this build)**: a LangGraph `StateGraph`
  (`src/dvxr/serve/agents/`) routes a request through ingestion → per-modality encoders →
  fusion → fail-closed calibration gate → grounded explanation, exposed at
  `POST /v1/research/predict/agentic`. Numeric output is **byte-identical** to the direct
  path (`tests/test_agentic_parity.py`); the graph orchestrates, it never computes a number.
- **EHR unstructured notes** (Goal-2 text modality): Bio_ClinicalBERT over 4,499 MTSamples
  notes, grouped CV — `outputs/clinical_notes_scoreboard.md` (specialty macro-AUROC 0.9606
  tfidf+lr; surgery 0.910 clinicalbert+lr).

## Goal 3 — single-vs-multimodal ablation (honest, including negatives)

The committed scoreboard is the Goal-3 comparison, reported faithfully — **learned fusion
does not currently beat the strongest non-fused baseline on any task** (a genuine negative
result, not hidden):

| task | best baseline | base err (1-AUROC) | fused err | RER % | meets ≥50%? | Holm p |
|---|---|---:|---:|---:|:--|---:|
| stress | rep:pca | 0.1079 | 0.1294 | −19.9 | False | 1.0 |
| wesad_stress | xgboost | 0.0453 | 0.1294 | −185.9 | False | 1.0 |
| deap_anxiety | single:physiology | 0.4658 | 0.4688 | −0.6 | False | 1.0 |
| deap_arousal | single:physiology | 0.4522 | 0.4575 | −1.2 | False | 1.0 |
| eegmat_workload | single:physiology | 0.2598 | 0.3649 | −40.4 | False | 1.0 |
| mumtaz_depression | sota | 0.0824 | 0.2046 | −148.2 | False | 1.0 |

Source: `outputs/benchmark_scoreboard.md` (repeats=5, folds=5, seed=7; grouped CV; bootstrap
CI + one-sided Wilcoxon + Holm). A win must beat BOTH floor and SOTA. `outputs/ablation_summary.md`
adds the fusion-strategy breakdown (e.g. glucose: CGM-only MAE 3.30 vs fusion ≥21.96 —
naïve access to CGM dominates; fusion adds nothing there).

**Glucose availability ablation (new, this build):** on the real BIG-IDEAS cohort, removing
the wearable modality drives prediction coverage to 0 (the model abstains rather than
guessing) — `neuroglycemic-runtime/runs/diatrend-style-bigideas/missing_modality_ablation.csv`.

## DiaTrend-style forecasting result + figures (this build)

Multi-horizon (30/60/90/120 min) causal glucose forecasting on the real BIG-IDEAS
wearable/CGM cohort (DUA-gated DiaTrend workbooks absent; substitute cohort labelled on
every figure; `prepare-diatrend` stays wired for real workbooks):

- Honest held-out metrics (10 patients, patient-disjoint): RMSE ~31 / MAE ~23 mg/dL at all
  horizons, PI-95 coverage ~0.91; `clinical_release_ready=false` with recorded release
  blockers; deterministic re-evaluation `passed=True` (max |diff| 1.4e-14 mg/dL).
- DiaTrend Figure-1-style overview suite (`neuroglycemic-sentinel/src/neuroglycemic/
  diatrend_figures.py`, driven by `scripts/build_diatrend_overview.py`): per-participant CGM
  traces, time-in-range, glucose distribution, data availability, cohort summary — written to
  `neuroglycemic-runtime/runs/diatrend-style-bigideas/figures/` (out-of-repo).

## Real-time RT Demo (this build)

`rt-demo-v1` streaming contract shared by a react-three-fiber web scene (`web/signal/src/
components/rtdemo/`, verifiable in-browser) and a Unity scaffold (`Assets/`, authored for
Editor). Avatar command is the `bci_real` analog; glucose abstains by construction; every
frame flagged experimental — `src/dvxr/serve/realtime_bridge.py`, `docs/RT_DEMO_STREAM_CONTRACT.md`.

## Multi-agent architecture (this build)

- Runtime: the LangGraph orchestration graph above (`dvxr[agents]` extra; ADR
  `docs/adr/0001-multi-agent-architecture.md` records langgraph-add / langchain-core+HF-reuse /
  full-langchain+tensorflow-reject).
- Dev-time: project Claude Code subagents (`.claude/agents/{honesty-auditor,benchmark-runner,
  figure-generator,dataset-ingestion}.md`) + a committed PostToolUse honesty gate
  (`.claude/hooks/honesty_gate.py`).

## Verify end-to-end

```bash
make audit                                             # torch-free honesty gate (73 tests)
python -m pytest tests/test_agentic_parity.py tests/test_realtime_bridge.py -q
python -m pytest neuroglycemic-sentinel/tests/test_diatrend_pipeline.py \
                 neuroglycemic-sentinel/tests/test_diatrend_figures.py -q
cd web/signal && npm install && npm run build          # r3f scene builds
# glucose forecasting + figures (out-of-repo, substitute cohort) via the sentinel CLI/bridge
# uvicorn "dvxr.serve.api:app" --factory  → ws://…/v1/realtime/stream ; POST /v1/research/predict/agentic
```
