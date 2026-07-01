# CACMF Refactor Plan (Prompt 0 — orientation, no code)

**Repo:** DVXR Goal-1 pipeline → **CACMF** (Cross-modal Aligned Codebook Multimodal Fusion).
**Baseline verified:** `python3 -m unittest discover -s tests` → **Ran 143 tests, OK** (Python 3.12.3, CPU, offline).
**This document is a plan only. No production code has been written. Awaiting review before Prompt 1.**

Companion artifact: `outputs/data_schema_report.md` (local `data/` profiling + canonical mapping).

---

## 0. Ground truth (verified from the codebase)

- One canonical 13-column event schema (`schemas.REQUIRED_EVENT_COLUMNS`) is the spine; every loader
  validates through `validate_events()`. **Do not break it — extend it.**
- Existing package `src/goal1_pipeline/` has ~22 modules. Verified public API (from orientation sweep):
  - **schemas** `REQUIRED_EVENT_COLUMNS, DataSummary, validate_events, summarize_events, ensure_columns`
  - **loaders** `load_noneeg_subject/_dataset, load_mimic_demo_ehr, load_shanghai_cgm_file/_dataset,
    load_canonical_csv, load_wesad_subject_pickle, load_deap_preprocessed_pickle`
  - **features** `build_stress_windows, build_deap_arousal_windows, build_signal_windows,
    build_glucose_forecast_table, latest_stress_feature_row, feature_columns`
  - **encoders** `EncoderRecommendation, RECOMMENDATIONS, FeatureEncoder, recommendation_table`
  - **neural_encoders** `NeuralBiosignalEncoder` (torch; `.fit_transform/.transform/.save/.from_pretrained/.gradient_saliency`)
  - **models** `TrainedModel, train_stress_classifier, train_arousal_classifier, train_binary_classifier, train_glucose_forecaster`
  - **calibration** `BinaryCalibrator, fit_platt_calibrator, classification_calibration_metrics,
    expected_calibration_error, risk_band, add_risk_bands, conformal_radius, interval_coverage`
  - **clinical_tasks** `ClinicalTask, CLINICAL_TASKS (7), derive_task_labels, train_clinical_task, clinical_tasks_table`
  - **registry** `ModelChoice, DatasetChoice, MODEL_CHOICES, DATASET_CHOICES, model_choice_table, dataset_choice_table`
  - **sota** `SotaModel, SOTA_MODELS, sota_model_table, selected_sota_table, write_sota_report`
  - **realtime** `RealtimeMonitor (.update/.reset), stream_predictions`; **streaming** `predict_latest_stress`
  - **biomarkers** `physiological_biomarkers, neural_biomarker_saliency`
  - **personalization** `per_subject_normalize, PersonalizedCalibrator (.fit/.predict)`
  - **omics** `generate_omics_like_table, load_omics_table, build_omics_features`
  - **reporting** `write_model_card`; **explain** `top_linear_contributors`; **sample_data** `generate_public_like_events, generate_deap_like_events`
  - **bci_real** (new, uncommitted) `EmotivRecording, GaleaRecording, ingest_emotiv, ingest_galea,
    epoch_emotiv, feature_cols, temporal_diffusion_map`
- **Deps:** numpy, pandas, scikit-learn, scipy, torch (lazy via `_require_torch()`). **pyyaml NOT present** →
  config must have a JSON fallback. torch import is already guarded and degrades gracefully.
- The 7 clinical tasks and their **documented proxies** are fixed (see `clinical_tasks.py:32-122`); reuse them,
  never invent labels. Proxies: anxiety = high EDA + low HRV; depression = low motion + low HRV;
  workload = EEG β/α ratio; glucose instability = CGM CV; complication = time>180 mg/dL; clinical = abnormal-lab fraction.
- Local real data present for **eeg, wearable_phys, cgm, ehr**; **omics & behavior have no real files** (synthetic only).
  `data/real/galea/` is empty — Galea == the OpenBCI `BoardGaleaBeta` session (now trimmed into `data/sample/openbci/`).

---

## 1. Target package layout (per ARCHITECTURE §A8)

```
src/dvxr/
  config.py            # CACMFConfig dataclass + YAML/JSON load/save; frozen defaults
  ingest/              # profile_data_dir, canonical mapping, moved converters, re-export loaders
  encoders/  base.py codebook.py eeg_adapter.py biosignal_adapter.py cgm_adapter.py ehr_adapter.py omics_adapter.py behavior_adapter.py
  fusion/    strategies.py aggregate.py model.py
  tasks/     heads.py losses.py train.py
  realtime/  monitor.py intervention.py
  explain/   attention_maps.py codebook_usage.py report.py (biomarkers re-export)
  llm/       client.py insight.py prompts/
  eval/      splits.py ablation.py metrics.py
src/goal1_pipeline/     # becomes thin re-export shims -> dvxr.* (backward compat)
scripts/   run_mmf_full.py run_ablation.py build_paper_tables.py
paper/                  # IEEEtran scaffold
docs/      ARCHITECTURE.md MASTER_BRIEF.md REFRACTOR_PLAN.md
configs/   default.yaml
```

**Backward-compat strategy:** physically move each `goal1_pipeline/X.py` into its `dvxr` subpackage, then
replace `goal1_pipeline/X.py` with `from dvxr.<subpkg>.X import *` (plus explicit re-exports of names tests import).
All 143 existing tests and every `scripts/*` entry point must keep passing unchanged.

---

## 2. Stage-by-stage plan (maps 1:1 to Prompts 1–11)

Legend: **New** = files created · **Touch** = files edited · **Offline** = how it stays CPU/no-network.

### Stage 1 — Scaffold, config, shims (Prompt 1)
- **New:** `dvxr/` tree with `__init__` exports; `config.py` (`CACMFConfig` with every §A7 hyperparameter:
  d=64, d_f=128, K=512, β=0.25, n_fusion_layers=4, n_heads=8, dropout=0.1, window=30/30s, mask_ratio=0.3,
  epochs=30, batch=64, τ=1.0, gumbel=False, uncertainty_weighting=False, fusion_strategy="cross_modal",
  aggregation="confidence_weighted", seed=7, all λ weights); `configs/default.yaml`;
  `dvxr/ingest/profile.py::profile_data_dir()`; `scripts/run_mmf_full.py` stub with `--profile`.
- **Touch:** move `goal1_pipeline/*` → `dvxr/*`; leave shims; `docs/ARCHITECTURE.md`, `docs/MASTER_BRIEF.md`.
- **Symbols:** `CACMFConfig, load, save, DEFAULTS, profile_data_dir`.
- **Risks:** shim breakage / circular imports; xlsx/wfdb/gz optional deps in profiler. **Mitigation:** re-export `*`
  plus explicit names; guard optional readers; unit import test `import goal1_pipeline, dvxr`.
- **Offline:** pyyaml optional (JSON fallback); profiler reads headers only.
- **DoD:** 143 tests still green; `run_goal1_full.py` unchanged.

### Stage 2 — VQ codebook tokenizer (Prompt 2)
- **New:** `dvxr/encoders/codebook.py::VectorQuantizer` (nn.Module): NN lookup, straight-through estimator,
  `L_vq` (commitment β), EMA updates, dead-code reinit, batch **perplexity**, optional Gumbel-softmax path;
  `VQBiosignalEncoder` subclassing `NeuralBiosignalEncoder`; `tests/test_codebook.py`.
- **Touch:** `neural_encoders` (preserve `fit_transform/transform/save/from_pretrained/gradient_saliency` exactly).
- **Risks:** codebook collapse; ST-grad correctness. **Mitigation:** EMA + dead-code reinit; test grad flow + perplexity∈(1,K].
- **Offline:** small fixture, fixed seed; skip if torch missing.

### Stage 3 — Per-modality adapters (Prompt 3)
- **New:** `dvxr/encoders/base.py::EncoderProtocol`; `eeg_adapter.py (EEGAdapter)`, `biosignal_adapter.py (BiosignalAdapter)`,
  `cgm_adapter.py (CGMAdapter)`, `ehr_adapter.py (EHRAdapter)`, `omics_adapter.py (OmicsAdapter)`,
  `behavior_adapter.py (BehaviorAdapter)`, `ModalityEncoderRegistry`; `tests/test_adapters.py`.
- **Every foundation-model path is a capability check** (import-guarded + weight-path-guarded + logs which encoder ran);
  fallbacks: EEG/biosignal → VQBiosignalEncoder/PCA; CGM → conformalized Ridge summary (mean/CV/MAGE/TIR/slope);
  EHR → tokenized-code timeline; omics/behavior → existing feature builders → linear proj to d.
- **Risks:** Galea vs EMOTIV variable channel count. **Mitigation:** channel-agnostic pooling to fixed d; unit test both.
- **Offline:** all adapters run via fallbacks with no network; uniform latent width d.

### Stage 4 — Fusion + aggregation (Prompt 4)
- **New:** `fusion/strategies.py` (early, intermediate w/ missing-modality mask tokens, late_weighted, attention, cross_modal transformer);
  `fusion/aggregate.py` (weighted_late, ensemble_avg, confidence_weighted via normalized-entropy c_m);
  `fusion/model.py::CACMFModel` (`.fuse/.attention_weights/.fusion_weights/latent exports`); `tests/test_fusion.py`.
- **Risks:** missing-modality masking correctness. **Mitigation:** attention mask for absent tokens; test drop-one-modality;
  assert α/weights sum to 1; confidence defers to higher-confidence modality on constructed example.

### Stage 5 — Heads, relative losses, training (Prompt 5)
- **New:** `tasks/heads.py` (6 softmax/logistic + 1 conformal forecasting, reuse `fit_platt_calibrator`, `conformal_radius`);
  `tasks/losses.py` (`L_total = Σλ_t L_task + λ_vq ΣL_vq + λ_rec ΣL_recon + λ_alg L_align`; class-weighted CE; Huber;
  InfoNCE; optional Kendall uncertainty σ_t); `tasks/train.py` (AdamW + warmup→cosine + grad-clip 1.0 + optional EMA;
  logs per-term losses + σ_t → `outputs/train_log.csv`); `tests/test_tasks_losses.py`.
- **Reuse** the 7 `CLINICAL_TASKS` + proxies verbatim; integrate `per_subject_normalize` + `PersonalizedCalibrator`;
  report **both** population and personalized held-out metrics.
- **Risks:** loss instability. **Mitigation:** finite-check + decrease-over-N-steps test on fixture; seed everything.

### Stage 6 — Real-time fused streaming + intervention (Prompt 6)
- **New:** `realtime/monitor.py::FusedRealtimeMonitor` (multi-modal rolling buffer; keeps `.update/.reset` compatible
  with `RealtimeMonitor`); `realtime/intervention.py` (declarative JITAI rules; deterministic = source of truth);
  streaming demo in `run_mmf_full.py` → `outputs/realtime_fused_stream.csv`; `tests/test_realtime_fused.py`.
- **Risks:** missing-modality steps. **Mitigation:** still emit output + present-modality list; interventions fire only on
  constructed threshold crossings; determinism test.

### Stage 7 — Explainability bundle (Prompt 7)
- **New:** `explain/attention_maps.py` (α_m, w_m → `outputs/fusion_attention.csv`);
  `explain/codebook_usage.py` (per-modality histogram, perplexity, top codes per positive label — association not causation);
  `explain/report.py::explain_prediction()` → physio biomarkers + neural saliency + attention + active codes →
  `outputs/explanation_example.md`; `tests/test_explain.py`. Reuses `physiological_biomarkers`, `neural_biomarker_saliency`.

### Stage 8 — LLM insight layer (Prompt 8)
- **New:** `llm/client.py` (provider-agnostic `.complete()`; default Anthropic Claude via env `DVXR_LLM_MODEL`,
  pluggable others, **mandatory `OfflineLLM` deterministic-template fallback**); `llm/insight.py` (grounded personal +
  clinician summaries with mandatory caveat line, no new clinical claims); `llm/prompts/`; `run_mmf_full.py --insight`
  → `outputs/insight_example.md`; `tests/test_llm.py` (no live calls).
- **Guardrail:** LLM explains, never overwrites a calibrated prediction. Keys from env only, never logged.
- **Anthropic note:** default model id will follow current API skill guidance at implementation time; offline fallback
  is the tested path so no key is ever required for green tests.

### Stage 9 — Goal-3 ablation harness (Prompt 9)
- **New:** `eval/splits.py` (subject/patient-held-out), `eval/ablation.py` (per task: each single modality, each fusion
  strategy, each aggregator; AUROC/AUPRC/F1/acc/ECE + MAE/coverage; records present modalities);
  `scripts/run_ablation.py` → `outputs/ablation_table.csv` + `outputs/ablation_summary.md`; `tests/test_ablation.py`.
- **Honest metrics:** do NOT assert fused ≥ best-single — report reality. Note existing `run_fusion_ablation.py` finding
  (EEG dominant, fusion ≈ best single) as a precedent.

### Stage 10 — Goal-4 IEEE paper scaffold (Prompt 10)
- **New:** `paper/main.tex` (IEEEtran; abstract/intro/related/method=CACMF/experiments/ablation/limitations/refs),
  `paper/references.bib` (stubs for EEG-X, LaBraM, BENDR, BIOT, MOMENT, GluFormer, Med-BERT/BEHRT, PHIA/PH-LLM/Health-LLM,
  MedFuse/MedPatch, WESAD/DEAP/MIMIC-IV/PhysioNet); `scripts/build_paper_tables.py` (reads `outputs/*` →
  `paper/tables/*.tex`, booktabs, every number traceable); `Makefile` `paper` target (skip if no pdflatex);
  `tests/test_paper_tables.py`. **No fabricated numbers.** Can seed prose from existing `PAPER_DRAFT.md`.

### Stage 11 — End-to-end wiring, docs, compliance, QA (Prompt 11)
- **Touch:** `scripts/run_mmf_full.py` (profile→ingest→VQ encoders→all fusions→multi-task train→fused stream+intervention→
  explain→offline insight→ablation→paper tables); `README.md` (dvxr quickstart + diagram); `GOAL1_COMPLIANCE.md`
  (extend to Goals 1–3 map with ✅/⚠️/❌ + file pointers); `Makefile` `all` target; final self-audit vs POW + guardrails.
- **DoD:** `unittest discover` all green; `run_mmf_full.py` completes offline/CPU/deterministic; `run_goal1_full.py` still works.

---

## 3. Cross-cutting guardrails (enforced every stage)

1. **Always-runnable first** — every stage runs on synthetic fixtures, no network, no GPU. Foundation models &
   LLM APIs are optional adapters behind capability checks; degrade to baseline and say so.
2. **LLMs explain, don't replace** — calibrated models produce numbers; LLM narrates only.
3. **Honest metrics** — subject/patient-held-out for real; fixtures validate plumbing, not science; keep Caveats sections.
4. **No fabricated labels** — proxies stay clearly labeled as proxies.
5. **Backward compatibility** — `goal1_pipeline` stays importable via shims; existing scripts/tests unchanged.
6. **Determinism** — seed torch + numpy everywhere (seed=7); reproducible embeddings/metrics.
7. **Small reviewable commits** — one stage = one focused diff + its tests.

---

## 4. Sequencing, dependencies, and orchestration

- **Strict order:** 1 → 2 → 3 → 4 → 5 → {6,7 parallelizable} → 8 → 9 → 10 → 11. Stage 2 blocks 3; 3 blocks 4; 4 blocks 5;
  5 blocks 6/7/9. 8 and 10 depend on outputs of 5/9.
- **Multi-agent execution (post-approval):** each stage can be a focused subagent (implement + its tests), with an
  adversarial verify pass (run the stage's tests + the full suite) before commit. Stages 6/7 can run in parallel;
  10's table-builder verified against fixture outputs. A completeness critic checks guardrail adherence per stage.
- **Per-stage gate:** `python3 -m unittest discover -s tests` stays green AND the new stage test passes AND
  `run_goal1_full.py` still runs — before moving on.

---

## 5. Key risks & open questions for reviewer

1. **Scope** — this is 10 implementation stages + a paper scaffold. Confirm you want the full CACMF build now, or a
   subset first (e.g., Stages 1–5 core, defer LLM/paper).
2. **`data/sample` fixtures (91 MB)** — `deap_like_events.csv` (68 MB) + `canonical_events.csv` (23 MB) are large
   generated fixtures. Recommend regenerating on demand / gitignoring rather than committing. Confirm.
3. **Foundation-model weights** — none are bundled; all primary adapters (EEG-X/LaBraM/BIOT/MOMENT/GluFormer/Med-BERT)
   run as guarded stubs with baseline fallbacks. Real weights are a later, optional data step. Confirm that's acceptable.
4. **Novelty claim** — "CACMF is first of its kind" will be stated as *our contribution*, to be re-checked against
   literature at write-up, not asserted as settled fact.
5. **Raw BCI resolution** — full-resolution raw EEG CSVs were dropped (only 5-record samples remain). Real BCI training
   metrics can no longer be regenerated locally from those files; ablation/real metrics for BCI will note this.

---

## 6. Definition of done (whole project)

`dvxr` implements CACMF: VQ-codebook tokenization; all five fusion strategies + three aggregation baselines; unified
multi-task objective with tunable relative-loss weights; real-time fused streaming with adaptive intervention;
explainability bundle (physio + neural + attention + codebook); optional provider-agnostic LLM insight with offline
fallback; Goal-3 ablation harness; Goal-4 IEEE paper scaffold with auto-filled tables. All green on
`python3 -m unittest discover -s tests`, plus one-command `python3 scripts/run_mmf_full.py`, offline/CPU/deterministic,
with `goal1_pipeline` still importable and its tests passing.
