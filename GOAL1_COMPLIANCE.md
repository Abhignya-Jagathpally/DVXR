# Goal 1 Deliverable — Compliance Status

This document maps the Goal 1 deliverable to what the pipeline actually implements.
Run `python3 scripts/run_goal1_full.py` to exercise every capability below on
always-runnable synthetic fixtures.

Status legend: ✅ implemented · ⚠️ implemented as a baseline/proxy (documented) · ❌ not present

## Top-line: three broader modalities → model pipeline

| Modality | Status | Where |
|---|---|---|
| PHR via wearables | ✅ | `loaders.load_noneeg_*` (real PhysioNet Non-EEG: EDA/GSR, temp, HR, SpO2, motion) |
| EHR by clinicians | ✅ | `loaders.load_mimic_demo_ehr` (real MIMIC-IV demo labs + demographics) |
| Multi-omics | ✅ | `omics.py` — genomics/proteomics/metabolomics ingestion + `scripts/convert_omics_subject.py` |
| Foundation-model representation | ✅ | `neural_encoders.NeuralBiosignalEncoder` — torch BIOT-style transformer producing learned embeddings (replaces PCA) |

## Sub-deliverable 1: pipeline for wearable & BCI data

### Ingest signal types
| | Status | Where |
|---|---|---|
| a) physiological wearable signals | ✅ | Non-EEG loaders; `features.build_signal_windows` |
| b) EEG signals | ✅ | DEAP loader + EMOTIV/Galea converters |
| c) biosensor streams | ✅ | EDA/PPG/resp in canonical schema |
| d) behavioral metrics | ✅ | VR/AR `behavior` modality (gaze/interactions) via `ingest_vr_session.py` |
| e) diabetes monitoring | ✅ | `loaders.load_shanghai_cgm_*` (real CGM) |

### Data sources
| | Status | Where |
|---|---|---|
| a) Galea headset | ✅ | `scripts/convert_galea_subject.py` (EEG + EDA + PPG) |
| b) EMOTIV EPOC X / FLEX | ✅ | `scripts/convert_emotiv_subject.py` (14/32-ch EEG) |
| c) smart wearables | ✅ | Non-EEG wrist loaders |
| d) CGMs | ✅ | Shanghai CGM loader |
| e) mobile health devices | ⚠️ | Covered by the generic canonical-CSV loader (`loaders.load_canonical_csv`); no device-specific parser |
| f) VR/AR environments | ✅ | `scripts/ingest_vr_session.py` (head pose, gaze, HR overlay) |

### Literature review → model selection
✅ `registry.py` + `sota.py` document and score EEG foundation models (EEG-X, LaBraM,
EEGPT), biosignal transformers (BIOT, MOMENT), CGM (GluFormer), EHR (Med-BERT/BEHRT),
and affective/stress systems. `scripts/compare_sota_models.py` writes the scored report.

### Tasks
✅ All seven named tasks are implemented as trainable heads in `clinical_tasks.py`
(`stress_detection`, `anxiety_prediction`, `depression_risk`, `cognitive_workload`,
`glucose_instability`, `diabetes_complication`, `clinical_risk`), reusing the calibrated
`models.train_binary_classifier`.

⚠️ Where a dataset has no ground-truth label, the task uses a **transparent, documented
proxy** (see `clinical_tasks.clinical_tasks_table()` — e.g. cognitive workload = EEG
beta/alpha ratio; glucose instability = CGM coefficient of variation; clinical risk =
abnormal-lab fraction). Replacing proxies with labeled cohorts is the next data step.

⚠️ "Fine-tune the selected models": the neural encoder is **trained** (self-supervised
masked-feature reconstruction), not fine-tuned from published EEG-X/BIOT weights — those
weights are not bundled. `NeuralBiosignalEncoder.from_pretrained(path)` is the hook to load
real checkpoints when available.

## Expected outcomes
| Outcome | Status | Where |
|---|---|---|
| a) Standardized wearable/BCI ingestion framework | ✅ | canonical schema (`schemas.py`) + loaders + converters |
| b) EEG & physiological embedding pipelines | ✅ | `neural_encoders.py` (neural) + `encoders.py` (PCA baseline) |
| c) Real-time stress & glucose monitoring | ✅ | `realtime.RealtimeMonitor` / `realtime.stream_predictions` |
| d) Explainable neural & physiological biomarkers | ✅ | `biomarkers.physiological_biomarkers` + `neural_biomarker_saliency` (gradient saliency) |
| e) Personalized diabetes risk prediction | ✅ | `personalization.per_subject_normalize` + `PersonalizedCalibrator`; CGM forecaster with conformal intervals |

## Honest limitations
- The bundled neural encoder uses a real transformer architecture and self-supervised
  training, but **not published pretrained weights** — load those via `from_pretrained`.
- Several clinical tasks use **documented signal proxies**, not clinically-validated labels.
- Multi-omics, VR/AR, and device demos run on **synthetic fixtures** until real exports are
  supplied; every converter accepts real files in the same shape.
- Metrics on synthetic fixtures validate the pipeline, not scientific performance.

---

# Goals 1–3 (CACMF / `dvxr`) compliance map

The pipeline is packaged as **`dvxr`** implementing CACMF. `goal1_pipeline` stays
importable via re-export shims. Run everything with `python3 scripts/run_mmf_full.py`
(offline/CPU/deterministic). Status legend as above.

## Goal 1 — ingestion, encoders, tasks, real-time, explainability

| Deliverable | Status | Where |
|---|---|---|
| Canonical 13-column schema (unchanged) | ✅ | `dvxr/schemas.py` |
| Local `data/` profiling → schema report | ✅ | `dvxr/ingest/profile.py::profile_data_dir` → `outputs/data_schema_report.md` |
| VQ-codebook tokenizer (EMA, dead-code, straight-through, perplexity) | ✅ | `dvxr/encoders/codebook.py` |
| Per-modality adapters behind one `EncoderProtocol` | ✅ | `dvxr/encoders/*_adapter.py`, `base.py` |
| **Real foundation-model weights** (primary) + guarded fallbacks | ✅ | `dvxr.config.FOUNDATION_MODELS`; MOMENT verified running on CPU |
| Seven clinical tasks (documented proxies, no invented labels) | ✅/⚠️ | `dvxr/clinical_tasks.py` reused in `dvxr/tasks/heads.py` |
| Real-time fused streaming + adaptive intervention | ✅ | `dvxr/realtime/monitor.py`, `intervention.py` |
| Explainable biomarkers + neural saliency + attention + codebook | ✅ | `dvxr/explain/` |
| Personalization (per-subject normalize + calibrator) | ✅ | `dvxr/tasks/train.py::population_and_personalized_metrics` |
| LLM insight layer (explains only, offline-safe) | ✅ | `dvxr/llm/` |

## Goal 2 — multimodal fusion

| Deliverable | Status | Where |
|---|---|---|
| Early / intermediate / late / attention / cross-modal transformer | ✅ | `dvxr/fusion/strategies.py` |
| Weighted-late / ensemble-avg / confidence-weighted aggregation | ✅ | `dvxr/fusion/aggregate.py` |
| Cross-modal InfoNCE alignment + relative-loss weights | ✅ | `dvxr/tasks/losses.py` |
| Arbitrary missing-modality handling (learned absent token, masking) | ✅ | `dvxr/fusion/strategies.py` |
| `CACMFModel` end-to-end + latent/attention/weight export | ✅ | `dvxr/fusion/model.py` |

## Goal 3 — ablation (fused vs single-modality)

| Deliverable | Status | Where |
|---|---|---|
| Subject/patient-held-out splits | ✅ | `dvxr/eval/splits.py` |
| Per-task single vs fusion vs aggregation, AUROC/AUPRC/F1/acc/ECE + MAE/coverage | ✅ | `dvxr/eval/ablation.py` → `outputs/ablation_table.csv` |
| Honest reporting (fused ≥ single **not** assumed) | ✅ | `dvxr/eval/ablation.py::ablation_summary` |

## Goal 4 — IEEE paper scaffold

| Deliverable | Status | Where |
|---|---|---|
| IEEEtran skeleton, TODO placeholders (no fabricated results) | ✅ | `paper/main.tex`, `references.bib` |
| Result tables auto-filled from `outputs/`, every number traceable | ✅ | `dvxr/eval/paper.py`, `scripts/build_paper_tables.py` |

## CACMF honest limitations
- Ablation uses a frozen-encoder linear-probe protocol on synthetic fixtures — it
  validates the harness, not scientific performance; real metrics need real labeled data.
- `EEG-X`, `GluFormer`, `Med-BERT`, and `PH-LLM` have **no usable open weights**; verified
  real substitutes are wired (LaBraM, MOMENT/CGM-JEPA, Bio_ClinicalBERT, Claude API) and the
  originals recorded in `dvxr.config.FOUNDATION_MODELS`.
- Paper prose is placeholder TODO; only the result tables are auto-generated.
