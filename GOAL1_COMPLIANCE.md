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
