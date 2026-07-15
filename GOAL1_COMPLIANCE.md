# Goal 1 Deliverable — Compliance Status

This document maps the Goal 1 deliverable to what the pipeline actually implements.
Run `python3 scripts/run_goal1_full.py` to exercise every capability below on
always-runnable synthetic fixtures.

Status legend: ✅ implemented (real data) · 🧪 implemented, runs on **synthetic fixtures only** (no
real exports supplied) · ⚠️ implemented as a baseline/proxy (documented) · ❌ not present

## Divergence from the proposal (honest scope)

The delivered work is an honest, evidence-driven **pivot** from the original Plan of Work, not a
fulfillment of its central bet. Stated plainly so every table below is read in context:

- **Thesis pivot — the LLM-fusion bet does not win.** The POW centered *LLM-based multimodal fusion*
  as the contribution. In the code, learned cross-modal fusion (CACMF) **loses on all six real tasks**
  and the LLM-in-the-predictive-path is off-by-default and weakest. The delivered flagship is a
  **single-modality** EEG depression screener (frozen LaBraM probe, AUROC ≈0.96) — a foundation model,
  not an LLM, and not fusion. The LLM's validated role is **explanation only** (`dvxr/llm/insight.py`).
  The contribution that *does* hold is a **reliability-gated do-no-harm late fusion** that beats the
  learned CACMF fusion on 4/6 tasks (see `BENCHMARK_FINDINGS.md`).
- **Cross-modal fusion is structurally untested.** **No single dataset co-registers EEG+CGM+EHR per
  subject**, so the proposal's core cross-domain integration hypothesis is never actually tested;
  fusion evidence lives *within* a modality family (peripheral-physiology stress), not across the full
  EEG+wearable+CGM+EHR span. This is a structural gap, not merely a negative result.
- **Diabetes / omics / VR-AR are under-delivered.** Real CGM *forecasting* works (Shanghai, MAE
  ~11 mg/dL), but the one diabetes-*risk classification* signal had label leakage and is **permanently
  excluded** from claims (`dvxr.serve.evidence.EXCLUDED_CLAIMS`). Multi-omics and VR/AR are implemented
  as converters but run on **synthetic fixtures** (🧪) — no real exports.
- **"Fine-tune the selected models" → train-from-scratch.** The neural encoder is trained via
  self-supervision, *not* fine-tuned from published checkpoints. Real weights load only for wearable
  (MOMENT), EHR (Bio_ClinicalBERT), omics (Geneformer), and EEG (LaBraM, as a frozen probe in the
  product screener). Proposal-named EEG-X, GluFormer, and Med-BERT have **no usable open weights** and
  degrade to baselines.
- **Scoped future work (explicit, not implied wins):** the LLM-as-predictor path and a genuine
  cross-modal fusion win on a **co-registered** multimodal cohort remain open goals, recorded here so
  they are never read as delivered.

The validated product claims and their exclusions are machine-checked in `dvxr.serve.evidence`
(`PRODUCT_CLAIMS` / `EXCLUDED_CLAIMS`, enforced by `tests/test_honesty_audit.py`); this document and
`README.md` defer to that registry as the single source of truth.

## Top-line: three broader modalities → model pipeline

| Modality | Status | Where |
|---|---|---|
| PHR via wearables | ✅ | `loaders.load_noneeg_*` (real PhysioNet Non-EEG: EDA/GSR, temp, HR, SpO2, motion) |
| EHR by clinicians | ✅ | `loaders.load_mimic_demo_ehr` (real MIMIC-IV demo labs + demographics) |
| Multi-omics | 🧪 | `omics.py` — genomics/proteomics/metabolomics ingestion + `scripts/convert_omics_subject.py`. **Synthetic fixtures only** — no real omics exports; Geneformer wired but unexercised on real data. |
| Foundation-model representation | ✅ | `neural_encoders.NeuralBiosignalEncoder` — torch BIOT-style transformer producing learned embeddings (replaces PCA) |

## Sub-deliverable 1: pipeline for wearable & BCI data

### Ingest signal types
| | Status | Where |
|---|---|---|
| a) physiological wearable signals | ✅ | Non-EEG loaders; `features.build_signal_windows` |
| b) EEG signals | ✅ | DEAP loader + EMOTIV/Galea converters |
| c) biosensor streams | ✅ | EDA/PPG/resp in canonical schema |
| d) behavioral metrics | 🧪 | VR/AR `behavior` modality (gaze/interactions) via `ingest_vr_session.py` — **synthetic fixtures only** |
| e) diabetes monitoring | ✅ | `loaders.load_shanghai_cgm_*` (real CGM) |

### Data sources
| | Status | Where |
|---|---|---|
| a) Galea headset | ✅ | `scripts/convert_galea_subject.py` (EEG + EDA + PPG) |
| b) EMOTIV EPOC X / FLEX | ✅ | `scripts/convert_emotiv_subject.py` (14/32-ch EEG) |
| c) smart wearables | ✅ | Non-EEG wrist loaders |
| d) CGMs | ✅ | Shanghai CGM loader |
| e) mobile health devices | ⚠️ | Covered by the generic canonical-CSV loader (`loaders.load_canonical_csv`); no device-specific parser |
| f) VR/AR environments | 🧪 | `scripts/ingest_vr_session.py` (head pose, gaze, HR overlay) — **synthetic fixtures only** |

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
abnormal-lab fraction). These proxies are **scaffolding only** (circular median-splits with
no ground truth) and must not be cited as predictive results. Which mental-health tasks are
scaffolding vs. real:

| Task | Label status |
|---|---|
| **anxiety** | Real label now available — the `deap_anxiety` benchmark task uses DEAP self-report SAM ratings (high-arousal + low-valence quadrant). The `clinical_tasks.anxiety_prediction` median-split remains scaffolding; prefer `deap_anxiety` for any claim. |
| **cognitive workload** | Real label now available — the `eegmat_workload` benchmark task uses the PhysioNet EEG mental-arithmetic cohort (resting baseline vs serial-subtraction; `scripts/fetch_data.py eegmat`, `load_eegmat_dataset`, 19-ch EEG + ECG @ 64 Hz). The `clinical_tasks.cognitive_workload` beta/alpha median-split remains scaffolding; prefer `eegmat_workload`. Evaluated result: real-decodable (best single modality ECG ≈ 0.74 AUROC), learned fusion still loses. |
| **depression** | Real label now available — the `mumtaz_depression` benchmark task uses the Mumtaz (2016) MDD-vs-healthy resting-EEG cohort (`scripts/fetch_data.py mumtaz-mdd`, `load_mumtaz_mdd_dataset`, 19-ch @ 64 Hz, subject-level diagnosis). The `clinical_tasks.depression_risk` motion/HRV median-split remains scaffolding; prefer `mumtaz_depression`. Evaluated result: highly decodable (floor/SOTA AUROC ≈ 0.92, dataset-specific), learned fusion loses hardest (−148%). |

**Every mental-health target now has a real labeled cohort** — anxiety (DEAP), arousal (DEAP), cognitive workload (eegmat), and depression (Mumtaz MDD). The `clinical_tasks.py` median-split proxies are superseded and must not be cited.

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
- `EEG-X`, `GluFormer`, `Med-BERT`, and `PH-LLM` have **no usable open weights**. Verified
  real substitutes are **named** in `dvxr.config.FOUNDATION_MODELS`, but only some have a
  loader wired here: **MOMENT (wearable), Bio_ClinicalBERT (EHR), and Geneformer (omics) load
  real weights**. **In the CACMF fusion pipeline**, LaBraM (EEG) and CGM-JEPA (CGM) are NOT wired —
  `make_primary_backend` returns `None` for both and the band-power+VQ / conformal-Ridge baselines
  run there (see the README "Runs here" table and finding C2). **The delivered product screener,
  however, does run real LaBraM** (below).
- **LaBraM — the braindecode blocker was worked around; real LaBraM now runs in the product.**
  `braindecode` installs but **cannot import** under the pinned `torch==2.12.0` (it hard-imports
  `torchaudio.functional`; no torchaudio build exists for torch 2.12). Rather than downgrade torch
  (which would destabilize MOMENT/CACMF), the EEG foundation model is loaded via a **vendored
  forward pass** over the LaBraM safetensors (`src/dvxr/encoders/labram_real.py`, strict load, no
  braindecode import). This is the path behind the **headline depression screener (AUROC 0.961)** —
  so a real EEG FM *is* delivered, on a raw-signal frozen probe. It is simply not plumbed into the
  CACMF `make_primary_backend` fusion path (which cannot consume the summary-stat table).
- Paper prose is placeholder TODO; only the result tables are auto-generated.
