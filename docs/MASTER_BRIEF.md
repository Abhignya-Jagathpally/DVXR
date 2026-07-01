# CACMF Master Brief

> Paste-as-background for every stage. This is the ground truth + guardrails for
> refactoring the existing DVXR research repo into the CACMF framework. We are
> **refactoring a working repo, not greenfielding** — respect what is there.

## 1.1 What the repo already does (verified from the codebase)

A reproducible pipeline (now under `src/dvxr/`, formerly `src/goal1_pipeline/`) that
ingests wearable/PHR, BCI/EEG, CGM/diabetes, and EHR signals into one canonical
event schema, builds per-modality features, trains calibrated baseline models, and
reports explainable predictions. Real public symbols:

- **schemas.py** — `REQUIRED_EVENT_COLUMNS` (13-col canonical table), `validate_events`, `summarize_events`, `DataSummary`, `ensure_columns`.
- **loaders.py** — `load_noneeg_dataset`, `load_mimic_demo_ehr`, `load_shanghai_cgm_dataset`, `load_deap_preprocessed_pickle`, `load_wesad_subject_pickle`, `load_canonical_csv`.
- **features.py** — `build_signal_windows`, `build_stress_windows`, `build_deap_arousal_windows`, `build_glucose_forecast_table`, `latest_stress_feature_row`, `feature_columns`.
- **encoders.py** — `FeatureEncoder` (PCA baseline), `EncoderRecommendation`, `recommendation_table`.
- **neural_encoders.py** — `NeuralBiosignalEncoder` (torch BIOT-style transformer, masked-feature reconstruction, `gradient_saliency`, `save`, `from_pretrained`). No VQ codebook yet.
- **models.py** — `TrainedModel`, `train_binary_classifier`, `train_stress_classifier`, `train_glucose_forecaster`.
- **clinical_tasks.py** — `CLINICAL_TASKS` (seven trainable heads), `train_clinical_task`, `derive_task_labels`, `clinical_tasks_table`. Proxies where no ground truth exists.
- **calibration.py** — Platt calibration, `expected_calibration_error`, `conformal_radius`, `interval_coverage`, `risk_band`, `add_risk_bands`.
- **registry.py / sota.py** — model & dataset choice tables; `write_sota_report`.
- **realtime.py** — `RealtimeMonitor`, `stream_predictions`. **streaming.py** — `predict_latest_stress`.
- **biomarkers.py** — `physiological_biomarkers`, `neural_biomarker_saliency`.
- **personalization.py** — `per_subject_normalize`, `PersonalizedCalibrator`.
- **omics.py** — multi-omics ingestion. **reporting.py** — `write_model_card`.
- **scripts/** — `run_demo.py`, `run_goal1_full.py`, `run_real_demo.py`, `run_deap_demo.py`, `compare_sota_models.py`, `fetch_data.py`, device converters `convert_{galea,emotiv,wesad,deap,omics}_subject.py`, `ingest_vr_session.py`.
- **tests/** — 10 modules (143 tests); real-data tests auto-skip when data absent.
- **GOAL1_COMPLIANCE.md** — deliverable-by-deliverable map. Keep it updated.

## 1.2 The canonical event schema (do not break it — extend it)

Every ingested signal becomes rows of this exact 13-column table (`schemas.REQUIRED_EVENT_COLUMNS`):

```
subject_id, session_id, timestamp_utc, source_system, device, modality,
channel, value, unit, sampling_rate_hz, quality_flag, label_name, label_value
```

`dvxr.ingest.profile_data_dir()` recursively lists/profiles `data/` + `data/sample/`,
proposes a mapping into the 13 columns, writes `outputs/data_schema_report.md`, and
**fails loudly** (strict mode) on any file it cannot map — never silently coerces.

## 1.3 Selected foundation models — wire as adapters, keep fallbacks

| Modality | POW primary | Real-weight status (verified) | Always-runnable fallback |
|---|---|---|---|
| EEG/BCI (Galea, EMOTIV) | EEG-X → LaBraM/BENDR | **EEG-X has no open weights → use LaBraM** `braindecode/labram-pretrained` | windowed band-power + VQ neural encoder |
| Wearable physiology | BIOT / MOMENT | **open** → `AutonLab/MOMENT-1-large` | `NeuralBiosignalEncoder` / PCA |
| CGM / glucose | GluFormer | **GluFormer no open weights → MOMENT / Chronos** | conformalized Ridge forecaster |
| EHR | Med-BERT / BEHRT | **no open weights → Bio_ClinicalBERT** `emilyalsentzer/Bio_ClinicalBERT` | tokenized-code timeline features |
| Omics | (none named) | `ctheodoris/Geneformer` | `build_omics_features` → linear proj |
| LLM insight | PHIA / PH-LLM / Health-LLM | **all closed → Anthropic Claude API** (or local BioMistral-7B) | deterministic template + offline summarizer |

The full canonical model registry lives in `dvxr.config.FOUNDATION_MODELS`.

## 1.4 Non-negotiable guardrails (preserve the repo's philosophy)

1. **Always-runnable first.** Every stage runs end-to-end on synthetic fixtures with
   no network and no GPU (`run_goal1_full.py` and `run_mmf_full.py` both pass).
   Foundation-model weights and LLM APIs are optional adapters behind capability
   checks; when unavailable, degrade to the bundled baseline and say so.
2. **LLMs explain, they do not replace.** Deterministic signal processing and
   calibrated models produce the numbers; the LLM layer narrates/retrieves/recommends.
   Never let a prompt-based prediction overwrite a calibrated model output.
3. **Honest metrics.** Synthetic-fixture scores validate the pipeline, not science.
   Real metrics use subject/patient-held-out splits. Keep the Caveats discipline.
4. **No fabricated labels.** Documented proxies stay clearly labeled as proxies.
5. **Backward compatibility.** `goal1_pipeline` stays importable via re-export shims.
   Existing scripts and tests must keep passing.
6. **Determinism.** Seed everything (`torch.manual_seed`, `np.random.seed`); embeddings
   and metrics reproducible run-to-run. Config `seed=7`.
7. **Small, reviewable commits.** One stage = one focused diff + its tests.

## 1.5 Definition of done (whole project)

A `dvxr` package implementing CACMF with: VQ-codebook tokenization; all five fusion
strategies (early, intermediate, late, attention, cross-modal transformer) plus the
three aggregation baselines (weighted late, ensemble averaging, confidence-weighted);
a unified multi-task objective with tunable relative-loss weights; real-time fused
streaming with adaptive intervention; an explainability bundle (physiological + neural
+ attention + codebook usage); an optional provider-agnostic LLM insight layer with
offline fallback; a Goal-3 ablation harness (fused vs single-modality); and a Goal-4
IEEE paper scaffold whose tables auto-fill from `outputs/`. All green on
`python3 -m unittest discover -s tests`, plus a one-command `scripts/run_mmf_full.py`.

## Real-weights directive (this engagement)

The user directed **"MAKE SURE TO USE REAL WEIGHTS."** Reconciliation with guardrail #1:
real checkpoints are the **default/primary** path (`config.use_real_weights=True`,
downloaded + loaded + run), while baseline fallbacks remain **only** so the offline test
suite stays green. Where the POW's named model has no usable open weights, a verified
real substitute is used and the original is recorded in `config.FOUNDATION_MODELS`.
