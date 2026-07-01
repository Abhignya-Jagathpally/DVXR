# DVXR — Multimodal Health Signal Ingestion & Modeling Pipeline

A reproducible pipeline that ingests **wearable/PHR, BCI/EEG, CGM/diabetes, and EHR**
signals into one canonical event schema, builds per-modality features, trains
auditable baseline models, and reports calibrated, explainable predictions.

The code starts with classical features and lightweight encoders because they run
reliably and make data quality visible. It keeps adapter points for larger foundation
models (see `model_choice_registry.csv`):

- **EEG/BCI:** EEG-X, then LaBraM / BENDR.
- **Wearable physiology:** BIOT for heterogeneous biosignals, MOMENT for generic windows.
- **CGM:** GluFormer if weights/data access are available, else a conformalized forecasting baseline.
- **EHR:** Med-BERT/BEHRT for structured events, NYUTron/Foresight for note/concept timelines.

## CACMF — the unified multimodal fusion framework (`dvxr`)

The pipeline is now packaged as **`dvxr`** implementing **CACMF** (Cross-modal Aligned
Codebook Multimodal Fusion). `goal1_pipeline` remains importable as thin re-export
shims, so every existing script and test keeps working.

```
raw files ─▶ ingest/validate (13-col canonical schema) ─▶ per-modality features
   │
   ├─ per-modality ENCODER  f_m ─▶ z_m         (dvxr/encoders/*_adapter.py, real weights)
   ├─ VQ CODEBOOK          q_m ─▶ ê_m, code k* (dvxr/encoders/codebook.py)
   ├─ FUSION g  (early|intermediate|late|attention|cross-modal) ─▶ h  (dvxr/fusion/)
   ├─ MULTI-TASK heads + relative losses ─▶ 7 calibrated tasks   (dvxr/tasks/)
   ├─ REAL-TIME fused stream + adaptive intervention            (dvxr/realtime/)
   ├─ EXPLAIN (physio + neural saliency + attention + codebook) (dvxr/explain/)
   └─ LLM INSIGHT (explains, never predicts; offline-safe)      (dvxr/llm/)
```

**Real foundation-model weights** (verified, CPU-runnable; see `dvxr.config.FOUNDATION_MODELS`).
Where the POW's named model has no usable open weights, a verified substitute is wired
(the original is recorded), and an always-runnable baseline keeps the offline test suite green:

| Modality | Primary (real weights) | Fallback | Baseline |
|---|---|---|---|
| EEG | LaBraM `braindecode/labram-pretrained` | EEGPT | band-power + VQ encoder |
| Wearable | MOMENT `AutonLab/MOMENT-1-large` | TimesFM | neural encoder / PCA |
| CGM | CGM-JEPA `CRUISEResearchGroup/CGM-JEPA` | Chronos-Bolt | conformal Ridge |
| EHR | CEHR-BERT-style (train-local) | Bio_ClinicalBERT | tokenized-code timeline |
| Omics | Geneformer `ctheodoris/Geneformer` | — | omics features |
| Insight LLM | Anthropic Claude API | Qwen2.5-7B (local) | deterministic template |

Real weights are the primary path (`config.use_real_weights=True`); they degrade to the
baseline behind capability checks, so **the whole pipeline runs with no network and no GPU**.

### Run CACMF (one command, offline/CPU/deterministic)

```bash
python3 scripts/run_mmf_full.py            # full pipeline -> outputs/
python3 scripts/run_mmf_full.py --profile  # profile data/ -> outputs/data_schema_report.md
python3 scripts/run_mmf_full.py --realtime # fused stream  -> outputs/realtime_fused_stream.csv
python3 scripts/run_mmf_full.py --insight  # LLM insight   -> outputs/insight_example.md (offline)
python3 scripts/run_ablation.py            # Goal-3 ablation-> outputs/ablation_table.csv
make paper                                 # build paper/tables/*.tex (PDF if pdflatex present)
make all                                   # ablation + full run + paper + tests
```

Optional real-weights setup: `pip install "braindecode[hug]"` (LaBraM); set
`ANTHROPIC_API_KEY` + `DVXR_LLM_MODEL` for the live insight layer (keys read from env only,
never logged). Architecture spec: `docs/ARCHITECTURE.md`; guardrails: `docs/MASTER_BRIEF.md`.

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run on synthetic fixtures (no download)

```bash
python3 scripts/run_demo.py
```

Outputs: dataset summary, schema validation, stress classification metrics, glucose
forecasting metrics, top explanations, one streaming-style prediction, plus registries
and calibrated risk bands / prediction intervals in `outputs/`.

## Real collected BCI data → decoding + dashboard (EMOTIV + Galea)

The headline tangible result. Runs the full pipeline on the **collected** EMOTIV
EPOC X (mental commands: Neutral/Left/Right/Push/Pull) and Galea recordings in
`data/*.zip`, producing a self-contained dashboard, figures, and metrics:

```bash
venv/bin/python scripts/run_bci_pipeline.py
# -> outputs/bci/dashboard.html  + PNG figures + metrics.json
```

Decodes intended cube movement from EEG (the avatarRT / MRAE / TPHATE analog):
4-class command direction at **bal-acc 0.82 trial-grouped / 0.72 drift-controlled**
(chance 0.25), with a PHATE neural manifold, leakage-controlled CV, real-time
streaming decode, and explainable channel×band biomarkers. Full writeup:
[`BCI_PIPELINE.md`](BCI_PIPELINE.md).

## End-to-end Goal 1 run (all capabilities)

```bash
python3 scripts/run_goal1_full.py
```

Exercises every Goal 1 capability on synthetic fixtures: multimodal + multi-omics
ingestion, real device/VR-AR converters, neural (torch BIOT-style) vs PCA embeddings, the
seven clinical task heads, real-time stress+glucose streaming, explainable neural +
physiological biomarkers, and per-subject personalization. See
[`GOAL1_COMPLIANCE.md`](GOAL1_COMPLIANCE.md) for the deliverable-by-deliverable map.

The neural encoder needs torch (CPU is fine):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run on real, credential-free public data

```bash
python3 scripts/fetch_data.py all-free --subjects 20   # noneeg + mimic-demo + shanghai-cgm
python3 scripts/run_real_demo.py
```

| Stage | Real dataset | Typical result |
|---|---|---|
| Wearable stress | PhysioNet Non-EEG (20 subjects) | AUROC ~0.90, ECE ~0.14 (subject-held-out) |
| CGM / diabetes | Shanghai T1DM/T2DM (19 patients) | 30-min MAE ~11 mg/dL, 90% interval coverage ~0.92 |
| EHR ingestion | MIMIC-IV clinical demo (100 patients) | 40k events, 254 lab/demographic concepts |

All real sources download over plain HTTP without accounts. The stress labels come from
the Non-EEG `.atr` phase annotations; CGM is the open Shanghai diabetes dataset; EHR is
the open MIMIC-IV demo subset.

## DEAP EEG/peripheral arousal benchmark

`scripts/run_deap_demo.py` runs the DEAP arousal-classification path: it loads EEG and
peripheral physiology into the canonical schema, builds 30s windows (EEG band-power +
per-channel statistics), encodes them, and trains a calibrated high/low-arousal classifier
with a subject-held-out split.

```bash
# 1. Synthetic DEAP-shaped fixture — always runnable, no download
python3 scripts/run_deap_demo.py

# 2. One official preprocessed subject file
python3 scripts/run_deap_demo.py \
  --deap-pickle /path/to/data_preprocessed_python/s01.dat \
  --max-trials 40

# 3. A directory of subject files (subject-held-out evaluation)
python3 scripts/run_deap_demo.py \
  --deap-dir /path/to/data_preprocessed_python \
  --max-subjects 10 \
  --max-trials 40
```

The preprocessed `.dat` files can be fetched with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("manh123df/deap-dataset")
# subjects land in <path>/deap-dataset/data_preprocessed_python/s01.dat ... s32.dat
```

| Mode | Data | Typical result |
|---|---|---|
| Synthetic fixture | generated in-process | AUROC ~1.0 (clean fixture, validation only) |
| Single subject (`--deap-pickle`) | one DEAP subject, within-subject split | AUROC ~0.92, ECE ~0.13 |
| Directory (`--deap-dir`) | N DEAP subjects, subject-held-out | AUROC near chance — cross-subject arousal does not transfer with a linear baseline |

The single-vs-directory gap is the expected DEAP result: within-subject splits exploit
subject-specific patterns, while whole-subject hold-out demands cross-subject generalization
that a linear model on raw features does not achieve. Per-subject normalization or a stronger
encoder is the next step.

## SOTA model comparison

`scripts/compare_sota_models.py` scores the candidate foundation models per task on
evidence, Goal-1 fit, integration effort, and calibration, and records which model is
selected for the pipeline. `run_demo.py` also emits this report.

```bash
python3 scripts/compare_sota_models.py
```

Writes `outputs/sota_comparison.csv` (all candidates) and `outputs/sota_selection.csv`
(the selected models). Selected per task: EEG-X (EEG/BCI), BIOT (wearable biosignals),
GluFormer with a conformalized Ridge fallback (CGM), Med-BERT/BEHRT (EHR timelines), and
PHIA (LLM insight layer).

## Datasets requiring credentials / access

`scripts/fetch_data.py kaggle-wesad` and `kaggle-deap` use `kagglehub` and need a Kaggle
token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). Convert official files
to the canonical schema with:

```bash
python3 scripts/convert_wesad_subject.py /path/to/WESAD/S2/S2.pkl data/sample/wesad_S2_events.csv
python3 scripts/convert_deap_subject.py  /path/to/DEAP/s01.dat       data/sample/deap_s01_events.csv
```

Real Galea / EMOTIV / VR-AR exports convert into the canonical event schema before modeling
(each accepts `--demo` to run on a synthetic sample now):

```bash
python3 scripts/convert_galea_subject.py  --demo --output outputs/galea_demo.csv
python3 scripts/convert_emotiv_subject.py --demo --device epocx --output outputs/emotiv_demo.csv
python3 scripts/ingest_vr_session.py      --demo --output outputs/vr_demo.csv
python3 scripts/convert_omics_subject.py  --demo --output outputs/omics_demo.csv
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The real-data tests auto-skip when the corresponding dataset has not been downloaded.

## Layout

```
src/goal1_pipeline/   schemas, loaders, features, encoders, models, calibration, registry, sota, explain, streaming,
                      neural_encoders (torch), omics, clinical_tasks, personalization, realtime, biomarkers
scripts/              run_demo.py, run_real_demo.py, run_deap_demo.py, run_goal1_full.py, compare_sota_models.py,
                      fetch_data.py, convert_{wesad,deap,galea,emotiv,omics}_subject.py, ingest_vr_session.py
tests/                test_pipeline_smoke.py, test_sota_selection.py, test_neural_encoders.py, test_omics.py,
                      test_device_converters.py, test_clinical_tasks.py, test_personalization.py, test_realtime.py,
                      test_biomarkers.py, test_real_data.py (real, auto-skipping)
outputs/              committed result artifacts (metrics, predictions, registries, model card, SOTA report); raw event dumps gitignored
GOAL1_COMPLIANCE.md   deliverable-by-deliverable compliance map
```

## Caveats

- Synthetic-fixture metrics are pipeline validation, not scientific evidence (the fixtures
  are intentionally clean, so stress scores near 1.0 are expected).
- Real metrics use subject/patient-held-out splits; personalized claims require per-subject
  calibration that improves held-out performance.
- LLM/agent layers should *explain* model outputs, not replace deterministic signal processing.
