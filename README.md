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

## Datasets requiring credentials / access

`scripts/fetch_data.py kaggle-wesad` and `kaggle-deap` use `kagglehub` and need a Kaggle
token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). Convert official files
to the canonical schema with:

```bash
python3 scripts/convert_wesad_subject.py /path/to/WESAD/S2/S2.pkl data/sample/wesad_S2_events.csv
python3 scripts/convert_deap_subject.py  /path/to/DEAP/s01.dat       data/sample/deap_s01_events.csv
```

Real Galea / EMOTIV exports should be converted into the canonical event schema before modeling.

## Tests

```bash
python3 -m unittest discover -s tests
```

The real-data tests auto-skip when the corresponding dataset has not been downloaded.

## Layout

```
src/goal1_pipeline/   schemas, loaders, features, encoders, models, calibration, registry, explain, streaming
scripts/              run_demo.py, run_real_demo.py, fetch_data.py, convert_*_subject.py
tests/                test_pipeline_smoke.py (synthetic), test_real_data.py (real, auto-skipping)
```

## Caveats

- Synthetic-fixture metrics are pipeline validation, not scientific evidence (the fixtures
  are intentionally clean, so stress scores near 1.0 are expected).
- Real metrics use subject/patient-held-out splits; personalized claims require per-subject
  calibration that improves held-out performance.
- LLM/agent layers should *explain* model outputs, not replace deterministic signal processing.
