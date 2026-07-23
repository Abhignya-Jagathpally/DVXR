# NeuroGlycemic Sentinel

Causal, multimodal neural glucose-forecasting research software. The package ingests
continuous glucose monitor (CGM), wearable-physiology, and clinical (EHR) records,
builds **causal** forecast windows (no future leakage), trains a neural forecaster with
split-conformal calibration, and evaluates it under **patient-disjoint** splits with
paired bootstrap superiority tests.

> **Research software, not a medical device.** No output here is validated for clinical
> use. Every trained checkpoint ships a `*.release.json` governance record whose
> `status` is `research_only`. The optional LLM (`HealthAgent`) *explains* numeric
> results after inference — it never produces a prediction.

## Code vs. data separation

The Git repository contains **software only**. Raw data, aligned windows, checkpoints,
run manifests, and figures live in an **external runtime workspace** that must be a
disjoint sibling of this repository (see `src/neuroglycemic/workspace.py`). The CLI
rejects any runtime path placed inside the repo. The conventional location is
`../neuroglycemic-runtime`.

```
neuroglycemic-sentinel/        # this repository — software
neuroglycemic-runtime/         # external workspace — raw/, aligned/, models/, runs/, figures/
```

## Install

Python `>=3.11,<3.14`, `numpy>=1.26,<2`. From the repository root:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[excel,plots]      # excel = openpyxl (DiaTrend workbooks); plots = matplotlib
# optional extras: acquisition (pylsl/pyxdf), llm (langchain), dev (pytest)
```

The neural forecaster requires `torch>=2.2,<3`.

## Command-line usage

All commands are explicit subcommands of `main.py` — no implicit study runs.

```bash
# 1. Build causal aligned windows from an external source dataset (out-of-repo)
python main.py prepare-diatrend  --workspace ../neuroglycemic-runtime \
    --source-dir ../neuroglycemic-runtime/raw/diatrend --source-timezone America/New_York
python main.py prepare-big-ideas --workspace ../neuroglycemic-runtime \
    --source-dir ../neuroglycemic-runtime/raw/big-ideas-glycemic-wearable --source-timezone America/Los_Angeles

# 2. Train (checkpoint -> runtime/models/<run>.pt ; artifacts -> runtime/runs/<run>/)
python main.py train-neural --data ../neuroglycemic-runtime/aligned/<cohort>_windows.csv.gz \
    --config config/diatrend_glucose.json --workspace ../neuroglycemic-runtime --run-name my-run

# 3. Deterministic re-evaluation from the checkpoint-recorded split + scalers
python main.py evaluate-neural --data ../neuroglycemic-runtime/aligned/<cohort>_windows.csv.gz \
    --config config/diatrend_glucose.json --workspace ../neuroglycemic-runtime --run-name my-run
```

Each run directory (`runtime/runs/<run>/`) contains `test_metrics.json`,
`training_acceptance.json` (superiority-gate booleans), `test_predictions.csv`,
`patient_split.csv`, `missing_modality_ablation.csv`, `model_card.json`, and
`figures/{training_loss,held_out_forecasts,fusion_weights}.png`. `evaluate-neural`
additionally writes `reloaded_test_metrics.json` and `evaluation_reproducibility.json`
(predictions reproduced to numerical tolerance).

## Interoperability with DVXR

The main `dvxr` project drives this package out-of-process through
`dvxr.integrations.glucose_forecasting.GlucoseForecastingBridge`, which builds the exact
`main.py` argv, runs it via `subprocess`, and reads the auditable artifacts back. See
`../docs/GLUCOSE_FORECASTING.md`. This keeps the two packages isolated (no shared
interpreter) so the `src`-as-package layout here never collides with `dvxr`'s own `src/`.

## Tests

```bash
python -m pytest tests -q
```
