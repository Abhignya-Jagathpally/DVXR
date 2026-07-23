# Glucose forecasting: the sentinel ↔ dvxr bridge

`dvxr` reuses the standalone **NeuroGlycemic Sentinel** package
(`neuroglycemic-sentinel/`) for causal, patient-disjoint CGM forecasting instead of
re-implementing it. The two are kept isolated and connected only through a thin
out-of-process bridge.

## Why out-of-process

`neuroglycemic-sentinel` uses a `src`-as-package layout (`from src.neuroglycemic import …`)
and pins `numpy>=1.26,<2`. Importing it into the `dvxr` interpreter would collide with
`dvxr`'s own `src/` and entangle dependency pins. The bridge sidesteps both by launching
the sentinel CLI as a subprocess with the sentinel repo as its working directory. The
CLI already writes every result as an auditable file into the external runtime workspace,
so the bridge only has to build argv, run it, and read the files back.

**Honesty boundary.** The bridge launches a process and reads artifacts — it never
computes, imputes, or edits a forecast number. Metrics come from files the sentinel CLI
wrote under patient-disjoint splits and superiority gates. All glucose artifacts stay
**out-of-repo** in `neuroglycemic-runtime/`; none are promoted into `dvxr`'s committed
`outputs/` scoreboards, so the torch-free `make audit` stays green.

## Environment

On this checkout the active `venv` (Python 3.12, `numpy 1.26.4`, `torch 2.12`, plus
`matplotlib`/`openpyxl`) satisfies **both** packages, so the sentinel CLI runs in the same
interpreter — no second venv is required and `python_executable` defaults to
`sys.executable`. If you ever need strict isolation (e.g. a numpy-2 dvxr environment),
create a dedicated sentinel venv and pass its interpreter:

```bash
python3.11 -m venv .venv-glucose && . .venv-glucose/bin/activate
pip install -e ./neuroglycemic-sentinel[excel,plots]
# then: GlucoseForecastingBridge(python_executable=".venv-glucose/bin/python")
```

The sentinel `pyproject.toml` references a `README.md` that now exists — `pip install -e`
succeeds.

## Directory contract

```
neuroglycemic-sentinel/        # software (this bridge shells into it)
neuroglycemic-runtime/         # external workspace (disjoint sibling; the CLI enforces this)
  aligned/  <cohort>_windows.csv.gz
  canonical/<cohort>_ingestion_audit.csv
  models/   <run>.pt (+ <run>.pt.release.json)
  runs/<run>/  test_metrics.json, training_acceptance.json, test_predictions.csv,
               missing_modality_ablation.csv, evaluation_reproducibility.json,
               figures/{training_loss,held_out_forecasts,fusion_weights}.png
```

## Usage from Python

```python
from dvxr.integrations import GlucoseForecastingBridge

bridge = GlucoseForecastingBridge()          # defaults to the sibling layout

art = bridge.train(
    data="neuroglycemic-runtime/aligned/big_ideas_wearable_cgm_windows.csv.gz",
    config="neuroglycemic-sentinel/config/diatrend_glucose.json",
    run_name="diatrend-style-bigideas",
)
art = bridge.evaluate(
    data="neuroglycemic-runtime/aligned/big_ideas_wearable_cgm_windows.csv.gz",
    config="neuroglycemic-sentinel/config/diatrend_glucose.json",
    run_name="diatrend-style-bigideas",
)
print(art.metrics)                # parsed test_metrics.json
print(art.acceptance)             # superiority-gate booleans (report verbatim)
print(list(art.figures))          # figure PNG paths under runs/<run>/figures/
```

The bridge caps BLAS/OMP threads (default 2, override with `DVXR_GLUCOSE_THREAD_CAP`)
because this is a shared multi-user host.

## DiaTrend note

The DiaTrend dataset (Synapse `syn38187184`) is data-use-agreement gated; no workbooks
are bundled. `prepare_diatrend(source_dir, source_timezone)` is wired and ready for real
workbooks. For the reproduced DiaTrend-paper figure suite we instead train on the real
CGM cohorts already on disk (BIG-IDEAS), using the DiaTrend training config and labelling
every figure with the actual cohort — no fabricated DiaTrend numbers.
