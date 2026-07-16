# CGM-only glucose forecast + calibrated interval — CGMacros (single cohort, honest)

A **CGM-only** continuous forecast of the glucose *level* at a 30/60-minute horizon, wrapped in a
**split-conformal** prediction interval (`src/dvxr/eval/glucose_forecast.py`). This complements the
binary excursion classifier (`glucose_ablation.py`): it answers "what will glucose *be*, and how sure
are we?" rather than "will a threshold be crossed?".

**Methodology.** Causal CGM-history features (`cgm_history_features`) only — no wearable, no EEG, no
fused signal. The target is the observed glucose at the sample closest to `t+h` (within tolerance;
anchors whose future is unobserved near `t+h` are censored, never imputed). Subject **K-fold with
disjoint test folds** (each participant scored once); a `GradientBoostingRegressor` is fit on the train
subjects, the conformal radius `q` is the finite-sample `(n+1)` quantile of absolute residuals on a
**held-out calibration subset of training subjects** (never the test rows), and the interval is
`ŷ ± q`. Reproduce with `run_glucose_forecast(load_cgmacros())`.

| Horizon | RMSE (mg/dL) | MAE (mg/dL) | Bias | Target coverage | Empirical coverage | Mean interval width (mg/dL) |
|---|---|---|---|---|---|---|
| 30 min | 14.10 | 10.27 | +0.56 | 0.90 | **0.877** | 42.3 |
| 60 min | 20.60 | 15.47 | +0.61 | 0.90 | **0.855** | 62.6 |

(34 subjects, ~2,458 held-out examples per horizon.)

## What is honest here — and what the coverage gap means

The point-error is a legitimate CGM-only baseline (30-min RMSE ≈ 14 mg/dL is in line with the published
CGM short-horizon forecasting range). The interval is **conformal**, so on exchangeable held-out data
its coverage is provably `≥ 1-alpha`.

But the empirical coverage lands **below** the 0.90 target (0.877 at 30 min, 0.855 at 60 min), and that
is reported rather than tuned away. The reason is structural and worth stating plainly: our test fold is
a set of **held-out subjects**, so the calibration residuals (from *other* people) and the test residuals
(from *new* people) are **not exchangeable** — between-subject glucose variability breaks the marginal
guarantee. This is the correct, conservative thing to surface: a subject-split conformal interval is
*approximately* calibrated, not exactly. Closing the gap honestly would need per-subject/online conformal
recalibration on each participant's own early data (future work), not a wider global fudge factor.

## Scope

CGM-only, CGMacros-only, threshold config `pilot-v1`. This says nothing about EEG and is **not** the
fused product claim — every EEG/fused arm remains `cannot_evaluate` (no cohort co-registers EEG with
CGM). The fused NeuroGlycemic Sentinel headline stays research-stage and abstains until synchronized
same-subject EEG+wearable+CGM pilot data exists.
