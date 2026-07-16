# Making the CGM product executable (PR34)

Before PR34 the Generate API abstained on every request: it defaulted to `AbstainingPredictionService`
unless a caller hand-injected a fitted predictor, and the HTTP API injected nothing. PR34 makes the
product genuinely executable — the API **loads a committed CGM artifact from the model registry and
returns a real prediction**, and it **never trains during a request**.

## Deployment path (fit offline → register → load-and-serve)

```
python scripts/build_cgm_artifact.py            # fits on CGMacros, saves artifacts, registers ACTIVE
```

The builder fits two **single-modality, single-cohort** models on CGMacros, saves each as a portable
artifact (`model.joblib` + sha256-verified `manifest.json`), and registers a pointer + hash in a local
model registry as the ACTIVE model for its report type:

| report_type | active artifact | serves |
|---|---|---|
| `cgm_glucose_risk` / `glucose_risk` | `CgmOnlyExcursionService` (`cgm-only/pilot-v1`) | per-horizon excursion probability **+** the continuous forecast |
| `cgm_glucose_forecast` | `CgmOnlyGlucoseForecastService` (`cgm-forecast/pilot-v1`) | calibrated continuous glucose forecast |
| `stress_glucose_risk` (fused) | **none — cannot exist** | **abstains** (no synchronized EEG+CGM data) |

At request time `dvxr.prediction.registry.resolve_predictor` loads the active artifact (verifying its
sha256), or returns a **fail-closed abstainer** when nothing is registered, the artifact is missing, or
the hash mismatches. The orchestrator resolves the cutoff to a concrete instant first (P0-2), so the
snapshot, consent evaluation, and persisted prediction all anchor to one reproducible time.

**Git-clean / fail-closed:** the real model binaries live under `artifacts/` (gitignored). A clean
checkout has no artifact, so the API abstains until `build_cgm_artifact.py` runs — honest and
fail-closed, never a stale committed blob.

## The built CGMacros forecaster vs. naive baselines (honest)

The forecaster is scored, at fit time, on a **subject-held-out calibration fold** against the two naive
baselines a glucose forecaster must beat — persistence (`ŷ = last`) and linear extrapolation
(`ŷ = last + slope·h`). Real CGMacros (34 subjects):

| Horizon | Learned RMSE | Persistence RMSE | Linear RMSE | Beats persistence? |
|---|---|---|---|---|
| 30 min | 11.0 mg/dL | 12.3 | 16.1 | yes (modest) |
| 60 min | 19.0 mg/dL | 19.2 | 28.6 | yes (**barely**) |

This is the honest picture: **persistence is a strong baseline**, and the learned model adds only a
modest edge at 30 min and a razor-thin one at 60 min. That margin — not a headline "beats naive by a
mile" claim — is what the numbers support. (The primary, fully subject-K-fold forecast evaluation with
conformal coverage is in `outputs/glucose_forecast_cgmacros.md`; the table here is the fit-time
calibration-fold sanity check the artifact records in its manifest.)

## What stays gated (guardrail, unchanged)

There is **no fused artifact** and there never can be until synchronized same-subject EEG+wearable+CGM
data exists — so `stress_glucose_risk` abstains by construction, pinned by the CI honesty audit
(`test_fused_report_type_has_no_committed_artifact_and_abstains`). Every EEG/fused arm remains
`cannot_evaluate`. No fabricated fused number, ever.
