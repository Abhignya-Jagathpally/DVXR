# DVXR presentation — slide outline & asset map

A multimodal **clinical risk prediction** framework (mental-health analytics + physiology +
diabetes), built and evaluated honestly. Each slide lists the asset in `presentation/`.

| # | Slide | Asset |
|---|---|---|
| 1 | **Title** — LLM-based multimodal clinical risk prediction & mental-health analytics (Galea/EMOTIV, wearables, EHR, diabetes) | — |
| 2 | **The framework** — one system, many signals (EEG + PHR + PPG → glucose + mind-state) | `diagrams/framework_overview.png` |
| 3 | **Goal 1 — ingestion** — the pipeline ingests all five modalities | `figures/fig_ingestion_matrix.png` |
| 4 | **Model flow** — device streams → experts → fusion → forecast → explanation | `diagrams/model_flow_diagram.html` (interactive) |
| 5 | **Model in detail** — availability-aware mixture-of-experts | `diagrams/model_architecture.png` |
| 6 | **Goal 2 result — glucose** — beats persistence at every horizon (RMSE ~13 @30 min) | `figures/fig_glucose_horizons.png` + `figures/glucose_forecast_scatter.png` |
| 7 | **Goal 3 — why this model?** — the causal representation is the win, not depth | `figures/fig_model_ladder.png` |
| 8 | **Which device matters** — per-device leave-one-out | `figures/fig_per_device.png` |
| 9 | **Mental-health heads vs SOTA** — depression 0.961, stress 0.955 | `figures/fig_heads_sota.png` |
| 10 | **DiaTrend-style cohort figures** — traces, time-in-range, summary | `figures/diatrend_*.png` |
| 11 | **Real-time demo** — BCI digital-twin (web + Unity) | web scene / Unity `DVXR_RT_Demo` |
| 12 | **Trust** — causal · calibrated · abstains · LLM explains (guarded) · never predicts | `../docs/{CAUSAL,EXPLAINABILITY}.md` |
| 13 | **Clinical positioning** — clinical purpose, pre-deployment maturity, path to deployment | `../docs/GOAL_ACHIEVEMENT.md` |

## Headline numbers (all real, subject/patient-held-out)
- Glucose forecast RMSE **12.8 / 21.9 / 26.6 / 29.1** mg/dL @30/60/90/120 min — superiority gate passed.
- Depression AUROC **0.961**, WESAD stress **0.955** — SOTA-competitive.
- Every output: causal, calibrated, abstaining, `validated_for_clinical_use = false` (pre-deployment).

## Rebuild
```
python presentation/build_presentation_assets.py          # result figures
python scripts/render_framework_overview.py --out presentation/diagrams/framework_overview.png
python scripts/render_model_architecture.py --out presentation/diagrams/model_architecture.png
```
