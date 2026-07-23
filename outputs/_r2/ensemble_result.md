# Deep-ensemble result vs the model ladder (@30 min, same split)

|   horizon_minutes |   ensemble_rmse |   best_member_rmse |   mean_member_rmse |
|------------------:|----------------:|-------------------:|-------------------:|
|                30 |          13.061 |             12.987 |             13.195 |
|                60 |         nan     |            nan     |            nan     |
|                90 |         nan     |            nan     |            nan     |
|               120 |         nan     |            nan     |            nan     |

- ensemble @30 min: **13.06** mg/dL
- gradient boosting @30 min: 12.48
- single deep net @30 min: 12.99
- persistence @30 min: 17.40

**Verdict: the ensemble does NOT beat gradient boosting** (13.06 vs 12.48) — reported honestly. Gradient boosting remains the best point forecaster; the deep ensemble's value is calibrated uncertainty + abstention + fusion.
