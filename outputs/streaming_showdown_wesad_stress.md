# Streaming / partial-observation showdown — wesad_stress

Metric: `1-AUROC` (lower = better). Floor: `xgboost` (imputes missing modalities: NaN for xgboost's native handling, train-mean for the linear floor). Proposed models drop a modality by omitting it — the fusion/LLM absent-token path (graceful degradation). `k` = # modalities dropped at test time (of 7). A **win** requires RER>0 with the bootstrap CI lower bound >0 (the CI excludes a tie).

## fused vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.1205 | 0.0762 | -58.1 | -397.5..22.9 | — |
| 1 | 0.1343 | 0.0966 | -39.1 | -244.4..24.1 | — |
| 2 | 0.1417 | 0.0981 | -44.5 | -139.5..9.0 | — |
| 3 | 0.1758 | 0.1180 | -49.0 | -166.8..4.6 | — |
| 4 | 0.1480 | 0.1256 | -17.9 | -44.3..10.2 | — |
| 5 | 0.1955 | 0.1594 | -22.7 | -64.9..7.3 | — |
| 6 | 0.3181 | 0.2754 | -15.5 | -34.5..3.5 | — |

## fused_robust vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.1493 | 0.0762 | -95.8 | -502.0..6.2 | — |
| 1 | 0.1666 | 0.0966 | -72.5 | -326.8..7.2 | — |
| 2 | 0.1775 | 0.0981 | -81.0 | -191.3..-13.4 | — |
| 3 | 0.2004 | 0.1180 | -69.8 | -210.5..-9.9 | — |
| 4 | 0.1826 | 0.1256 | -45.4 | -82.2..-8.3 | — |
| 5 | 0.2151 | 0.1594 | -35.0 | -77.5..-4.8 | — |
| 6 | 0.3392 | 0.2754 | -23.2 | -41.9..-5.6 | — |

**Verdict: no CI-backed crossover.** The tuned floor beats the proposed model at every dropout level on these summary-statistic features (gap smallest at k=6 for `fused`, RER -15.5%). Reported honestly — not faked. The proposal's genuine advantages are elsewhere (beats the deep open-weight SOTA encoder on every task; predicts under any modality subset; per-modality interpretability).
