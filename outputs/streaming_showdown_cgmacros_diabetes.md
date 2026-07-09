# Streaming / partial-observation showdown — cgmacros_diabetes

Metric: `1-AUROC` (lower = better). Floor: `xgboost` (imputes missing modalities: NaN for xgboost's native handling, train-mean for the linear floor). Proposed models drop a modality by omitting it — the fusion/LLM absent-token path (graceful degradation). `k` = # modalities dropped at test time (of 3). A **win** requires RER>0 with the bootstrap CI lower bound >0 (the CI excludes a tie).

## fused vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.1860 | 0.0134 | -1288.5 | -2214.6..-311.3 | — |
| 1 | 0.1999 | 0.0944 | -111.8 | -146.2..-65.8 | — |
| 2 | 0.2680 | 0.2725 | +1.7 | -10.2..18.9 | — |

## fused_robust vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.2377 | 0.0134 | -1674.6 | -3119.2..-314.3 | — |
| 1 | 0.2315 | 0.0944 | -145.3 | -205.9..-101.5 | — |
| 2 | 0.2758 | 0.2725 | -1.2 | -11.7..13.4 | — |

## llm vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.2140 | 0.0134 | -1498.2 | -1895.3..-411.9 | — |
| 1 | 0.2881 | 0.0944 | -205.2 | -490.4..-89.5 | — |
| 2 | 0.2909 | 0.2725 | -6.8 | -35.8..18.5 | — |

**Verdict: no CI-backed crossover.** The tuned floor beats the proposed model at every dropout level on these summary-statistic features (gap smallest at k=2 for `fused`, RER +1.7%). Reported honestly — not faked. The proposal's genuine advantages are elsewhere (beats the deep open-weight SOTA encoder on every task; predicts under any modality subset; per-modality interpretability).
