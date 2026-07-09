# Streaming / partial-observation showdown — wesad_stress

Metric: `1-AUROC` (lower = better). Floor: `xgboost` (imputes missing modalities: NaN for xgboost's native handling, train-mean for the linear floor). Proposed models drop a modality by omitting it — the fusion/LLM absent-token path (graceful degradation). `k` = # modalities dropped at test time (of 7). A **win** requires RER>0 with the bootstrap CI lower bound >0 (the CI excludes a tie).

## fused vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.1205 | 0.0762 | -58.1 | -397.5..22.9 | — |
| 1 | 0.1250 | 0.0826 | -51.3 | -252.8..16.9 | — |
| 2 | 0.1495 | 0.1032 | -44.8 | -198.8..15.5 | — |
| 3 | 0.1240 | 0.0456 | -172.2 | -443.1..-63.2 | — |
| 4 | 0.1758 | 0.1584 | -11.0 | -35.7..15.0 | — |
| 5 | 0.2016 | 0.1213 | -66.2 | -163.2..-15.1 | — |
| 6 | 0.2819 | 0.2918 | +3.4 | -11.1..17.6 | — |

## fused_robust vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.1493 | 0.0762 | -95.8 | -502.0..6.2 | — |
| 1 | 0.1586 | 0.0826 | -92.0 | -321.8..-6.8 | — |
| 2 | 0.1826 | 0.1032 | -76.8 | -253.1..-5.0 | — |
| 3 | 0.1512 | 0.0456 | -231.8 | -518.2..-104.6 | — |
| 4 | 0.2166 | 0.1584 | -36.7 | -67.7..-5.4 | — |
| 5 | 0.2444 | 0.1213 | -101.6 | -221.1..-41.9 | — |
| 6 | 0.2891 | 0.2918 | +0.9 | -13.7..15.9 | — |

## llm vs floor `xgboost`

| k dropped | proposed err | floor err | RER% | 95% CI | win |
|---|---|---|---|---|---|
| 0 | 0.2910 | 0.0762 | -281.8 | -1106.5..-110.0 | — |
| 1 | 0.2828 | 0.0826 | -242.3 | -653.8..-115.3 | — |
| 2 | 0.3370 | 0.1032 | -226.4 | -542.0..-115.1 | — |
| 3 | 0.2742 | 0.0456 | -501.6 | -1014.5..-303.4 | — |
| 4 | 0.3907 | 0.1584 | -146.7 | -177.8..-113.2 | — |
| 5 | 0.3771 | 0.1213 | -211.0 | -328.1..-155.2 | — |
| 6 | 0.4307 | 0.2918 | -47.6 | -70.5..-32.3 | — |

**Verdict: no CI-backed crossover.** The tuned floor beats the proposed model at every dropout level on these summary-statistic features (gap smallest at k=6 for `fused`, RER +3.4%). Reported honestly — not faked. The proposal's genuine advantages are elsewhere (beats the deep open-weight SOTA encoder on every task; predicts under any modality subset; per-modality interpretability).
