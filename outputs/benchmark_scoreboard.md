# CACMF relativity scoreboard — real labels, held-out subjects

**Protocol:** repeats=5, folds=5, seed=7, sota=True

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.

```
     task  metric best_baseline  base_err  prop_err  delta_abs  RER_pct  RER_CI_low  RER_CI_high  p_wilcoxon  p_holm  cliffs_delta  n_folds  meets_>=50%
   stress 1-AUROC       rep:pca    0.1079    0.1294    -0.0215   -19.87      -28.58       -13.50     0.99999     1.0        -0.254       25        False
  glucose     MAE       rep:raw   10.6631   13.0939    -2.4308   -22.80      -25.66       -20.15     1.00000     1.0        -0.712       25        False
mortality 1-AUROC       rep:pca    0.1784    0.3599    -0.1815  -101.71     -156.98       -61.07     0.99990     1.0        -0.539       23        False
```

## Verdict

- **stress** (1-AUROC): fused 0.1294 vs rep:pca 0.1079 -> RER -19.9% (95% CI -28.6..-13.5, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **glucose** (MAE): fused 13.0939 vs rep:raw 10.6631 -> RER -22.8% (95% CI -25.7..-20.1, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **mortality** (1-AUROC): fused 0.3599 vs rep:pca 0.1784 -> RER -101.7% (95% CI -157.0..-61.1, Wilcoxon p=0.9999, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Per-configuration CV error (lower is better)


### stress  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
       config  1-AUROC
      rep:pca   0.1079
      rep:raw   0.1113
classical_gbm   0.1204
    rep:fused   0.1294
    cacmf_e2e   0.1666
single:motion   0.1670
   single:ppg   0.2505
         sota   0.2743
       rep:vq   0.3166
   rep:neural   0.3245
   single:eda   0.3416
  single:temp   0.3599
     majority   0.5000
```

### glucose  (SOTA backend: cgm_summary)
```
       config     MAE
      rep:raw 10.6631
   single:cgm 10.6631
      rep:pca 10.6636
classical_gbm 11.6248
         sota 11.7073
  persistence 12.8761
    rep:fused 13.0939
   rep:neural 14.8668
       rep:vq 15.3913
    cacmf_e2e 20.1736
```

### mortality  (SOTA backend: hf:emilyalsentzer/Bio_ClinicalBERT)
```
       config  1-AUROC
      rep:pca   0.1784
      rep:raw   0.2254
   single:ehr   0.2254
       rep:vq   0.2282
classical_gbm   0.2457
   rep:neural   0.3092
         sota   0.3126
    cacmf_e2e   0.3175
    rep:fused   0.3599
     majority   0.5000
```

## True modality ablation (retrain without the modality)


### stress  (contribution = error increase when dropped)
```
dropped_modality  err_without (1-AUROC)  err_with_all (1-AUROC)  contribution  ci_low  ci_high
          motion                 0.2120                  0.1264        0.0855  0.0668   0.1059
             ppg                 0.1468                  0.1264        0.0204  0.0119   0.0296
            temp                 0.1420                  0.1264        0.0155  0.0085   0.0221
             eda                 0.1327                  0.1264        0.0063 -0.0032   0.0159
```
