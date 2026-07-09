# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=2, folds=3, seed=7, sota=False

**Protocol:** repeated subject/patient-held-out 5x5 CV (single-level; opponent selection and RER share folds — nested CV deferred)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); glucose = single-modality (CGM only); mortality = single-modality (EHR only). Multimodal-fusion conclusions rest on the **stress** task; no dataset co-registers EEG+CGM+EHR per subject.

```
             task  metric best_baseline  base_err  prop_err  delta_abs  RER_pct  RER_CI_low  RER_CI_high  p_wilcoxon  p_holm  cliffs_delta  n_folds  meets_>=50%
     wesad_stress 1-AUROC       rep:pca    0.0645    0.1172    -0.0527   -81.75     -153.65         4.72     0.92188     1.0        -0.222        6        False
cgmacros_diabetes 1-AUROC    single:ehr    0.1088    0.1203    -0.0115   -10.58     -164.80        38.96     0.82812     1.0        -0.167        6        False
 cgmacros_glucose     MAE       rep:raw   10.8567   11.6542    -0.7974    -7.35       -8.69        -6.09     1.00000     1.0        -1.000        6        False
```

## Verdict

- **wesad_stress** (1-AUROC, ): fused 0.1172 vs rep:pca 0.0645 -> RER -81.7% (95% CI -153.7..4.7, Wilcoxon p=0.9219, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **cgmacros_diabetes** (1-AUROC, ): fused 0.1203 vs single:ehr 0.1088 -> RER -10.6% (95% CI -164.8..39.0, Wilcoxon p=0.8281, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **cgmacros_glucose** (MAE, ): fused 11.6542 vs rep:raw 10.8567 -> RER -7.3% (95% CI -8.7..-6.1, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### wesad_stress
```
       config  1-AUROC
      rep:pca   0.0645
      rep:raw   0.0765
    rep:fused   0.1172
    cacmf_e2e   0.1332
  single:resp   0.1412
classical_gbm   0.1577
   single:eda   0.1584
   rep:neural   0.2452
single:motion   0.2607
   single:ecg   0.2657
   single:ppg   0.2835
       rep:vq   0.3021
   single:emg   0.4066
  single:temp   0.4478
     majority   0.5000
```

### cgmacros_diabetes
```
              config  1-AUROC
          single:ehr   0.1088
             rep:raw   0.1102
          single:cgm   0.1136
           cacmf_e2e   0.1192
           rep:fused   0.1203
          rep:neural   0.1691
              rep:vq   0.1786
             rep:pca   0.2624
            majority   0.5000
       classical_gbm   0.5000
single:wearable_phys   0.5479
```

### cgmacros_glucose
```
       config     MAE
      rep:raw 10.8567
   single:cgm 10.8567
      rep:pca 10.8596
classical_gbm 11.0915
  persistence 11.3638
    rep:fused 11.6542
       rep:vq 11.8823
   rep:neural 12.0441
    cacmf_e2e 13.0122
```

## True modality ablation (retrain without the modality)


### wesad_stress  (contribution = error increase when dropped)
```
dropped_modality  err_without (1-AUROC)  err_with_all (1-AUROC)  contribution  ci_low  ci_high
            resp                 0.1943                  0.1172        0.0771  0.0224   0.1407
            temp                 0.1550                  0.1172        0.0378 -0.0394   0.1196
          motion                 0.1504                  0.1172        0.0332 -0.0477   0.1141
             ppg                 0.1419                  0.1172        0.0247 -0.0386   0.0929
             ecg                 0.1055                  0.1172       -0.0117 -0.0850   0.0598
             eda                 0.1028                  0.1172       -0.0145 -0.0938   0.0649
             emg                 0.0952                  0.1172       -0.0220 -0.1048   0.0586
```

### cgmacros_diabetes  (contribution = error increase when dropped)
```
dropped_modality  err_without (1-AUROC)  err_with_all (1-AUROC)  contribution  ci_low  ci_high
             cgm                 0.1791                  0.1203        0.0588  0.0067   0.1103
   wearable_phys                 0.1408                  0.1203        0.0205 -0.0274   0.0670
             ehr                 0.1336                  0.1203        0.0133 -0.0309   0.0594
```
