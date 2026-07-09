# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=2, folds=3, seed=7, sota=True

**Protocol:** repeated subject/patient-held-out 5x5 CV (single-level; opponent selection and RER share folds — nested CV deferred)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); glucose = single-modality (CGM only); mortality = single-modality (EHR only). Multimodal-fusion conclusions rest on the **stress** task; no dataset co-registers EEG+CGM+EHR per subject.

```
             task  metric best_baseline  base_err  prop_err  delta_abs  RER_pct  RER_CI_low  RER_CI_high  p_wilcoxon  p_holm  cliffs_delta  n_folds  meets_>=50%
cgmacros_diabetes 1-AUROC       xgboost    0.0100    0.1203    -0.1103 -1103.03    -1666.67      -350.51     1.00000     1.0        -0.889        6        False
 cgmacros_glucose     MAE       rep:raw   10.8567   11.6542    -0.7974    -7.35       -8.69        -6.09     1.00000     1.0        -1.000        6        False
     wesad_stress 1-AUROC       rep:raw    0.0859    0.1560    -0.0700   -81.48     -282.47       -17.51     0.96875     1.0        -0.556        6        False
```

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

```
             task  metric   floor floor_err   floor_ECE sota sota_err    sota_ECE  proposed proposed_err proposed_ECE
cgmacros_diabetes 1-AUROC xgboost    0.0100 0.068/0.041 sota   0.1892 0.179/0.125 cacmf_e2e       0.1192  0.256/0.176
 cgmacros_glucose     MAE rep:raw   10.8567           —    —      nan           — rep:fused      11.6542            —
     wesad_stress 1-AUROC rep:raw    0.0859 0.170/0.104 sota   0.1908 0.171/0.091 cacmf_e2e       0.1415  0.253/0.155
```

- **cgmacros_diabetes**: vs floor (xgboost 0.0100): proposed cacmf_e2e 0.1192 -> does NOT beat; vs SOTA (sota 0.1892): BEATS.
- **cgmacros_glucose**: vs floor (rep:raw 10.8567): proposed rep:fused 11.6542 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **wesad_stress**: vs floor (rep:raw 0.0859): proposed cacmf_e2e 0.1415 -> does NOT beat; vs SOTA (sota 0.1908): BEATS.

## Verdict

- **cgmacros_diabetes** (1-AUROC, ): fused 0.1203 vs xgboost 0.0100 -> RER -1103.0% (95% CI -1666.7..-350.5, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **cgmacros_glucose** (MAE, ): fused 11.6542 vs rep:raw 10.8567 -> RER -7.3% (95% CI -8.7..-6.1, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **wesad_stress** (1-AUROC, ): fused 0.1560 vs rep:raw 0.0859 -> RER -81.5% (95% CI -282.5..-17.5, Wilcoxon p=0.9688, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- **cgmacros_glucose**: failures by config = {'sota': 6}; unstable (NaN >20% folds) = ['sota']

## Per-configuration CV error (lower is better)


### cgmacros_diabetes  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
              config  1-AUROC
             xgboost   0.0100
          single:ehr   0.1088
             rep:raw   0.1102
          single:cgm   0.1136
           cacmf_e2e   0.1192
           rep:fused   0.1203
          rep:neural   0.1691
              rep:vq   0.1786
                sota   0.1892
             rep:pca   0.2624
            majority   0.5000
       classical_gbm   0.5000
single:wearable_phys   0.5479
```

### cgmacros_glucose  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
       config     MAE
      rep:raw 10.8567
   single:cgm 10.8567
ridge_history 10.8567
      rep:pca 10.8596
      xgboost 11.0629
classical_gbm 11.0915
  persistence 11.3638
    rep:fused 11.6542
       rep:vq 11.8823
   rep:neural 12.0441
    cacmf_e2e 13.0122
         sota     NaN
```

### wesad_stress  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
       config  1-AUROC
      rep:raw   0.0859
      xgboost   0.1054
classical_gbm   0.1194
      rep:pca   0.1197
  single:resp   0.1412
    cacmf_e2e   0.1415
    rep:fused   0.1560
         sota   0.1908
   rep:neural   0.2068
  single:temp   0.2547
   single:eda   0.2589
   single:ecg   0.2657
single:motion   0.2820
   single:ppg   0.2835
       rep:vq   0.3252
   single:emg   0.4066
     majority   0.5000
```
