# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=3, folds=4, seed=7, sota=True

**Protocol:** repeated subject/patient-held-out 5x5 CV (single-level; opponent selection and RER share folds — nested CV deferred)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); glucose = single-modality (CGM only); mortality = single-modality (EHR only). Multimodal-fusion conclusions rest on the **stress** task; no dataset co-registers EEG+CGM+EHR per subject.

```
             task  metric best_baseline  base_err  prop_err  delta_abs  RER_pct  RER_CI_low  RER_CI_high  p_wilcoxon  p_holm  cliffs_delta  n_folds  meets_>=50%
     wesad_stress 1-AUROC       rep:raw    0.0571    0.1568    -0.0997  -174.63     -331.56       -97.44     1.00000     1.0        -0.875       12        False
     deap_arousal 1-AUROC classical_gbm    0.4503    0.4561    -0.0058    -1.29       -9.63         8.03     0.82520     1.0        -0.139       12        False
cgmacros_diabetes 1-AUROC       xgboost    0.0175    0.1291    -0.1116  -636.63    -1552.15      -218.86     0.99805     1.0        -0.744       11        False
 cgmacros_glucose     MAE       rep:raw   10.8829   11.7111    -0.8282    -7.61       -8.62        -6.65     1.00000     1.0        -0.778       12        False
```

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

```
             task  metric         floor floor_err   floor_ECE sota sota_err    sota_ECE  proposed proposed_err proposed_ECE
     wesad_stress 1-AUROC       rep:raw    0.0571 0.108/0.092 sota   0.2126 0.187/0.105 cacmf_e2e       0.1334  0.251/0.167
     deap_arousal 1-AUROC classical_gbm    0.4503 0.140/0.075 sota   0.4880 0.297/0.304 rep:fused       0.4561  0.305/0.313
cgmacros_diabetes 1-AUROC       xgboost    0.0175 0.063/0.041 sota   0.2506 0.188/0.139 rep:fused       0.1291  0.123/0.094
 cgmacros_glucose     MAE       rep:raw   10.8829           —    —      nan           — rep:fused      11.7111            —
```

- **wesad_stress**: vs floor (rep:raw 0.0571): proposed cacmf_e2e 0.1334 -> does NOT beat; vs SOTA (sota 0.2126): BEATS.
- **deap_arousal**: vs floor (classical_gbm 0.4503): proposed rep:fused 0.4561 -> does NOT beat; vs SOTA (sota 0.4880): BEATS.
- **cgmacros_diabetes**: vs floor (xgboost 0.0175): proposed rep:fused 0.1291 -> does NOT beat; vs SOTA (sota 0.2506): BEATS.
- **cgmacros_glucose**: vs floor (rep:raw 10.8829): proposed rep:fused 11.7111 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **wesad_stress** (1-AUROC, ): fused 0.1568 vs rep:raw 0.0571 -> RER -174.6% (95% CI -331.6..-97.4, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **deap_arousal** (1-AUROC, ): fused 0.4561 vs classical_gbm 0.4503 -> RER -1.3% (95% CI -9.6..8.0, Wilcoxon p=0.8252, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **cgmacros_diabetes** (1-AUROC, ): fused 0.1291 vs xgboost 0.0175 -> RER -636.6% (95% CI -1552.2..-218.9, Wilcoxon p=0.9980, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **cgmacros_glucose** (MAE, ): fused 11.7111 vs rep:raw 10.8829 -> RER -7.6% (95% CI -8.6..-6.7, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- **cgmacros_glucose**: failures by config = {'sota': 12}; unstable (NaN >20% folds) = ['sota']

## Per-configuration CV error (lower is better)


### wesad_stress  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
       config  1-AUROC
      rep:raw   0.0571
      xgboost   0.0597
classical_gbm   0.0640
      rep:pca   0.1208
    cacmf_e2e   0.1334
  single:resp   0.1381
    rep:fused   0.1568
single:motion   0.1870
         sota   0.2126
   rep:neural   0.2413
   single:eda   0.2463
   single:ecg   0.2662
   single:ppg   0.2743
  single:temp   0.2783
      rep:llm   0.2970
       rep:vq   0.3550
   single:emg   0.3952
     majority   0.5000
```

### deap_arousal  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
           config  1-AUROC
    classical_gbm   0.4503
single:physiology   0.4507
        rep:fused   0.4561
           rep:vq   0.4565
       rep:neural   0.4614
          xgboost   0.4633
          rep:pca   0.4837
             sota   0.4880
         majority   0.5000
          rep:llm   0.5053
        cacmf_e2e   0.5190
          rep:raw   0.5221
       single:eeg   0.5466
```

### cgmacros_diabetes  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
              config  1-AUROC
             xgboost   0.0175
          single:ehr   0.1072
          single:cgm   0.1186
           rep:fused   0.1291
             rep:raw   0.1317
              rep:vq   0.1648
           cacmf_e2e   0.1716
          rep:neural   0.1850
             rep:pca   0.2387
                sota   0.2506
             rep:llm   0.2704
            majority   0.5000
       classical_gbm   0.5000
single:wearable_phys   0.6167
```

### cgmacros_glucose  (SOTA backend: moment:AutonLab/MOMENT-1-large)
```
       config     MAE
      rep:raw 10.8829
   single:cgm 10.8829
ridge_history 10.8829
      rep:pca 10.8852
      xgboost 11.0236
classical_gbm 11.0729
  persistence 11.3477
    rep:fused 11.7111
       rep:vq 11.8811
   rep:neural 12.0526
      rep:llm 12.5095
    cacmf_e2e 13.2274
         sota     NaN
```
