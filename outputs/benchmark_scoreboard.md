# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=2, folds=3, seed=7, sota=True

**Protocol:** repeated subject/patient-held-out 5x5 CV (single-level; opponent selection and RER share folds — nested CV deferred)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); glucose = single-modality (CGM only); mortality = single-modality (EHR only). Multimodal-fusion conclusions rest on the **stress** task; no dataset co-registers EEG+CGM+EHR per subject.

```
             task  metric best_baseline  base_err  prop_err  delta_abs  RER_pct  RER_CI_low  RER_CI_high  p_wilcoxon  p_holm  cliffs_delta  n_folds  meets_>=50%
cgmacros_diabetes 1-AUROC       xgboost      0.01    0.1203    -0.1103 -1103.03    -1666.67      -350.51         1.0     1.0        -0.889        6        False
```

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

```
             task  metric   floor floor_err   floor_ECE sota sota_err    sota_ECE  proposed proposed_err proposed_ECE
cgmacros_diabetes 1-AUROC xgboost    0.0100 0.068/0.041 sota   0.1892 0.179/0.125 cacmf_e2e       0.1192  0.256/0.176
```

- **cgmacros_diabetes**: vs floor (xgboost 0.0100): proposed cacmf_e2e 0.1192 -> does NOT beat; vs SOTA (sota 0.1892): BEATS.

## Verdict

- **cgmacros_diabetes** (1-AUROC, ): fused 0.1203 vs xgboost 0.0100 -> RER -1103.0% (95% CI -1666.7..-350.5, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

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
             rep:llm   0.2247
             rep:pca   0.2624
            majority   0.5000
       classical_gbm   0.5000
single:wearable_phys   0.5479
```
