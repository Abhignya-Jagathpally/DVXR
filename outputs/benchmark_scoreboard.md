# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=5, folds=5, seed=7, sota=True

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); wesad_stress = MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC); deap_anxiety = MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label); deap_arousal = MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label); eegmat_workload = MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label); mumtaz_depression = EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label). Multimodal-fusion evidence spans the peripheral-physiology stress task(s) and the DEAP EEG+peripheral affective/BCI tasks; no single dataset co-registers EEG+CGM+EHR per subject.

| task              | metric   | best_baseline     |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:------------------|:---------|:------------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| stress            | 1-AUROC  | rep:pca           |     0.1079 |     0.1294 |     -0.0215 |    -19.87 |       -28.58 |        -13.5  |      0.99999 |        1 |         -0.254 |        25 | False         |
| wesad_stress      | 1-AUROC  | xgboost           |     0.0453 |     0.1294 |     -0.0841 |   -185.9  |      -441.57 |        -80.54 |      0.99982 |        1 |         -0.667 |        25 | False         |
| deap_anxiety      | 1-AUROC  | single:physiology |     0.4658 |     0.4688 |     -0.0029 |     -0.63 |        -6.37 |          5.39 |      0.73645 |        1 |         -0.027 |        25 | False         |
| deap_arousal      | 1-AUROC  | single:physiology |     0.4522 |     0.4575 |     -0.0053 |     -1.16 |        -7.05 |          4.7  |      0.79412 |        1 |         -0.027 |        25 | False         |
| eegmat_workload   | 1-AUROC  | single:physiology |     0.2598 |     0.3649 |     -0.105  |    -40.42 |       -56.7  |        -26.8  |      1       |        1 |         -0.686 |        25 | False         |
| mumtaz_depression | 1-AUROC  | sota              |     0.0824 |     0.2046 |     -0.1222 |   -148.19 |      -224.39 |        -86.68 |      0.99999 |        1 |         -0.555 |        25 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task              | metric   | floor             |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE    | proposed   |   proposed_err | proposed_ECE   |
|:------------------|:---------|:------------------|------------:|:------------|:-------|-----------:|:------------|:-----------|---------------:|:---------------|
| stress            | 1-AUROC  | rep:pca           |      0.1079 | 0.059/0.035 | sota   |     0.1912 | 0.049/0.042 | rep:fused  |         0.1294 | 0.048/0.033    |
| wesad_stress      | 1-AUROC  | xgboost           |      0.0453 | 0.059/0.029 | sota   |     0.2217 | 0.214/0.118 | cacmf_e2e  |         0.1115 | 0.248/0.158    |
| deap_anxiety      | 1-AUROC  | single:physiology |      0.4658 | 0.285/0.308 | sota   |     0.4931 | 0.298/0.307 | rep:fused  |         0.4688 | 0.311/0.319    |
| deap_arousal      | 1-AUROC  | single:physiology |      0.4522 | 0.273/0.298 | sota   |     0.4872 | 0.291/0.300 | rep:fused  |         0.4575 | 0.302/0.311    |
| eegmat_workload   | 1-AUROC  | single:physiology |      0.2598 | 0.059/0.092 | sota   |     0.3381 | 0.270/0.025 | rep:fused  |         0.3649 | 0.123/0.027    |
| mumtaz_depression | 1-AUROC  | xgboost           |      0.0827 | 0.089/0.017 | sota   |     0.0824 | 0.092/0.027 | cacmf_e2e  |         0.1921 | 0.177/0.087    |

- **stress**: vs floor (rep:pca 0.1079): proposed rep:fused 0.1294 -> does NOT beat; vs SOTA (sota 0.1912): BEATS.
- **wesad_stress**: vs floor (xgboost 0.0453): proposed cacmf_e2e 0.1115 -> does NOT beat; vs SOTA (sota 0.2217): BEATS.
- **deap_anxiety**: vs floor (single:physiology 0.4658): proposed rep:fused 0.4688 -> does NOT beat; vs SOTA (sota 0.4931): BEATS.
- **deap_arousal**: vs floor (single:physiology 0.4522): proposed rep:fused 0.4575 -> does NOT beat; vs SOTA (sota 0.4872): BEATS.
- **eegmat_workload**: vs floor (single:physiology 0.2598): proposed rep:fused 0.3649 -> does NOT beat; vs SOTA (sota 0.3381): does NOT beat.
- **mumtaz_depression**: vs floor (xgboost 0.0827): proposed cacmf_e2e 0.1921 -> does NOT beat; vs SOTA (sota 0.0824): does NOT beat.

## Verdict

- **stress** (1-AUROC, MULTIMODAL (4 peripheral-physiology streams, one wearable)): fused 0.1294 vs rep:pca 0.1079 -> RER -19.9% (95% CI -28.6..-13.5, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **wesad_stress** (1-AUROC, MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC)): fused 0.1294 vs xgboost 0.0453 -> RER -185.9% (95% CI -441.6..-80.5, Wilcoxon p=0.9998, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **deap_anxiety** (1-AUROC, MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label)): fused 0.4688 vs single:physiology 0.4658 -> RER -0.6% (95% CI -6.4..5.4, Wilcoxon p=0.7364, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **deap_arousal** (1-AUROC, MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label)): fused 0.4575 vs single:physiology 0.4522 -> RER -1.2% (95% CI -7.0..4.7, Wilcoxon p=0.7941, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **eegmat_workload** (1-AUROC, MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label)): fused 0.3649 vs single:physiology 0.2598 -> RER -40.4% (95% CI -56.7..-26.8, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **mumtaz_depression** (1-AUROC, EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label)): fused 0.2046 vs sota 0.0824 -> RER -148.2% (95% CI -224.4..-86.7, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### stress  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config        |   1-AUROC |
|:--------------|----------:|
| rep:pca       |    0.1079 |
| rep:raw       |    0.1113 |
| xgboost       |    0.1141 |
| classical_gbm |    0.1204 |
| rep:fused     |    0.1294 |
| cacmf_e2e     |    0.1666 |
| single:motion |    0.167  |
| sota          |    0.1912 |
| single:ppg    |    0.2505 |
| rep:vq        |    0.3166 |
| rep:neural    |    0.3245 |
| single:eda    |    0.3416 |
| single:temp   |    0.3599 |
| majority      |    0.5    |

### wesad_stress  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config        |   1-AUROC |
|:--------------|----------:|
| xgboost       |    0.0453 |
| rep:raw       |    0.0528 |
| classical_gbm |    0.0582 |
| rep:pca       |    0.1042 |
| cacmf_e2e     |    0.1115 |
| single:resp   |    0.1243 |
| rep:fused     |    0.1294 |
| single:motion |    0.1369 |
| rep:neural    |    0.1985 |
| sota          |    0.2217 |
| single:eda    |    0.2407 |
| single:ppg    |    0.2577 |
| single:ecg    |    0.2623 |
| single:temp   |    0.2691 |
| rep:vq        |    0.2969 |
| single:emg    |    0.3852 |
| majority      |    0.5    |

### deap_anxiety  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.4658 |
| rep:fused         |    0.4688 |
| rep:vq            |    0.473  |
| rep:neural        |    0.4732 |
| rep:pca           |    0.4738 |
| classical_gbm     |    0.4741 |
| xgboost           |    0.4828 |
| sota              |    0.4931 |
| majority          |    0.5    |
| raw_cnn           |    0.5037 |
| cacmf_e2e         |    0.5152 |
| rep:raw           |    0.5246 |
| single:eeg        |    0.5464 |

### deap_arousal  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.4522 |
| rep:fused         |    0.4575 |
| rep:pca           |    0.4652 |
| classical_gbm     |    0.4658 |
| rep:vq            |    0.473  |
| xgboost           |    0.4764 |
| rep:neural        |    0.477  |
| sota              |    0.4872 |
| majority          |    0.5    |
| raw_cnn           |    0.5044 |
| cacmf_e2e         |    0.518  |
| rep:raw           |    0.5293 |
| single:eeg        |    0.553  |

### eegmat_workload  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.2598 |
| sota              |    0.3381 |
| xgboost           |    0.341  |
| raw_cnn           |    0.3429 |
| rep:raw           |    0.3487 |
| classical_gbm     |    0.3562 |
| rep:fused         |    0.3649 |
| single:eeg        |    0.365  |
| cacmf_e2e         |    0.3859 |
| rep:pca           |    0.3864 |
| rep:neural        |    0.4499 |
| rep:vq            |    0.4611 |
| majority          |    0.5    |

### mumtaz_depression  (SOTA backend: moment:AutonLab/MOMENT-1-large)
| config        |   1-AUROC |
|:--------------|----------:|
| sota          |    0.0824 |
| xgboost       |    0.0827 |
| classical_gbm |    0.0927 |
| rep:raw       |    0.1121 |
| single:eeg    |    0.1121 |
| raw_cnn       |    0.1662 |
| rep:pca       |    0.1757 |
| cacmf_e2e     |    0.1921 |
| rep:fused     |    0.2046 |
| rep:vq        |    0.2266 |
| rep:neural    |    0.2303 |
| majority      |    0.5    |

## True modality ablation (retrain without the modality)


### stress  (contribution = error increase when dropped)
| dropped_modality   |   err_without (1-AUROC) |   err_with_all (1-AUROC) |   contribution |   ci_low |   ci_high |
|:-------------------|------------------------:|-------------------------:|---------------:|---------:|----------:|
| motion             |                  0.212  |                   0.1264 |         0.0855 |   0.0668 |    0.1059 |
| ppg                |                  0.1468 |                   0.1264 |         0.0204 |   0.0119 |    0.0296 |
| temp               |                  0.142  |                   0.1264 |         0.0155 |   0.0085 |    0.0221 |
| eda                |                  0.1327 |                   0.1264 |         0.0063 |  -0.0032 |    0.0159 |

### wesad_stress  (contribution = error increase when dropped)
| dropped_modality   |   err_without (1-AUROC) |   err_with_all (1-AUROC) |   contribution |   ci_low |   ci_high |
|:-------------------|------------------------:|-------------------------:|---------------:|---------:|----------:|
| eda                |                  0.2022 |                   0.1417 |         0.0605 |   0.0159 |    0.114  |
| resp               |                  0.1853 |                   0.1417 |         0.0436 |   0.0003 |    0.0844 |
| motion             |                  0.1701 |                   0.1417 |         0.0284 |  -0.0088 |    0.0655 |
| temp               |                  0.1538 |                   0.1417 |         0.012  |  -0.0161 |    0.0432 |
| ppg                |                  0.1159 |                   0.1417 |        -0.0259 |  -0.0616 |    0.0153 |
| emg                |                  0.0877 |                   0.1417 |        -0.0541 |  -0.0873 |   -0.0185 |
| ecg                |                  0.0831 |                   0.1417 |        -0.0587 |  -0.09   |   -0.0265 |

### deap_anxiety  (contribution = error increase when dropped)
| dropped_modality   |   err_without (1-AUROC) |   err_with_all (1-AUROC) |   contribution |   ci_low |   ci_high |
|:-------------------|------------------------:|-------------------------:|---------------:|---------:|----------:|
| eeg                |                  0.5086 |                   0.4685 |         0.0401 |   0.0037 |    0.0774 |
| physiology         |                  0.4976 |                   0.4685 |         0.0291 |  -0.006  |    0.0676 |

### deap_arousal  (contribution = error increase when dropped)
| dropped_modality   |   err_without (1-AUROC) |   err_with_all (1-AUROC) |   contribution |   ci_low |   ci_high |
|:-------------------|------------------------:|-------------------------:|---------------:|---------:|----------:|
| eeg                |                  0.4999 |                   0.4545 |         0.0454 |   0.0197 |    0.072  |
| physiology         |                  0.4742 |                   0.4545 |         0.0196 |  -0.0113 |    0.0482 |

### eegmat_workload  (contribution = error increase when dropped)
| dropped_modality   |   err_without (1-AUROC) |   err_with_all (1-AUROC) |   contribution |   ci_low |   ci_high |
|:-------------------|------------------------:|-------------------------:|---------------:|---------:|----------:|
| physiology         |                  0.419  |                   0.3548 |         0.0642 |   0.0168 |    0.1096 |
| eeg                |                  0.3858 |                   0.3548 |         0.031  |  -0.0153 |    0.0776 |
