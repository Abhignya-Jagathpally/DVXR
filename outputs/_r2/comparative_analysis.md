# Comparative performance analysis — single modality vs integrated (Goal 3)

Committed, subject/patient-held-out results. AUROC ↑ for classification; RMSE ↓ for glucose. No configuration is assumed to win — measured, with statistical tests.

| task                        | metric             |   best_single_modality |   integrated_fusion |   delta | verdict              |   holm_p |
|:----------------------------|:-------------------|-----------------------:|--------------------:|--------:|:---------------------|---------:|
| Stress (PhysioNet)          | AUROC              |                  0.892 |               0.871 |  -0.021 | single-modality wins |        1 |
| Stress (WESAD)              | AUROC              |                  0.955 |               0.871 |  -0.084 | single-modality wins |        1 |
| Anxiety (DEAP)              | AUROC              |                  0.534 |               0.531 |  -0.003 | ~tie                 |        1 |
| Arousal (DEAP)              | AUROC              |                  0.548 |               0.542 |  -0.006 | single-modality wins |        1 |
| Cognitive workload (EEGMAT) | AUROC              |                  0.74  |               0.635 |  -0.105 | single-modality wins |        1 |
| Depression (Mumtaz)         | AUROC              |                  0.918 |               0.795 |  -0.123 | single-modality wins |        1 |
| Glucose forecast (CGMacros) | RMSE@30 (mg/dL, ↓) |                 13.33  |              12.99  |  -0.34  | integrated wins      |      nan |

## Verdict
- **Mental-health / EEG tasks:** the integrated learned fusion does **not** beat the best single modality — every task's fusion RER is negative and non-significant (Holm p=1.0). The strongest *single* modality (wearable for stress, EEG/LaBraM for depression, ECG for workload) wins.
- **Glucose:** integration **helps** — CGM+meals (12.99) beats CGM-only (13.33) @30 min, and adding the wearable/pulse device lowers it further (12.77). This is the one task where the real data co-registers multiple informative modalities per subject.
- **Honest conclusion:** multimodal integration is not universally better; it pays off where modalities carry complementary signal on the same subject (glucose), and adds noise where one modality dominates (mental health). Reported exactly as measured.
