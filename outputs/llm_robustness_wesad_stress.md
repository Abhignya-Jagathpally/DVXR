# LLM predictor — missing-modality robustness (wesad_stress)

The shared head is trained ONCE on all modalities; at test time each modality is individually replaced by its learned absent token. A single-modality model cannot do this at all — it needs its one modality present.

Full-modality test error (1-AUROC): **0.2970**

| dropped at test | 1-AUROC | degradation |
|---|---|---|
| none | 0.2970 | +0.0000 |
| ecg | 0.2160 | -0.0810 |
| eda | 0.3665 | +0.0695 |
| emg | 0.3724 | +0.0754 |
| motion | 0.2307 | -0.0662 |
| ppg | 0.3152 | +0.0182 |
| resp | 0.3713 | +0.0744 |
| temp | 0.2762 | -0.0207 |
