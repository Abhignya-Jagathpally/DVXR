# LLM predictor — missing-modality robustness (cgmacros_diabetes)

The shared head is trained ONCE on all modalities; at test time each modality is individually replaced by its learned absent token. A single-modality model cannot do this at all — it needs its one modality present.

Full-modality test error (1-AUROC): **0.2247**

| dropped at test | 1-AUROC | degradation |
|---|---|---|
| none | 0.2247 | +0.0000 |
| cgm | 0.3341 | +0.1094 |
| wearable_phys | 0.2988 | +0.0741 |
| ehr | 0.1674 | -0.0573 |
