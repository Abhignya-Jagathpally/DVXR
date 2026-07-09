# LLM predictor — missing-modality robustness (deap_arousal)

The shared head is trained ONCE on all modalities; at test time each modality is individually replaced by its learned absent token. A single-modality model cannot do this at all — it needs its one modality present.

Full-modality test error (1-AUROC): **0.5053**

| dropped at test | 1-AUROC | degradation |
|---|---|---|
| none | 0.5053 | +0.0000 |
| eeg | 0.5164 | +0.0112 |
| physiology | 0.5013 | -0.0039 |
