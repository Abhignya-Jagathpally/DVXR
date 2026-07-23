# DEAP full-rate (128 Hz) band-power probe — subject-held-out

Subjects: 32 | features: 128 (32 EEG ch x 4 bands, relative power) | 5-fold GroupKFold by subject. Chance = 0.50; the decimated canonical pipeline sits ~0.53.

| target | logistic AUROC | gradient-boosting AUROC |
|---|---:|---:|
| valence (high vs low) | 0.555 | 0.470 |
| arousal (high vs low) | 0.483 | 0.546 |
