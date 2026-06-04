# Goal 1 Model Card

## Data

- Rows: 178254
- Subjects: 3
- Sessions: 3
- Modalities: cgm, eda, eeg, ehr, motion, ppg, resp, temp

This demo uses synthetic public-data-shaped fixtures. Treat its metrics as pipeline validation, not scientific evidence.

## Stress Risk Model

- Task: binary stress vs non-stress classification from wearable and EEG windows.
- Split: subject-level train/calibration/test split.
- Calibration: Platt calibration on a held-out calibration split.
- Accuracy: 1.000
- F1: 1.000
- AUROC: 1.000
- Brier score: 0.035
- Expected calibration error: 0.171

## Glucose Forecast Model

- Task: short-horizon glucose forecasting from recent CGM features.
- Uncertainty: split-conformal residual interval from held-out calibration data.
- MAE: 5.76 mg/dL
- RMSE: 6.93 mg/dL
- 90% interval radius: 5.06 mg/dL
- Interval coverage: 0.333

## Limitations

- Synthetic fixtures are intentionally clean, so perfect stress scores are expected.
- Real WESAD/DEAP/CGM/EHR results must use subject-held-out or time-forward splits.
- Personalized risk must be claimed only after per-subject calibration improves held-out performance.
- LLM or agent outputs should explain model results; they should not replace deterministic signal processing.
