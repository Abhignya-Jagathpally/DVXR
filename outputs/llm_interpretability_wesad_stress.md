# LLM predictor interpretability — wesad_stress

Backend: `Qwen/Qwen2.5-0.5B-Instruct`

Modality attribution — share of the frozen-LLM representation shift when each modality is replaced by its learned absent token (higher = more influential):

- **emg**: 0.254
- **ecg**: 0.219
- **eda**: 0.180
- **motion**: 0.108
- **ppg**: 0.096
- **temp**: 0.076
- **resp**: 0.066
