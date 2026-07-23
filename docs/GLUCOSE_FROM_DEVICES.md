# Predicting glucose from the DVXR device streams (POW Goal 1)

The LLM/multimodal framework's purpose is to predict glucose from whatever device streams
the DVXR lab provides — CGM, the wearable/pulse watch (HR/HRV), meals, and (once
co-registered data exists) EMOTIV/Galea EEG. This is done by **availability-aware fusion**:
each device is an expert, weighted by how trustworthy its data currently is, and the model
**abstains** when no device is usable. That mechanism *is* Goal 1.

## Concrete multi-device result (real, CGMacros, 45 subjects)

Trained on CGM + wearable/pulse HR + meals together, patient-disjoint, superiority gate
**passed** (beats persistence at every horizon under patient-clustered 95% CIs). Run:
`neuroglycemic-runtime/runs/cgmacros-devices-v1`.

| Horizon | All devices (RMSE) |
|---|---:|
| 30 min | **12.77** |
| 60 min | 21.92 |
| 90 min | 26.61 |
| 120 min | 29.06 |

## What each device contributes (leave-one-device-out, same test patients)

RMSE mg/dL — how much worse the forecast gets when that device is removed:

| Horizon | all devices | − wearable/pulse | − meals | − CGM (fallback) |
|---|---:|---:|---:|---:|
| 30 min | 12.77 | 12.78 | 13.14 | 35.13 |
| 60 min | 21.92 | 22.04 | 22.30 | 35.85 |
| 90 min | 26.61 | 26.93 | 26.90 | 37.67 |
| 120 min | 29.06 | 29.50 | 29.34 | 38.20 |

**Honest reading:**

- **CGM is the dominant device for glucose.** Remove it and RMSE jumps to ~35 — expected,
  since CGM measures glucose directly. Its history is the backbone of the forecast.
- **The wearable/pulse device (HR/HRV) adds little at 30 min but more at longer horizons**
  (+0.01 @30 → +0.44 @120). Physiologically sensible: as CGM autocorrelation decays over
  hours, the autonomic/activity signal starts to matter.
- **Meals add a small, steady gain** (~0.3–0.4 mg/dL across horizons).
- **Graceful degradation is real:** with CGM removed, the wearable + meals still produce a
  forecast for **~90 % of windows** (RMSE ~35) instead of the model going dark. That is the
  availability-aware value — the wearable/pulse device is a genuine fallback when CGM drops.

And from the wearable-*only* cohort (BIG-IDEAS, HR/HRV → glucose, no CGM): RMSE ~31 @30 min —
the pulse device alone is a real but weak glucose predictor, consistent with the fallback
number here.

## Where EEG (EMOTIV / Galea) fits — trained for mind, wired for glucose

- The framework **is already trained on real EEG** — the depression head reaches AUROC
  **0.961** using the real LaBraM EEG foundation model, and stress/workload heads use EEG +
  physiology. So "trained on EEG" is done, for the mental-health targets.
- **Glucose-from-EEG is a different claim** and needs a dataset with EEG *and* CGM on the
  same person at the same time. No open dataset provides that, so a glucose-from-EEG expert
  cannot be honestly *trained or validated* yet — and the honesty gate forbids faking it.
- The framework is **wired for it**: the availability-aware fusion accepts an `eeg` modality
  slot, and the LSL contract (`config/lsl_streams.json`) already defines synchronized
  `eeg + wearable + reference_glucose` streams. The moment the DVXR lab captures a
  co-registered EMOTIV/Galea + CGM session (the Week-4 plan), the EEG expert trains and joins
  the glucose fusion with no redesign — including the documented hypoglycemia-EEG signature.

In short: glucose **is** predicted from the DVXR devices today (CGM + wearable/pulse + meals);
EEG is trained for the mental-health heads and is a ready, honest slot for glucose the instant
co-registered data exists.
