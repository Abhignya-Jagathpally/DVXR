# What "causal" means here (and why it makes the numbers honest)

"Causal" is used in the **no-future-information** sense: every feature the model sees at a
prediction time *t* is computed **only from data at or before *t*** — never from the future
it is being asked to forecast. This is what prevents *leakage* (a model cheating by peeking
ahead), and it's the reason the reported RMSE reflects real forecasting rather than hindsight.

## Concretely, in the glucose pipeline

- **History features use past values only.** Every CGM lag/delta/rolling/slope feature is
  built with `glucose.shift(...)` and trailing windows (`cgmacros_data.py`,
  `diatrend.py`) — e.g. `cgm_lag_30m` at time *t* is the glucose at *t−30 min*. No feature
  reads a value after *t*.
- **The target is strictly separated and lies in the future.** The label
  `target_glucose_30m_mg_dl` is the real CGM reading at *t+30 min*, matched from the raw
  stream (`_nearest_future_cgm`) — kept out of the feature set entirely.
- **Availability is time-gated.** A modality can only contribute at anchor *t* if its
  `available_time ≤ t` (`neural_dataset.py` rejects any modality whose data arrived after the
  anchor). Meals/events are right-labeled so an event at 12:04 first appears in the 12:05
  forecast, never the already-issued 12:00 one.
- **Splits are patient-disjoint.** No patient appears in both train and test, so "causal in
  time" is paired with "no subject leakage across the split."

## The check that proves it

If features secretly contained the future, the 30-min target would equal the current value
and RMSE would collapse toward zero. It doesn't: the 30-min target differs from current
glucose by ~11 mg/dL on average, and the model's RMSE (~13) sits sensibly *below* the
persistence baseline (17.4) but *well above* zero — the signature of a genuine causal
forecast, not leakage. The forecast scatter widening with horizon
(`held_out_forecasts.png`) is the same story.

## Two senses of "causal" — which one this is

- ✅ **Temporal causality (used here):** no future leakage; features precede the target. This
  is a hard, enforced property of the pipeline.
- ⚠️ **Causal *inference* (not claimed):** we do **not** claim the features *cause* the future
  glucose in the interventional sense (e.g. "raising HR *causes* the glucose change"). The
  model is a calibrated *forecaster*; contributions (`docs/EXPLAINABILITY.md`) are associational
  attributions, honestly labeled as such — not causal-effect estimates.

Keeping these separate is itself part of the honesty posture: the pipeline is temporally
causal (no cheating), and it does not overreach into causal-effect claims it cannot support.
