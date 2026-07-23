# The framework, simply — what each part contains and how it predicts

A simple, honest picture of the whole system. Diagram: `outputs/_r2/framework_overview.png`.
It has four layers: **data → shared ingestion → parallel prediction heads → LLM explanation.**
This page says what each box actually contains, how it works, and what it contributes.

---

## Layer 1 — What data is considered

| Source | What it contains | Real dataset(s) used |
|---|---|---|
| **BCI / EEG** (Galea, EMOTIV) | multi-channel brain signals (14–32 ch) | Mumtaz-MDD, EEGMAT, DEAP, EMOTIV pilot |
| **Wearables** | heart rate, EDA, PPG/BVP, temperature, motion | WESAD, PhysioNet Non-EEG, CGMacros HR |
| **EHR + notes** | lab values, demographics, free-text clinical notes | MIMIC-IV demo, MTSamples |
| **CGM + meals** | continuous glucose (1-min) + logged carbs/macros | **CGMacros (45 subjects)**, BIG-IDEAS |

Each source is real and on disk. They are *not* all recorded on the same person — that
constraint shapes everything below.

## Layer 2 — Shared ingestion (one representation for everything)

Every source is mapped onto a **canonical event schema**, then turned into features by a
**modality-specific encoder** — the piece that "understands" that signal:

- **EEG → real LaBraM foundation model.** A pretrained EEG transformer produces a 200-d
  embedding per window. This is the encoder that lifted depression from 0.910 (hand-crafted
  band-power) to **0.961**.
- **CGM → causal history features.** 17 past-only features (current value, lags at
  5/15/30/60/120 min, deltas, rolling mean/SD, slopes). This representation is what makes
  glucose forecasting strong (see below).
- **Notes → Bio_ClinicalBERT.** A frozen clinical language model embeds each note.
- **Wearables → summary physiology features** (HR/EDA/PPG statistics per window).

Contribution: this layer is why one framework can hold five different signals — they all
leave here as comparable feature vectors/embeddings.

## Layer 3 — Prediction heads (run in parallel, each honest about its inputs)

Each head is a small model on top of its modality's features. What feeds it, how it works,
and where it stands today (all subject/patient-held-out):

| Head | Fed by | How it predicts | Status (real) |
|---|---|---|---|
| **Stress** | wearable physiology | classifier on HR/EDA/PPG features | AUROC 0.892 (PhysioNet) / 0.955 (WESAD) |
| **Anxiety** | EEG + physiology | classifier on EEG embedding + physiology | **at chance** on DEAP (data-limited, honest) |
| **Depression** | EEG (LaBraM) | logistic head on the LaBraM embedding | **AUROC 0.961** — SOTA-competitive |
| **Cognitive workload** | EEG + ECG | classifier; ECG is the strongest signal | AUROC 0.740 |
| **Glucose forecast** | CGM history + meals | availability-aware mixture-of-experts, residual over persistence | **RMSE ~13 mg/dL @30 min** — beats persistence at every horizon |

Contribution: each head answers one clinical question from the modality that actually
carries that signal. They share the framework (schema, encoders, calibration, serving,
abstention), not a single fused tensor.

## Layer 4 — The LLM (explains, never predicts)

After the heads produce numbers, a language model writes the human-readable explanation:
which inputs drove the estimate, the risk band, and the caveats. It is **gated so it can only
restate numbers the heads already produced** — it cannot invent a value (enforced by
`tests/test_no_hallucinated_numbers.py`). If inputs are missing, the system **abstains**
rather than guessing.

Contribution: turns raw probabilities into decision-support a clinician can read and trust,
without adding any unverified number.

---

## So how does it predict blood glucose "with respect to" depression, anxiety, stress?

Honestly: **it predicts each of them in the same framework, but glucose is forecast from CGM
history + meals — not from the person's stress or mood.** The reason is data, not design:

> **No open dataset records EEG + CGM on the same subject at the same time.** So a model that
> predicts glucose *from* depression/anxiety cannot be trained or validated on real data.

What the framework genuinely delivers is the **honest version** of the POW vision:
- strong, calibrated **glucose forecasting** (RMSE ~13 @30 min) from the signal that actually
  predicts near-future glucose (CGM history + meals);
- strong, SOTA-competitive **mental-health heads** (depression 0.961, stress 0.955) from EEG
  and physiology;
- one shared ingestion + encoder + calibration + explanation + abstention stack around them;
- and it **abstains** on the cross-modal glucose-from-mood claim instead of faking it.

When a cohort that co-registers EEG + CGM per subject becomes available, the same framework
fuses them with no redesign — the plumbing (schema, encoders, fusion, abstention) is already
there. Until then, "glucose from stress/mood" is a **scoped goal, honestly flagged**, not a
validated result — which is exactly what the diagram's amber note says.

See also: `docs/MODEL_ARCHITECTURE.md` (the glucose model's internals),
`docs/MODEL_JUSTIFICATION.md` (why each model), `docs/HEADS_SOTA.md` (the head numbers vs
published SOTA).
