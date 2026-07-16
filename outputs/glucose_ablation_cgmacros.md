# Glucose-excursion ablation — CGMacros (single cohort, honest)

Prospective 30/60-min glucose-excursion prediction (`dvxr.targets`, threshold `pilot-v1`:
hypo <70, hyper >180 mg/dL) on the **CGMacros** cohort (34 subjects, same-subject CGM + Fitbit
HR/METs).

**Corrected methodology** (`src/dvxr/eval/glucose_ablation.py`): subject **K-fold with disjoint test
folds** (each participant scored once), a **calibration subset carved from each fold's training
subjects**, the alert threshold **frozen on the calibration fold** (never read on the test rows), arms
paired by **exact example key** (subject|anchor|horizon — not positional truncation), a
**participant-level bootstrap** for the delta CI (resampling subjects, not correlated rows), and
**real observed person-time** for the false-alert rate. Reproduce with `run_glucose_ablation(load_cgmacros())`.

| Arm | AUROC | AUPRC | Sens@frozen thr | FA/participant-day | Brier | ECE | Status |
|---|---|---|---|---|---|---|---|
| CGM-only | 0.909 | 0.887 | 0.784 | 0.80 | 0.093 | 0.038 | evaluated (single-modality) |
| CGM + wearable (Fitbit HR/METs) | 0.907 | 0.889 | 0.774 | 0.88 | 0.091 | 0.026 | evaluated (same-subject, non-EEG) |
| CGM + EEG | — | — | — | — | — | — | **cannot_evaluate** — no cohort co-registers EEG+CGM |
| Fused EEG+CGM+wearable (headline) | — | — | — | — | — | — | **cannot_evaluate** — no synchronized pilot data |

**Paired delta (CGM+wearable − CGM-only): −0.003 AUROC, 95% CI [−0.018, +0.008] → does NOT add value**
(the CI straddles 0; participant-level bootstrap, n=4,913 paired examples across 34 subjects).

## Honest negative result (supersedes the earlier claim)

An earlier run reported **+0.042 AUROC, CI [0.031, 0.056] → "adds value."** That interval was
**statistically overconfident** — it came from a row-level bootstrap (correlated, overlapping windows
treated as independent), an operating threshold read on the same test predictions it scored,
predictions pooled across seeds, and arms aligned by `min(len)` truncation. Under the corrected
participant-level methodology the wearable benefit **disappears**: adding Fitbit HR/METs does **not**
measurably improve prospective CGM-only excursion prediction on CGMacros. AUPRC/Brier/ECE are
approximately tied (a marginal calibration improvement, not a discrimination gain).

**What this does NOT show.** It says nothing about EEG, and it is not the fused product claim. Every
EEG/fused arm is `cannot_evaluate` because no public cohort co-registers EEG with CGM on the same
subjects. The fused NeuroGlycemic Sentinel headline stays research-stage and abstains until
synchronized same-subject EEG+wearable+CGM pilot data exists. Numbers here are single-modality-family,
CGMacros-only, at threshold `pilot-v1`.
