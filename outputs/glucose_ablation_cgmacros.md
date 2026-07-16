# Glucose-excursion ablation — CGMacros (single cohort, honest)

Prospective 30/60-min glucose-excursion prediction (`dvxr.targets`, threshold `pilot-v1`:
hypo <70, hyper >180 mg/dL) on the **CGMacros** cohort (20 subjects, same-subject CGM + Fitbit
HR/METs). Protocol: **subject-held-out** outer split, a **separate subject-held-out calibration
fold** (Platt), identical target/threshold across arms, 3 seeds, paired bootstrap CI over pooled
test rows. Reproduce: `python -c "from dvxr.eval.glucose_ablation import *; ..."` (see
`src/dvxr/eval/glucose_ablation.py`).

| Arm | AUROC | Sensitivity @ FAR 0.1 | Brier | Status |
|---|---|---|---|---|
| CGM-only | 0.841 | 0.691 | 0.104 | evaluated (single-modality) |
| CGM + wearable (Fitbit HR/METs) | 0.883 | 0.771 | 0.090 | evaluated (same-subject, non-EEG) |
| CGM + EEG | — | — | — | **cannot_evaluate** — no cohort co-registers EEG+CGM |
| Fused EEG+CGM+wearable (the product headline) | — | — | — | **cannot_evaluate** — no synchronized same-subject pilot data |

**Paired delta (CGM+wearable − CGM-only): +0.042 AUROC, 95% CI [0.031, 0.056] → adds value**
(the CI lower bound clears 0). On the same subjects, Fitbit autonomic/activity data genuinely
improves prospective CGM-only excursion prediction — an honest, in-scope, single-cohort finding.

**What this does NOT show.** It does not establish that EEG adds value to CGM forecasting, and it is
not the fused product claim. Every EEG/fused arm is `cannot_evaluate` because no public cohort
co-registers EEG with CGM on the same subjects (spec §1.B). The fused NeuroGlycemic Sentinel headline
therefore stays research-stage and abstains until synchronized same-subject EEG+wearable+CGM pilot
data exists. Numbers here are single-modality-family, CGMacros-only, at threshold `pilot-v1`.

Caveat: `false_alerts_per_participant_day` from the harness scales with a nominal `participant_days`
argument and is identical across arms at a fixed FAR, so it is not a differentiator here; the honest
comparison is AUROC / sensitivity@FAR / Brier above.
