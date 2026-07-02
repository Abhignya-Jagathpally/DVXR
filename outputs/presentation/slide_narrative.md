# Slide narrative (honest)

## 1. Title — DVXR: multimodal health signals, honestly evaluated
- What we built + the honesty stance. Figure: none. Note: lead with the framing.

## 2. The pipeline (CACMF)
- Per-modality encoders -> VQ codebook -> cross-modal fusion -> calibrated heads.
- Figure: codebook_usage.png. Note: architecture is real and runs offline/CPU.

## 3. The honest benchmark
- Real labels, subject/patient-held-out 5x5 CV, CIs + significance.
- Figure: benchmark_scoreboard.png. Note: "fusion does NOT beat strong baselines — and reporting that is the contribution."

## 4. Where multimodality DOES help
- Concatenation beats best single modality (stress); learned fusion doesn't beat concat.
- Figure: fusion_vs_concat.png + modality_ablation.png. Note: motion dominates stress.

## 5. BCI pilot — honest controls
- Single-subject/single-session; labels from Emotiv MC engine (not cues).
- Figure: bci_honest_controls.png. Note: engaged 0.489, lateralization 0.541 (chance); 0.82 -> 0.45 under leave-one-block-out.

## 6. BCI geometry (exploratory)
- Figure: bci_manifold.png + bci_confusion.png. Note: interesting geometry, NOT validated decoding.

## 7. Limitations & next steps
- Summary-stat features; single wearable/subject; no EEG+CGM+EHR co-registration.
- Next: raw-signal encoders, cued multi-subject BCI (PhysioNet MI), nested-CV headline.
