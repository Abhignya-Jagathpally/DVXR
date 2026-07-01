# Decoding Intended Movement from Consumer-Grade EEG: A Multimodal Manifold-Based BCI Pipeline

*Draft methods + results section (IEEE conference format) — DVXR Lab, UNT, Summer 2026.*
*Author: Abhignya Jagathpally. Advisors: Dr. Sharad Sharma (Director), Dr. Bishnu Sarker.*

> Status: working draft generated from the validated pipeline (`scripts/run_bci_pipeline.py`,
> `scripts/run_fusion_ablation.py`). All numbers below are reproduced by re-running those
> scripts on the recordings in `data/`. Omics modality intentionally deferred. Figures are in
> `outputs/bci/`.

---

## Abstract

We present a reproducible pipeline that ingests consumer- and research-grade brain–computer
interface (BCI) recordings — EMOTIV EPOC X and Galea/OpenBCI — into a single canonical event
schema, builds per-window spectral and physiological features, learns a low-dimensional neural
*manifold*, and decodes intended movement in a leakage-controlled manner. On a 23-minute EMOTIV
session in which a subject issued four trained mental commands (Left, Right, Push, Pull) to move a
virtual cube, a multinomial decoder attains **0.82 balanced accuracy** under trial-grouped
cross-validation and **0.72** under a stricter temporal-block scheme that controls for session
drift (chance = 0.25). A dissociation — direction decodes far above chance while an
engaged-vs-neutral detector sits at chance (AUROC 0.49) — indicates the decoder reads the spatial
pattern that distinguishes movement *types* rather than a global arousal state. We further evaluate
weighted late fusion across EEG, head-motion, and affective streams (Goal 2) and report a
single-vs-multimodal ablation (Goal 3). This mirrors, on wearable hardware, the real-time
neural-manifold movement decoding of avatar-control systems (avatarRT; MRAE; TPHATE).

## I. Introduction

Real-time decoding of intended movement from a learned neural manifold has enabled closed-loop
avatar control from intracranial and fMRI signals [avatarRT; MRAE]. Manifold-geometry methods such
as PHATE and its temporally-aware variant TPHATE [TPHATE] recover smooth low-dimensional structure
from high-dimensional neural time series. We ask whether an analogous manifold-decoding workflow can
operate on inexpensive, dry/saline consumer EEG, using the EMOTIV mental-command paradigm as a
wearable stand-in for intended-movement decoding, as a first step toward the multimodal clinical and
mental-health analytics targeted by the broader project.

## II. Methods

### A. Data acquisition

Two devices were recorded. **EMOTIV EPOC X** (serial E5020C07): 14 channels (10–20 montage:
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) at 128 Hz for 1373 s (175 857 samples),
exported from EmotivPRO with the on-board Mental-Command, band-power (POW.\*), Performance-Metrics
(PM.\*), and motion (MOT.\*) streams. The subject issued four trained commands plus Neutral; command
labels were taken from the `MC.Action`/`MC.ActionPower` stream. **Galea/OpenBCI** (BrainFlow export):
16 channels at 250 Hz, resting/unlabeled, used for multi-device ingestion and signal-quality
assessment.

### B. Preprocessing and epoching

Recordings are read directly from their archives into a canonical long event schema. The EEG is
sliced into 2 s windows at 0.5 s step. Each window is labeled by the dominant active mental command
(`ActionPower` > 0.05), else Neutral. To control information leakage between overlapping windows, a
*trial* identifier groups temporally-contiguous windows that share a label; all cross-validation
folds keep a trial intact.

### C. Features

Two spectral feature sets are computed per window: (i) **Welch** relative band power per channel in
five bands (θ 4–8, α 8–12, β-low 12–16, β-high 16–25, γ 25–45 Hz), recomputed from the raw EEG; and
(ii) the device's own **POW** FFT band power averaged over the window. Two auxiliary modalities are
aggregated per window: **motion** (accelerometer/quaternion/magnetometer, mean+std) and **affective**
performance metrics (only EMOTIV *Excitement* was active in this session; Stress/Engagement/etc. were
inactive).

### D. Neural manifold

A diffusion-based manifold is learned with PHATE (built-in temporal diffusion-map fallback when the
package is unavailable). A BIOT/MRAE-style masked-feature transformer autoencoder additionally
produces a 16-d self-supervised embedding (`emotiv_encoder.pt`).

### E. Decoding and evaluation

A standardized multinomial logistic regression with balanced class weights is evaluated by
**StratifiedGroupKFold (5-fold, grouped by trial)**. We report balanced accuracy, macro-F1, and
macro one-vs-rest AUROC against the 5-class chance rate (0.20). For the avatar-control framing we
report a two-stage analysis: (A) engaged-vs-Neutral detection, and (B) 4-class command direction
among engaged windows (chance 0.25). To control for slow session drift we add a **temporal-block CV**
that splits each command's windows chronologically into four contiguous blocks.

### F. Multimodal late fusion (Goal 2/3)

Per-modality decoders (EEG, motion, affective) are trained on identical folds; their out-of-fold
class probabilities are combined by mean and by confidence-weighted averaging (weights ∝ each
modality's CV balanced accuracy above chance). The single-vs-fusion comparison constitutes the Goal 3
ablation.

## III. Results

### A. 4-class command decoding (Goal 1)

Under trial-grouped CV the 4-class command decoder reaches **balanced accuracy 0.822, macro-F1 0.819**
(n = 569 engaged windows). Per-class recall: Left 0.75, Right 0.80, Push 0.82, Pull 0.92 (Fig. 2).
Under the drift-controlled temporal-block CV it reaches **balanced accuracy 0.722, macro-F1 0.663** —
a modest, expected drop that confirms the decode reflects movement geometry rather than nearby-in-time
leakage.

**TABLE I — 5-class decoding by feature set (trial-grouped CV, chance bal-acc 0.20)**

| Feature set | # feat | balanced acc | macro-F1 | macro-AUROC |
|---|---|---|---|---|
| Welch band power | 70 | 0.265 | 0.175 | 0.722 |
| EMOTIV POW (FFT) | 70 | 0.303 | 0.197 | 0.782 |
| Welch + POW + motion + PM | 162 | 0.275 | 0.182 | 0.754 |

The engaged-vs-Neutral detector gives AUROC **0.49** (≈ chance). The dissociation between this and the
0.72–0.82 direction decode shows the discriminative signal is the spatial pattern separating movement
*types*, not a global engagement/drift state. A single hemisphere-β contrast separates Left vs Right at
AUROC **0.54** only, so the direction signal is not simple contralateral μ/β lateralization — consistent
with idiosyncratic strategies learned under EMOTIV biofeedback.

### B. Neural manifold

PHATE embeddings of the band-power windows show the four commands occupying distinguishable regions of
the manifold against a faint Neutral background (Fig. 1). The self-supervised autoencoder latent, decoded
linearly, underperforms the explicit band-power features (bal-acc 0.16), indicating that for this small,
single-session dataset hand-built spectral features remain stronger than the learned embedding.

### C. Multimodal late fusion and ablation (Goal 2/3)

**TABLE II — single modality vs late fusion, 4-class command (chance 0.25)**

| Modality | # feat | balanced acc | macro-F1 |
|---|---|---|---|
| EEG (Welch + POW) | 140 | **0.780** | 0.787 |
| Motion (MOT.\*) | 20 | 0.605 | 0.579 |
| Affective (Excitement) | 2 | 0.539 | 0.517 |
| Fusion — mean | 162 | 0.772 | 0.767 |
| Fusion — confidence-weighted | 162 | 0.780 | 0.782 |

EEG is the dominant modality; motion and affective streams carry weaker but clearly above-chance signal
(Fig. 3). Confidence-weighted fusion (weights EEG 0.45 / motion 0.30 / affective 0.25) *matches* but does
not exceed the best single modality, indicating the auxiliary streams are largely redundant with EEG for
this task. The above-chance motion result is noted as a possible head-movement/EMG correlate of the
mental-command paradigm and a target for artifact control in future work.

### D. Multi-device ingestion

The Galea/OpenBCI session ingests at 250 Hz with **11/16 channels usable**; five channels were railed
(poor contact) and flagged automatically. A resting-EEG PHATE manifold colored by time is recovered
(Fig. 6), demonstrating the schema and manifold stages generalize across devices.

### E. Figures

| Fig. | File | Content |
|---|---|---|
| 1 | `outputs/bci/manifold_emotiv.png` | PHATE neural manifold; full session and command-only |
| 2 | `outputs/bci/command_confusion.png` | 4-class command confusion matrix (held-out) |
| 3 | `outputs/bci/ablation.png` | Single modality vs late fusion |
| 4 | `outputs/bci/realtime_decode.png` | Streaming decode — true vs decoded timeline + 90 s zoom |
| 5 | `outputs/bci/channel_band_importance.png` | Explainable channel × band decoder weights |
| 6 | `outputs/bci/galea_quality.png` | Galea signal quality + resting manifold |

Top explainable biomarkers (mean |decoder weight|): AF4 β-low, T8 γ, O2 β-high, FC6 γ, O1 β-low (Fig. 5).

## IV. Discussion and limitations

Results are from a single subject and single session; the figures should be read as a working
demonstration of the pipeline, not a population claim. Command blocks cluster in time, so even the
temporal-block CV cannot fully exclude slow non-stationarity; the 0.72 drift-controlled figure is the
conservative headline. The affective channel was limited to *Excitement* because other EMOTIV
performance metrics were inactive in this recording — and notably no usable stress label was present,
motivating dedicated stress-elicitation data collection.

## V. Future work

A Unity + Lab Streaming Layer (LSL) stress protocol (~15–20 min blocked design: eyes-open/closed
baseline → Stroop and mental-arithmetic stressors with social-evaluative threat → recovery, with STAI-6
and SAM self-reports as ground truth) will provide the stress-labeled EEG the current data lacks, with
markers time-synced across EMOTIV (Cortex/band-power outlet) and Galea (BrainFlow→LSL). Subsequent work
adds the deferred multi-omics modality and the EHR clinical-language path toward the full multimodal
clinical-risk objective.

## References (to expand for submission)

- E. Busch et al., real-time neural-manifold avatar decoding (*Nature Neuroscience*, 2026) — avatarRT.
- Manifold-Regularized Autoencoder (MRAE) for neural decoding.
- TPHATE / PHATE: temporally-aware manifold embedding of neural time series (KrishnaswamyLab).
- EMOTIV EPOC X / EmotivBCI documentation; Galea Beta EEG documentation; BrainFlow.
