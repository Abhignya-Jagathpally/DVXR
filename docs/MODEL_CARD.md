# Model Card — DVXR NeuroGlycemic Sentinel

**Product headline (research-stage):** the **DVXR NeuroGlycemic Sentinel** — a multimodal
**glucose-excursion early-warning** framework that aims to predict 30/60-minute glucose-instability
risk from recent CGM dynamics, acute neural/autonomic stress state, and clinical context, then produce
a grounded explanation and a protocol-controlled next action.

**Status of the headline:** **RESEARCH-STAGE — NOT YET VALIDATED.** The fused end-to-end
glucose-excursion claim requires **synchronized same-subject** EEG+wearable+CGM pilot data, which does
not yet exist. Public datasets validate the individual *components* (below) but, being separate
cohorts, cannot establish that EEG adds value to CGM forecasting. Fusion on unrelated cohorts is
blocked by the synchronized-same-subject gate, and the default glucose report **abstains** until pilot
data exists. The vision is real; the fused product is not yet a claim.

**What it is:** a research-grade multimodal toolkit whose *validated components* each return a
*calibrated* risk probability, a risk band, a conformal interval, an explanation, and the held-out
accuracy of the model that produced it.

**What it is not:** a diagnostic device. It is **not a diagnosis**. Every output is a research-cohort
estimate, not a clinical determination, and must not be used to diagnose, treat, or make medical
decisions.

---

## Intended use

- **Users:** BCI / digital-health researchers exploring EEG- and wearable-based screening.
- **Use:** score research-cohort subjects (or compatible device exports) to triage / prioritize /
  study risk. Decision-*support* and research only.
- **Out of scope (never supported):** clinical diagnosis, autonomous decisioning, deployment on
  patients, any use on populations unlike the research cohorts below.

## Validated components (the modules the glucose architecture is built from)

These are the individually validated encoders/screeners the NeuroGlycemic Sentinel architecture is
assembled from (spec §1.A "public datasets for component development"). They are real, scoreboard-traced
results — and they are *component* validation, not the fused end-to-end claim, which stays gated on
synchronized same-subject data (spec §1.B). All numbers are AUROC under **subject-held-out**
cross-validation and trace to a committed scoreboard file
(`dvxr.serve.evidence.verify_against_scoreboards()` re-checks them). AUROC = 1 − the scoreboard's
`1-AUROC` error cell.

| Capability | Model | AUROC (95% CI) | Beats | Source |
|---|---|---|---|---|
| **Depression (MDD vs healthy), resting EEG** | real LaBraM EEG FM (frozen linear-probe) | **0.961 (0.942–0.976)** | band-power 0.889, GBM 0.930, raw-CNN 0.840, CACMF 0.795 | `outputs/_dnh_labram/…csv` |
| Acute stress, wearable physiology | band-power + tuned GBM | 0.955 (0.930–0.978) | CACMF 0.871 | `outputs/benchmark_scoreboard.csv` |
| Cognitive workload (rest vs task) | ECG autonomic (task best) | 0.740 | LaBraM-EEG 0.663 > band-power EEG 0.636 | `outputs/benchmark_scoreboard.csv` |
| Stress, peripheral physiology | band-power (concat) | 0.892 | CACMF 0.871 | `outputs/benchmark_scoreboard.csv` |

### Two granularities (depression)

The depression AUROC is reported at **window-level 0.961** (every held-out window scored) AND
**subject-level 0.986** (CI 0.966–0.999, n=58 — each subject's windows aggregated to one probability,
then AUROC over subjects). Subject-level is *higher* because averaging a subject's ~14 window
probabilities denoises the per-subject signal, so the headline is **not** inflated by window-pooling.
Honest caveats: N=58 is small (wide CI), and Mumtaz is a comparatively separable cohort. Workload and
stress are *within-subject state* tasks (a subject carries both classes) → the epoch/window-level
AUROC is the appropriate unit; a subject-level number does not apply and is not reported.

### DVXR vs published SOTA (same/comparable cohort — protocol-labeled)

Cross-subject (leave-one-subject-out / subject-independent) is the honest bar; many published 90%+
numbers are segment-level with subject leakage and are **not** comparable to our subject-held-out CV
(shown for context, never as a head-to-head win). Numbers via PubMed.

| Cohort | DVXR (subject-held-out CV) | Published (cross-subject) | Published (segment-level) |
|---|---|---|---|
| Depression EEG | window **0.961** / subject **0.986** AUROC | MDD-SSTNet LOSO **65.1% acc** on MODMA ([doi](https://doi.org/10.1093/cercor/bhae505)); Metin ext-val **73.3%** ([doi](https://doi.org/10.1177/15500594241273181)) | EEGNet 93.7% ([doi](https://doi.org/10.1515/bmt-2021-0232), 3-ch segment) |
| WESAD stress | window **0.955** AUROC | Vos LOSO **~85% acc** ([doi](https://doi.org/10.1016/j.jbi.2023.104556)) | Ghosh 94.8% ([doi](https://doi.org/10.3390/bios12121153)); EDA 96.4% ([doi](https://doi.org/10.1142/S0129065724500278)) |
| eegmat workload | window **0.663** (EEG) / **0.740** (ECG) AUROC | Khanam subject-independent MAT ([doi](https://doi.org/10.1371/journal.pone.0291576)) | Yedukondalu 97.4% ([doi](https://doi.org/10.1038/s41598-024-84429-6), 4 s segment) |

Different cohorts (MODMA/HUSM ≠ Mumtaz) and different protocols are noted per row; our contribution
is a *calibrated, cross-subject-honest, evidence-forward* screener, not a leaderboard accuracy.

**Method contribution:** reliability-gated **do-no-harm** late fusion (`dnh_gated`) beats the
proposal's own learned cross-modal CACMF fusion on **4 of 6** tasks and the best single modality on
3 of 6 — a nuanced positive (universal held-out do-no-harm does *not* hold at N≤60; reported, not
hidden). Provenance: Super-Learner oracle (van der Laan et al., 2007).

## Data

Research cohorts only: Mumtaz 2016 (MDD resting EEG, 19-ch @ 64 Hz), PhysioNet EEG Mental-Arithmetic
(Zyma et al., 2019), WESAD (Schmidt et al., 2018), PhysioNet Non-EEG (Birjandtalab et al., 2016). No
patient data. Subject-disjoint folds throughout; the head is never fit on a test subject.

## Calibration & uncertainty

Platt calibration on out-of-fold predictions (ECE reported per screener); risk bands
(low <0.25, watch <0.50, elevated <0.75, high); a 90% conformal interval from held-out residuals.

**Clinical utility — decision-curve analysis (net benefit).** AUROC measures ranking, not whether
*acting* on the screen helps. Each screener also carries a **decision curve** (Vickers & Elkin,
2006, [doi](https://doi.org/10.1177/0272989X06295361)) computed from the same held-out out-of-fold
predictions behind the AUROC — subject-level for the single-class-per-subject screening tasks,
epoch-level for within-subject state tasks. It plots net benefit — `TP/n − (FP/n)·pₜ/(1−pₜ)` — against the decision threshold `pₜ`,
beside the two default policies (**treat-all**, **treat-none**), and reports the threshold band where
screening beats *both*. The "useful" verdict is **bootstrap-gated**: the peak net-benefit gain's
one-sided 95% lower bound must stay positive, so a noise-level advantage (a random score can win a
single threshold by chance) reads as *not useful* rather than being oversold. Surfaced in the
per-subject report and the evidence one-pager; the honest negative — where a screener shows no
stable net-benefit advantage — is reported, not hidden.

**Serve-time personalization (opt-in, `dvxr fit --personalize`).** A per-subject recalibrator
(`PersonalizedCalibrator`) can be wired into the screener for *within-subject* state tasks (a subject
carries both classes; it does not apply to subject-level-diagnosis tasks like depression, where it is
a correct no-op). Honest finding: on these **small** research cohorts it does **not** improve
calibration — on WESAD (8 subjects) personalized ECE is *worse* (0.156 vs population 0.124) because
each subject's calibration sample is too small. It is therefore **off by default**, retained as a
documented opt-in with the negative reported rather than hidden. Persistence is versioned
(`dvxr-screener/2`) and back-compatible (v1 artifacts load unchanged).

## Clinical evaluation metrics (for the glucose product, when synchronized data exists)

AUROC measures ranking, not operational usefulness. The glucose early-warning product is evaluated
with the clinically-relevant metrics in `dvxr.eval.clinical_metrics` (spec §9): **sensitivity at a
prespecified false-alert rate**, **false alerts per participant-day** (alert fatigue), **median event
lead time** and **fraction of events detected ≥15 min early**, plus **Brier / RMSE / MAE / bias** for
the glucose forecast. Within-person evaluation uses a **chronological personalization split** (each
participant's earliest window is baseline/calibration, later windows are evaluation) so a participant's
future never leaks into its own baseline. These are reported only on synchronized same-subject data;
until such pilot data exists the fused product remains research-stage and abstains — no operational
number is claimed. Research-grade decision-support, not a diagnosis.

## Explicit limitations (the honesty gate)

- **Fidelity-limited EEG:** the EEG cohorts are 64 Hz (≤32 Hz content) vs LaBraM's 200 Hz training;
  wins are real but under-resourced on sampling rate and would plausibly improve at native rate.
- **Small cohorts (N≤60):** CIs are wide; the do-no-harm floor diverges from held-out subjects.
- **Improvement experiment — two negatives, one real win (all measured):** a pre-registered SimVQ
  tokenizer *underperformed* the existing VQ (negative); the sensor-dropout showdown found graceful
  degradation but no CI-backed crossover (negative); but raising the LLM-path VQ training epochs 8→30
  lifts the proposed fLLM's held-out AUROC 0.716→0.843 (CI-backed win, folded in). The improved fLLM is
  still the weakest predictor (below the 0.955 winner) — no product claim changes. Full record:
  `docs/IMPROVEMENT_EXPERIMENT.md`.
- **Not claimed — and blocked from being claimed:** DEAP affective decoding (at chance), the learned
  CACMF fusion as a win (loses on all 6 tasks), the LLM-in-the-loop as a predictor (weakest;
  explanation-only), MIMIC mortality (untrustworthy here), and the old `cgmacros_diabetes` numbers
  (label leakage). A CI honesty-audit test asserts these never surface as product claims.

## Ethical & safety notes

Offline / CPU / deterministic default path — no network, no PHI egress. Screening ≠ diagnosis;
outputs carry a mandatory research-prototype caveat. A positive screen is a prompt to consult a
qualified clinician, never a conclusion.

## Reproduce

```
pip install -e .
dvxr fit --task mumtaz_depression --out screeners/depression   # AUROC ~0.96
dvxr predict --screener screeners/depression                   # score a held-out subject
dvxr report                                                     # this evidence, scoreboard-traced
dvxr demo                                                       # self-contained HTML on real subjects
```

The depression headline was re-derived from raw EEG + the real LaBraM model (byte-for-byte identical
board) and its provenance is guarded offline; full record and commands in
[`REPRODUCE.md`](REPRODUCE.md).
