# Model Card — DVXR Screen

**What it is:** a research-grade, multimodal **clinical-risk screening** toolkit. Given a subject's
biosignals it returns a *calibrated* risk probability, a risk band, a conformal interval, an
explanation, and the held-out accuracy of the model that produced it. Headlined by **depression
screening from a short resting-EEG recording** (real LaBraM EEG foundation model).

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

## The models (validated, headline-able)

All numbers are AUROC under **subject-held-out** cross-validation and trace to a committed
scoreboard file (`dvxr.serve.evidence.verify_against_scoreboards()` re-checks them). AUROC = 1 −
the scoreboard's `1-AUROC` error cell.

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

## Explicit limitations (the honesty gate)

- **Fidelity-limited EEG:** the EEG cohorts are 64 Hz (≤32 Hz content) vs LaBraM's 200 Hz training;
  wins are real but under-resourced on sampling rate and would plausibly improve at native rate.
- **Small cohorts (N≤60):** CIs are wide; the do-no-harm floor diverges from held-out subjects.
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
