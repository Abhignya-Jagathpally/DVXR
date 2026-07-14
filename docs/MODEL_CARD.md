# Model Card — DVXR Screen

**What it is:** a research-grade, multimodal **clinical-risk screening** toolkit. Given a subject's
biosignals it returns a *calibrated* risk probability, a risk band, a conformal interval, an
explanation, and the held-out accuracy of the model that produced it. Headlined by **depression
screening from a short resting-EEG recording** (real LaBraM EEG foundation model).

**What it is not:** a diagnostic device. Every output is a research-cohort estimate, not a clinical
determination. It must not be used to diagnose, treat, or make medical decisions.

---

## Intended use

- **Users:** BCI / digital-health researchers exploring EEG- and wearable-based screening.
- **Use:** score research-cohort subjects (or compatible device exports) to triage / prioritize /
  study risk. Decision-*support* and research only.
- **Out of scope:** clinical diagnosis, autonomous decisioning, deployment on patients, any use on
  populations unlike the research cohorts below.

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
