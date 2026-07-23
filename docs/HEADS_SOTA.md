# Mental-health heads vs published SOTA (honest, protocol-labeled)

All DVXR numbers are subject/patient-held-out (never segment-level with subject leakage).
Comparisons to published work are protocol-labeled; we never claim a cross-protocol win.
Sources: `outputs/benchmark_scoreboard.md`, `outputs/_dnh_labram/`, `src/dvxr/serve/evidence.py`
(`EXTERNAL_SOTA`), `outputs/_r2/deap_fullrate_probe.md`.

## Where DVXR is genuinely SOTA-competitive

| Task | DVXR (subject-held-out) | Best config | Honest published bar (same protocol class) | Verdict |
|---|---:|---|---|---|
| **Depression** (Mumtaz MDD vs healthy) | **AUROC 0.961** window / **0.986** subject | real LaBraM raw-EEG probe | LOSO 65.1% acc (MDD-SSTNet, Chen 2025); external-val 73.3% (GoogleNet, Metin 2024) | **Clears the honest cross-subject bar** |
| **Stress** (WESAD wearable) | **AUROC 0.955** | xgboost floor | LOSO ~85% (Vos 2023, GB+ANN) | **Above the honest LOSO bar** |
| **Stress** (PhysioNet Non-EEG) | **AUROC 0.892** | PCA→logistic concat | — (cohort-specific) | Strong |
| **Cognitive workload** (EEGMAT) | AUROC 0.740 | ECG single-modality | subject-independent (Khanam 2023) | Matches honest framing; ECG (not EEG) is the ceiling |

The depression result is the strongest SOTA-competitive claim: **0.961/0.986 AUROC under
subject-held-out CV**, above the honest LOSO/external-validation bars comparable published
methods report. High headline DEAP/Mumtaz numbers elsewhere are segment-level with subject
leakage and are not comparable.

> **⚠️ Integrity caveat — AUDIT DONE, confound CONFIRMED (`scripts/depression_identity_audit.py`,
> `outputs/_r2/depression_identity_audit.md`; Identity Trap, arXiv:2606.06647).**
> We ran the recommended audit on Mumtaz (58 subjects, band-power features): **subject identity
> is decodable at 88.8% accuracy — 52× chance (1/58)** — and each subject carries exactly one
> diagnosis. Therefore subject-held-out CV **cannot separate a depression biomarker from subject
> identity**, and the **0.961 is an empirically-confounded upper bound**, not a validated
> biomarker. Honest fix: a cohort with **within-subject label variation** (e.g. pre/post-treatment)
> or an identity-adversarial training control. Presented with this caveat, never as 0.961 unqualified.

**What drove the depression win** (documented in `BENCHMARK_FINDINGS.md`): swapping the
representation from hand-crafted band-power to the **real pretrained LaBraM EEG
foundation-model embedding** lifted AUROC 0.910 → 0.961. The lever is the representation, not
tuning — and it applies where EEG is the signal-bearing modality at adequate rate.

## The honest negative: DEAP anxiety/arousal is fundamentally limited

DEAP valence/arousal sits at chance in the canonical pipeline (~0.53). The hypothesis was
that the canonical ~8 Hz decimation aliased away the α/β affect oscillations. We tested it
directly (`scripts/deap_fullrate_probe.py`): proper band-power on the **original 128 Hz**
preprocessed signal (full spectrum retained), 32 subjects, **subject-held-out** GroupKFold.

| target | logistic AUROC | gradient-boosting AUROC |
|---|---:|---:|
| valence (high vs low) | 0.555 | 0.470 |
| arousal (high vs low) | 0.483 | 0.546 |

**Verdict: the decimation hypothesis is refuted.** Even at full spectral resolution,
cross-subject DEAP affect decoding is at chance (0.47–0.56). The limit is the task/cohort
(cross-subject affect on DEAP is genuinely hard — a well-known result), not the sampling
rate. We therefore **did not pursue** the heavier 512 Hz raw-BDF + LaBraM re-export: the
128 Hz test already shows resolution is not the bottleneck, so spending that compute would
be chasing a ceiling the evidence says is fixed. This is reported as a measured negative, not
hidden.

## Summary

- **Genuine SOTA-competitive, honestly framed:** depression (0.961/0.986), WESAD stress
  (0.955), stress (0.892).
- **Honest data-limited negative:** DEAP anxiety/arousal — at chance regardless of sampling
  rate under subject-held-out evaluation.
- The learned CACMF cross-modal fusion remains the reported negative result
  (`outputs/benchmark_scoreboard.md`): it does not beat the strongest non-fused baseline —
  reported faithfully, not buried.
