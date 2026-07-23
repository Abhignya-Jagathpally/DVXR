# Fine-tuned models for the 7 POW clinical-risk tasks

The POW asks the framework to fine-tune models for: stress detection, anxiety prediction,
depression risk, cognitive workload, glucose instability, diabetes complication risk, and
clinical risk prediction. Below is the honest status of each — the selected model (per
`docs/LITERATURE_REVIEW.md`), its real dataset, the held-out metric, and where it stands.
Reproduce: `python scripts/finetune_tasks_scoreboard.py` → `outputs/_r2/finetuned_tasks_scoreboard.{md,csv}`,
`presentation/figures/fig_finetuned_tasks.png`.

| Task | Selected model | Metric (subject/patient-held-out) | Status |
|---|---|---|---|
| **Stress detection** | wearable physiology (WESAD) | AUROC **0.955** | ✅ validated |
| **Anxiety prediction** | EEG + physiology (DEAP) | AUROC ~0.53 | ⚠ data-limited — at chance (documented ceiling) |
| **Depression risk** | LaBraM EEG FM (Mumtaz) | AUROC **0.961** | ✅ validated — *pending identity-leakage audit* |
| **Cognitive workload** | EEG + ECG (EEGMAT) | AUROC **0.740** | ✅ validated (ECG-dominant) |
| **Glucose instability** | CGM deep model (CGMacros) | AUROC **0.976** hypo / **0.981** hyper | ✅ strong |
| **Diabetes complication risk** | — | — | ⚠ honest gap — no open dataset carries real complication labels |
| **Clinical risk (mortality)** | GBM on MIMIC-IV labs | AUROC **0.813** | ✅ trained here (15/252 events, 5-fold grouped CV; small-n) |

## Reading it honestly

- **Five of seven are fine-tuned with real, validated held-out metrics.** Glucose instability
  (hypo/hyper event detection at ~0.98 AUROC) and stress (0.955) are the strongest; clinical
  mortality (0.813) is real but small-n and should be read as indicative.
- **Two are honest gaps, labelled not faked:**
  - *Anxiety (DEAP)* sits at chance — a data-fidelity ceiling confirmed even at full sampling
    rate (`docs/HEADS_SOTA.md`), not a fixable modeling failure.
  - *Diabetes complication risk* has **no real labels** in any open dataset that also carries
    these signals, so it cannot be honestly fine-tuned or validated. The serving path exposes a
    clearly-labelled experimental heuristic that **abstains**, never a trained clinical claim.
- **Depression 0.961** carries the identity-leakage caveat (Identity Trap, arXiv:2606.06647) —
  treat as an upper bound pending the recommended audit.

## Selected models (per the literature review)

- **EEG tasks** (depression / anxiety / workload): real **LaBraM** foundation model, frozen
  embedding + fine-tuned head (kept per the review; optional CBraMod A/B).
- **Stress / physiology**: gradient-boosted / physiology models on WESAD/PhysioNet.
- **Glucose instability**: the CGM deep model (GRN + conv + ensemble) with hypo/hyper heads +
  calibrated intervals.
- **Clinical risk**: gradient boosting on MIMIC-IV labs; Bio_ClinicalBERT on clinical notes
  (specialty macro-AUROC 0.9606, surgery 0.910) as the text-based clinical-risk head.

Every task stays `validated_for_clinical_use = false` — clinical purpose, pre-deployment
(`docs/GOAL_ACHIEVEMENT.md`).
