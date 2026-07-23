# DVXR Goals — Q&A notes (verbatim), gap-check, and closures

These notes record the goal-verification questions asked and the responses given, word for
word, starting from the pipeline-ingestion question. Two things were done while compiling:
(1) a **critical gap-check** of every Goal outcome, and (2) closing the one gap that was
actionable — the **depression identity-leakage audit** (result below), which also surfaced
and fixed a real `np.trapezoid` (numpy 2.0 vs 1.26) bug in band-power. Remaining gaps are
data-bound and honestly flagged.

---

## Critical gap-check — what was and wasn't achieved

| Item | Status |
|---|---|
| Goal-1 outcomes a–e (ingestion / embeddings / real-time / explainable biomarkers / personalized diabetes) | ✅ 5/5 **tested** (`scripts/verify_goal1_outcomes.py`) |
| Ingestion modalities a,b,c,e (wearable / EEG / biosensor / diabetes-CGM) | ✅ validated with real data |
| Ingestion modality d (behavioral metrics) | ⚠ **wired, not run on a real behavioral dataset** (schema slot + `behavior_adapter` + `ingest_vr_session.py` exist) |
| Goal 2 — 5 fusion strategies + 3 aggregators + real-time + EHR LM + datasets + standard tests | ✅ explored/implemented |
| Goal 3 — single-vs-integrated ablation + comparative table | ✅ done (`outputs/_r2/comparative_analysis.md`) |
| Fine-tuned tasks (7) | ✅ 5 validated; ⚠ anxiety data-limited; ⚠ diabetes-complication **no real labels** |
| **Depression 0.961 identity-leakage audit** | ✅ **NOW DONE — confound CONFIRMED** (see below) |

**Closure done this pass — the identity audit (the review's #1 recommendation):**
Mumtaz, 58 subjects, band-power features — **subject identity decodable at 88.8% accuracy
(52× chance)**, and diagnosis is subject-level, so subject-held-out CV cannot separate a
depression biomarker from subject identity. **The 0.961 is an empirically-confounded upper
bound.** Fix needs within-subject label variation. (`outputs/_r2/depression_identity_audit.md`.)

**Genuine remaining gaps (data-bound, not fixable by code):** behavioral-metrics ingestion on a
real dataset; diabetes-complication labels; a within-subject depression cohort. Flagged, not faked.

---

## Q1 — "Is the pipeline capable of ingesting: a) physiological wearable signals, b) EEG signals, c) biosensor streams, d) behavioral metrics, e) diabetes-related physiological monitoring data? for sure?"

Yes — four of five validated with real data; the fifth (behavioral) is wired but not yet run.

| Modality | Ingests? | Evidence |
|---|---|---|
| a) Physiological wearable | ✅ Yes, validated | WESAD · PhysioNet Non-EEG · CGMacros HR → biosignal encoder |
| b) EEG signals | ✅ Yes, validated | DEAP · EMOTIV · Galea · Mumtaz · EEGMAT → real LaBraM (depression 0.961) |
| c) Biosensor streams | ✅ Yes, validated | EDA · BVP · temp · respiration · SpO₂ → biosignal encoder |
| d) Behavioral metrics | ⚠ Wired, not yet run | canonical `behavior` slot + `behavior_adapter` + `scripts/ingest_vr_session.py` (VR/AR gaze/interaction/head-pose); no real behavioral dataset run; MOMENT encoder falls back to PCA/VQ |
| e) Diabetes / CGM | ✅ Yes, validated | CGMacros · DiaTrend · BIG-IDEAS · Shanghai → CGM-history encoder (RMSE ~13) |

Four of five are fully wired end-to-end with real data and validated results; the fifth
(behavioral) has the ingestion path built but hasn't been exercised on a real behavioral
dataset yet. I won't call it "done" when it isn't.

---

## Q2 — "So how are the heterogeneous modalities [combined] into a unified predictive framework?"

Four stages, plus one honest constraint.

1. **Canonical event schema (unify the format).** Every device export is mapped onto one shared
   13-column event schema (`dvxr.schemas.REQUIRED_EVENT_COLUMNS`) — timestamp, modality, channel,
   value, patient_id, quality, availability — so an EEG stream and a CGM stream live in one table.
2. **Per-modality encoders (unify into comparable vectors).** EEG → LaBraM (200-d); CGM →
   CGM-history features; wearable/physiology → biosignal encoder; notes → Bio_ClinicalBERT; omics/
   behavior → their adapters.
3. **Availability-aware learned fusion (the actual combining).** Per-modality embeddings become
   experts; a learned gate weights each by quality/staleness/clock-certainty and **abstains** if
   none is usable. A missing modality is **not zero-imputed** — it gets a learned "absent" token
   and is masked. Five fusion strategies + three aggregators are available.
4. **Multi-task calibrated heads** (stress/anxiety/depression/workload/glucose/clinical), each
   split-conformal calibrated.

**Honest constraint:** genuine cross-modal fusion is only *learned/validated* where modalities
co-occur on the same subject (wearable+CGM in CGMacros; EEG+physiology in DEAP). No open dataset
co-registers EEG+CGM+EHR per subject, so the framework fuses co-occurring clusters and degrades
gracefully across the gaps. The benchmark honestly reports that learned fusion (CACMF) does not
yet beat naive concat / strong single-modality baselines (Holm p=1.0).

---

## Q3 — "What was the Multimodal Fusion Strategy? Which architectures (all or partial) were capable of integrating EEG/BCI, wearable, glucose, diabetes biomarkers, EHR — explored and implemented?"

**Overall:** per-modality latent → VQ codebook → selected fusion strategy → joint latent → heads
(`CACMFModel`, modality-agnostic). On glucose, `LearnedMaskedFusion` (availability-aware
mixture-of-experts).

**Implemented (all 5 the POW proposed):** `EarlyFusion`, `IntermediateFusion`, `LateWeightedFusion`,
`AttentionFusion`, `CrossModalFusion` (`nn.TransformerEncoder`). Plus three aggregation baselines:
`ensemble_avg`, `weighted_late`, `confidence_weighted`.

**Modalities integrable (all — modality-agnostic on latents):** `MODALITIES = [eeg, wearable_phys,
cgm, ehr, ehr_notes, omics, behavior]`. Mapped to the list: EEG/BCI (LaBraM), wearable (biosignal),
glucose (CGM), diabetes biomarkers (CGM-derived hypo/hyper + metabolic meta-model + omics), EHR
(structured + Bio_ClinicalBERT notes). Integrates any subset (partial or full).

**Explored vs implemented vs validated:** all implemented + benchmarked; genuine cross-modal fusion
validated only on co-occurring subsets; learned fusion does not yet beat simple baselines on the
current data — reported honestly.

---

## Q4 — "Were all the following fusion strategies explored: Early, Intermediate, Late, Attention-based, Cross-modal transformers?"

Yes — all five were run and measured, not just coded. Each has evaluated ablation rows on multiple
tasks. Glucose (MAE): early 22.99, intermediate 22.34, late_weighted 21.96, attention 26.23,
cross_modal 23.83; each also has a classification-task row (AUROC). (`outputs/ablation_summary.md`.)

---

## Q5 — "Was the baseline followed: weighted late fusion, ensemble averaging, and confidence-weighted prediction aggregation?"

Yes, followed accordingly. All three implemented — `ensemble_avg`, `weighted_late`,
`confidence_weighted` (`src/dvxr/fusion/aggregate.py`, registered in `AGGREGATORS`), tested
(`tests/test_fusion.py` — 9 passed + 10 subtests), and `ensemble_avg` appears in the committed
ablation with measured results.

---

## Q6 — "What Real-Time Capability was explored: streaming wearable/EEG input, continuous glucose monitoring, real-time stress inference, adaptive intervention recommendations?"

All four explored/implemented:
1. **Streaming wearable/EEG → LSL** (`lsl.py` + `lsl_windowing.py`, `config/lsl_streams.json` with
   eeg/wearable/reference_glucose; CLI lsl-audit/replay/session-replay).
2. **Continuous glucose (streaming)** — reference_glucose stream + `realtime_bridge.py` rt-demo-v1
   frames (glucose point + interval, honest abstention); `streaming_eval.py`.
3. **Real-time stress inference** — `FusedRealtimeMonitor.stream_fused_predictions`; EMOTIV PM.Stress.
4. **Adaptive interventions → rule-based JITAI** (`realtime/intervention.py`): condition → approved
   policy action via `dvxr.safety.policy.select_action`; LLM may rephrase, never originate.

**Honest caveats:** the real-time monitors are flagged `EXPERIMENTAL_ONLY`/`NOT_FOR_CLINICAL_INFERENCE`
(transparent heuristics for demo); interventions are policy-gated (not autonomous); live LSL needs
`pylsl` on the lab machine (present + exercised via replay here).

---

## Q7 — "Were the benchmark datasets explored: MIMIC-IV, PhysioNet, WESAD, DEAP, diabetes monitoring datasets, etc?"

Yes — all present and used, with committed results: MIMIC-IV (mortality 0.813; ehr-glucose),
PhysioNet (Non-EEG stress 0.892; CGMacros; CogWear), WESAD (stress 0.955), DEAP (anxiety, honest
data-limit), diabetes/CGM (CGMacros, DiaTrend, BIG-IDEAS, Shanghai), plus Mumtaz (depression),
EEGMAT (workload), MTSamples (notes), and the real EMOTIV/Galea device sessions.

---

## Q8 — "Design of the fusion strategy based on existing literature."

- Strategy taxonomy = classic multimodal-ML fusion levels (early/intermediate/late) + attention +
  cross-modal transformer (MulT-style).
- Per-modality foundation-model encoders (LaBraM, Bio_ClinicalBERT) + VQ-VAE codebooks.
- **Default = availability-aware (masked) late fusion**, grounded in the missing-modality literature
  (arXiv:2409.07825; VCR arXiv:2605.18837) — robust + interpretable when modalities are absent.
- Contrastive cross-modal alignment (CLIP/InfoNCE) evaluated as an option and **deferred** — real
  but dataset-specific, not shown to beat availability-aware fusion on an external cohort (review §3).
- Empirically validated: the ablation shows learned cross-modal fusion does not beat availability-
  aware late fusion / concat on current data — consistent with the literature that late fusion is a
  strong, robust baseline. (`docs/LITERATURE_REVIEW.md §3`.)

---

## Q9 — "The proposed integration strategy. In its simplest form, was a late-fusion weighted-average model implemented?"

Yes. `weighted_late` in `src/dvxr/fusion/aggregate.py`: `p = Σ_m w_m p_m` with weights normalized
**over present modalities** (availability-aware); its plainest case `ensemble_avg` = `mean_m p_m`.
Companion baselines `ensemble_avg` and `confidence_weighted` also implemented; registered in
`AGGREGATORS`, tested. It's the product default; the more complex learned fusion was explored but
does not outperform it yet.

---

## Q10 — "Were standard tests performed to score model performance for the prediction task?"

Yes, rigorously. Protocol: repeated subject/patient-held-out grouped CV (repeats=5, folds=5,
seed=7). Metrics: AUROC, Average Precision, Brier, F1, accuracy, ECE (classification); RMSE, MAE,
R², MARD, PI-95 coverage/width, MASE (regression). Significance: paired one-sided Wilcoxon, Holm
correction, bootstrap 95% CI, Cliff's delta, patient-clustered CIs, deterministic reproduction.
Results committed (`benchmark_scoreboard`, `clinical_notes_scoreboard`, `finetuned_tasks_scoreboard`,
`sota_comparison`). Strict bar (RER ≥50% AND beat floor+SOTA); many configs don't clear it and that
is reported, not hidden.

---

## Q11 — "Was an LLM pipeline prepared for EHR — a transformer clinical LM ingesting structured and unstructured EHR?"

Yes, both sides:
- **Unstructured (clinical LM):** frozen **Bio_ClinicalBERT** over note text (`NotesEHRAdapter`),
  4,499 MTSamples notes, 5-fold grouped CV (surgery-binary + 40-way specialty) → clinical_notes_scoreboard.
- **Structured:** `EHRAdapter` (CEHR-BERT-style encoder) on MIMIC-IV labs/demographics → mortality
  0.813, EHR-glucose.
- Both are modalities (`ehr` + `ehr_notes`) in the fusion framework.
Honest nuance: the clinical transformer is a frozen embedding extractor (not a generative LLM
predicting); the generative LLM is confined to explanation. Optional upgrade: Clinical ModernBERT.

---

## Q12 — "Was an ablation study performed comparing the integrated model with a single modality?"

Yes, extensively: single (eeg/wearable/cgm) vs fusion vs aggregation per task
(`outputs/ablation_summary.md`); real glucose leave-one-modality-out (CGMacros); and fused-vs-best-
single-modality columns in `outputs/benchmark_scoreboard.md` with RER, Wilcoxon, Holm.

---

## Q13 — "Was single modality benchmarked on a test dataset for the prediction task?"

Yes — single-modality models were scored on held-out test data (the `single eeg/wearable/cgm` rows,
the leave-one-modality-out, and the `best_baseline` = single-modality opponent columns).

---

## Q14 — "Prepare a table for comparative performance analysis." (Goal-3 deliverable — done)

Committed as `outputs/_r2/comparative_analysis.{md,csv}` + `presentation/figures/fig_comparative_analysis.png`.

| Task | Metric | Best single modality | Integrated fusion | Verdict | Holm p |
|---|---|---:|---:|:--|---:|
| Stress (PhysioNet) | AUROC ↑ | 0.892 | 0.871 | single-modality wins | 1.0 |
| Stress (WESAD) | AUROC ↑ | 0.955 | 0.871 | single-modality wins | 1.0 |
| Anxiety (DEAP) | AUROC ↑ | 0.534 | 0.531 | ~tie (both chance) | 1.0 |
| Arousal (DEAP) | AUROC ↑ | 0.548 | 0.542 | single-modality wins | 1.0 |
| Cognitive workload (EEGMAT) | AUROC ↑ | 0.740 | 0.635 | single-modality wins | 1.0 |
| Depression (Mumtaz) | AUROC ↑ | 0.918 | 0.795 | single-modality wins | 1.0 |
| **Glucose (CGMacros)** | RMSE@30 ↓ | 13.33 (CGM only) | **12.99 (CGM+meals)** | **integrated wins** | — |

**Conclusion:** multimodal integration is not universally better — it pays off where modalities carry
complementary signal on the same subject (glucose: CGM+meals 12.99 < CGM-only 13.33, +wearable 12.77),
and adds noise where one modality dominates (mental health). Measured on subject/patient-held-out
splits with Wilcoxon + Holm; negatives included.
