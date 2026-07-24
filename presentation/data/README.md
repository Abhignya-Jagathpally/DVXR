# Presentation data package — DVXR multimodal clinical-risk

Every CSV here is copied or curated **verbatim from real committed result files** in this
repo. No number was invented, edited, or "improved". Where the honest result is a **negative**
(learned CACMF fusion does NOT beat the best single modality on the mental-health / EEG tasks;
fusion only helps for glucose), it is reported as measured — see the verdict columns.

All headline metrics are **subject/patient-held-out** (never segment-level with subject leakage).

## Top-level

| File | Source(s) | What it shows |
|---|---|---|
| `best_models_summary.csv` | `outputs/_r2/finetuned_tasks_scoreboard.csv`, `outputs/_r2/comparative_analysis.csv`, `outputs/_labram_mumtaz/benchmark_scoreboard.csv`, `outputs/_labram_eegmat/benchmark_scoreboard.csv`, `outputs/benchmark_scoreboard.csv`, `outputs/clinical_notes_scoreboard.csv`, `outputs/product/screeners/mumtaz_depression/manifest.json`, `outputs/_r2/eval_cgmacros_v1.log`, `neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1/missing_modality_ablation.csv` | One row per task: the genuinely best real config, its metric/value, protocol, source, and a `caveat` column carrying every honesty caveat (identity-leakage confound, small-n, honest gaps). |

## Goal 2 — Multimodal Integration Framework (`goal2_fusion/`)

| File | Source | What it shows |
|---|---|---|
| `fusion_vs_baseline_scoreboard.csv` | `outputs/benchmark_scoreboard.csv` (verbatim) | (a) Integrated learned fusion (`prop_err`) vs best single-modality/baseline (`base_err`) per task, with RER, CIs, Wilcoxon/Holm p, Cliff's delta. `meets_>=50%` is False everywhere — fusion does **not** clear the bar on stress/DEAP/EEGMAT/Mumtaz. |
| `fusion_strategies_modalities.csv` | `src/dvxr/fusion/strategies.py`, `src/dvxr/fusion/aggregate.py`, `src/dvxr/bench/gated_fusion.py`, `src/dvxr/encoders/*` (verified in code) | (b) All 5 POW fusion strategies (early / intermediate / late-weighted / attention / cross-modal transformer) + 3 POW baselines (ensemble avg, weighted late, confidence-weighted) + gated abstaining fusion, and all 5 POW modality encoders — each marked implemented=yes with its file. |
| `ehr_llm_clinical_notes_scoreboard.csv` | `outputs/clinical_notes_scoreboard.csv` (verbatim) | (c) EHR LLM pipeline: Bio_ClinicalBERT on MTSamples. Surgical AUROC **0.910** (ClinicalBERT wins); 40-way specialty macro-AUROC **0.961** where **TF-IDF beats ClinicalBERT** (0.931) — honest split, reported as measured. |
| `benchmark_datasets.csv` | curated from the scoreboards above (each row cites its source) | (d) POW benchmark-dataset coverage: WESAD, PhysioNet, DEAP, Mumtaz, EEGMAT, CGMacros, MIMIC-IV, MTSamples — dataset, modality, task, best metric, protocol. |
| `glucose_model_ladder.csv` | `outputs/_r2/glucose_model_ladder.csv` (verbatim) | Glucose forecast model ladder (persistence -> ridge -> tree -> RF -> GBM -> MLP -> neuroglycemic_net) across 30/60/90/120 min. Honest: GBM (12.48 @30) edges the deep net (12.99) on the tabular ladder. |

## Goal 3 — Ablation: integrated vs single modality (`goal3_ablation/`)

| File | Source | What it shows |
|---|---|---|
| `comparative_performance.csv` | `outputs/_r2/comparative_analysis.csv` (verbatim) | **THE deliverable table.** Best single modality vs integrated fusion + delta + verdict per task. Single-modality wins on Stress (PhysioNet/WESAD), Arousal, Workload, Depression; ~tie on Anxiety; **integrated wins only on Glucose (CGMacros)**. Verdicts are exactly as measured. |
| `leave_one_modality_out_cgmacros.csv` | `neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1/missing_modality_ablation.csv` (verbatim) | Real per-modality CGMacros ablation: RMSE/MAE/MARD per horizon (30/60/90/120) per scenario (observed / without_cgm / without_events / all_unavailable). Dropping CGM roughly triples error; dropping meal events adds ~0.3 RMSE — quantifies the fusion gain. |
| `wesad_modality_contribution.csv` | `outputs/presentation/tables/modality_ablation_table.csv` (verbatim) | WESAD single-stream drop-one contribution (motion > ppg > temp > eda) with bootstrap CIs. |

## Honest negatives to carry into the deck (do NOT present as wins)

1. **Learned CACMF fusion loses to the best single modality on every mental-health / EEG task**
   (Stress PhysioNet -0.021, Stress WESAD -0.084, Arousal DEAP -0.006, Workload EEGMAT -0.105,
   Depression Mumtaz -0.123). `meets_>=50%` is False across the board in the scoreboard.
2. **Integrated fusion helps ONLY for glucose** (CGMacros RMSE@30 12.99 vs CGM-only 13.33, -0.34).
   This is the single genuine integration win.
3. **Depression 0.961/0.986 is an identity-leakage-confounded upper bound**, not a validated
   biomarker (subject identity decodable at 88.8%, 52x chance; diagnosis is subject-level).
4. **Anxiety/Arousal (DEAP) sit at chance** — a documented data/cohort ceiling, not a fixable model.
5. **Diabetes complication risk has no real labels** — an honest gap; the model abstains.
6. **On 40-way specialty, TF-IDF beats Bio_ClinicalBERT** — the transformer does not win everywhere.
7. **On the glucose tabular ladder, GBM edges the deep net** at 30 min (12.48 vs 12.99).
8. **MIMIC-IV mortality 0.813 is small-n** (15/252 events) — indicative, not validated.
