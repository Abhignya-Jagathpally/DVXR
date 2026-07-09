# Triangulated benchmark + LLM-in-the-predictive-path (branch `fixes-and-assets`)

Two gaps the user called out are closed: the benchmark now **triangulates** against an
honest floor and an open-weight SOTA ceiling, and the LLM is put in the **predictive path**
(not just the explanation layer). Honest reporting held — the floor still wins on clean
tabular; nothing is faked.

- **Calibration in the harness** (`dvxr.calibration`): added `TemperatureScaler` +
  `fit_temperature_scaler` (logit-space, AUROC-preserving). `bench.run` now pools
  out-of-fold predictions per config and reports **ECE raw / after temperature scaling**
  for every classification config. `tests/test_calibration_temp.py` guards it.
- **Floor opponents** (`bench.baselines`, import-guarded so the offline suite stays green):
  `xgboost` tuned-GBM floor, `tabpfn` (TabPFN-v2, small-tabular classification), and
  `ridge_history` (CGM forecast floor). Registered only when their dep imports.
- **Triangulation scoreboard** (`bench.scoreboard`): a "floor vs SOTA vs proposed" section
  buckets every config into floor / SOTA / proposed and prints, per task, the strongest of
  each with err + ECE and a verdict *vs floor* and *vs SOTA* separately. A win must beat
  **both**. SOTA encoders that can't load on this box are labeled, not faked.
- **True LOSO**: `run_benchmark.py --loso` sets folds = #subjects, repeats = 1.
- **LLM predictor — Option 3 soft-prompt fusion** (`dvxr.llm.predictor`, `rep:llm`): each
  modality's VQ codebook tokens are projected into a frozen causal-LM's embedding space
  (default `Qwen/Qwen2.5-0.5B-Instruct`, CPU; `DVXR_LLM_PREDICTOR`/CUDA → 7B), prepended as
  soft-prompt tokens to a short grounded prompt, and the pooled hidden state feeds the
  **same shared bench head** as every other representation — so `rep:llm` flows through the
  identical LOSO + RER/Wilcoxon/Holm stats. The frozen forward is run once per window and
  cached (label-free, no leakage), matching the SOTA-opponent contract. Predict and explain
  stay separate: the head originates the number; `llm/insight.py` only narrates.
  - **Interoperability**: a missing modality uses a learned **absent token**, so arbitrary
    modality subsets work at test time. `--llm` writes `outputs/llm_robustness_<task>.md` —
    train on all modalities, test with each dropped (graceful degradation a single-modality
    model can't do). On `cgmacros_diabetes`, dropping cgm hurts most (+0.109 1-AUROC),
    dropping ehr slightly helps (−0.057, a noisy modality here).
  - **Interpretability**: `outputs/llm_interpretability_<task>.md` reports per-modality
    attribution (L2 shift of the representation when each modality → absent token):
    wearable_phys 0.402 > cgm 0.344 > ehr 0.254.
  - **Honest result**: `rep:llm` predicts (cgmacros_diabetes err 0.225, ECE 0.229) but, as
    expected, loses to the xgboost floor (0.010) on clean tabular — reported, not hidden.
    `tests/test_llm_predictor.py` (skip-guarded on a local LLM) covers it.

---

# Real-dataset refactor (branch `inherit-skills-and-real-datasets`)

First careful step toward the POW goal: inherit external capability skills and wire three
real datasets into the pipeline (replacing synthetic/placeholder modalities), keeping the
offline/CPU/deterministic guardrails and honest reporting.

- **Skills inherited** into `.claude/skills/`: `addyosmani/agent-skills` (24 engineering
  skills) + `imbad0202/academic-research-skills` (deep-research, academic-paper,
  academic-paper-reviewer, academic-pipeline). `karpathy/autoresearch` vendored under
  `third_party/` as GPU-only reference (not a skill).
- **Canonical schema relaxed** (`schemas.validate_events`): the 13 columns are now a
  required floor, not an exact set — loaders may carry dataset-specific extras
  (`glucose_source`, `meal_photo_path`, …).
- **WESAD** (Siegen `WESAD.zip`): `fetch_wesad_siegen` + `load_wesad_dataset`; real
  baseline/stress/amusement/meditation labels across chest+wrist physiology.
- **CGMacros** (PhysioNet): new `load_cgmacros_*` loaders splitting each subject CSV into
  `cgm` (Libre+Dexcom), `wearable_phys` (Fitbit), `behavior` (meal macros), plus `bio.csv`
  → `ehr`; real diabetes strata derived from HbA1c. Profiler rule + zip-in-zip fetch added.
- **DEAP** (Kaggle `sayuksh/deap-datasetraw-data`): `load_deap_dataset` auto-detects
  preprocessed `.dat` vs raw `.bdf` (`load_deap_raw_bdf` via `mne`, ratings-joined);
  `fetch_kaggle` accepts the newer `KAGGLE_API_TOKEN` (KGAT_) bearer token; `fetch_deap`
  links the kagglehub cache into `data/real/deap`. `deap_arousal` benchmark task added.
- **Real benchmark**: new `bench.tasks` builders `wesad_stress`, `cgmacros_glucose`,
  `cgmacros_diabetes`, `deap_arousal` run under held-out-subject CV. Honest result — CACMF
  fusion does **not** beat baselines (RER: WESAD stress -81.7%, CGMacros diabetes -10.6%,
  CGMacros glucose -7.3%, DEAP arousal -3.2%); see `outputs/benchmark_scoreboard.md`.
- **Data-cleanliness fixes** (value-level audit): raw DEAP `.bdf` now drops the `Status`
  trigger channel, band-passes (0.5-45 Hz) and average-references the EEG (removing the
  large BioSemi DC offsets), and keeps the file's real channel names — EEG comes out
  mean~0, ±hundreds µV instead of ±tens of thousands. WESAD chest/wrist channels are now
  location-prefixed (`chest_eda`/`wrist_eda`) so same-modality streams no longer collide,
  and ACC units are honest (`counts` vs `counts_1_64g`). Documented the intentional
  Libre+Dexcom dual-CGM rows in CGMacros (disambiguated by `glucose_source`).
  `tests/test_data_cleanliness.py` guards all three.

---

# CHANGES — final fixes & presentation assets (branch `fixes-and-assets`)

Honest self-audit against the code review (C1/C2, M1–M5, minors). Guardrails held:
offline/CPU, deterministic (seed=7), one commit per prompt, no fabricated numbers,
every reported number traces to an `outputs/` file, single-subject/proxy/exploratory
results labelled as such.

## Fixed

- **M1 — reproducibility.** `ingest_emotiv`/`ingest_galea` accept a `.zip`, a directory,
  or a direct CSV; `run_bci_pipeline.py` gained `--emotiv/--galea` + a `$DVXR_BCI_DATA`
  resolver that defaults to the committed sample and stamps `data_source`
  (`full`|`sample`). On the tiny sample it exits cleanly (writes `metrics_sample.json`,
  never clobbers the full-run `metrics.json`). `tests/test_bci_smoke.py` (5 tests) and
  `BCI_PIPELINE.md` provenance added.
- **C1 — circular BCI labels.** Confirmed labels come from Emotiv's MC engine (MC.Action)
  + POW features, with the `intervalMarker.csv` empty (no cues). Reframed everywhere:
  `labels_source: emotiv_mc_engine`, command decode DEMOTED, honest controls
  (engaged 0.489 / lateralization 0.541 AUROC, chance 0.5) lead. See `metrics.json`
  framing + `bci_honest_controls.png`.
- **C2 — single subject/session.** Labelled single-subject, single-session, exploratory
  in every caption/brief/narrative; a leave-one-time-block-out control (raw-EEG welch,
  balAcc 0.445 vs chance 0.25) was added to show the block/time confound.
- **M2 — silent NaN swallowing.** `bench/run.py` now logs each config/fold failure with
  type, counts them, flags configs that NaN on >20% of folds as "unstable", excludes
  them from best-baseline selection, and surfaces failures/unstable in the scoreboard.
- **M4 — multimodal labeling.** Scoreboard + findings state stress = multimodal,
  glucose = CGM-only, mortality = EHR-only, and that fusion conclusions rest on stress.
- **M5 — transductive SOTA leak.** SOTA baseline now uses the RAW frozen-FM embedding
  (`make_primary_backend._embed`) with a per-fold StandardScaler+head; no PCA is fit
  over test rows. Benchmark re-run to refresh numbers under the corrected protocol.
- **Minors.** m1: `codebook_loss` documented as diagnostic-only (zero-gradient under the
  EMA buffer); bench `vq` documented as the continuous latent. m3: `test_package_parity`
  guards goal1_pipeline↔dvxr drift. m4: headset serial scrubbed from all committed files.
- **E — presentation assets.** `scripts/make_presentation_assets.py` emits the honest
  pack (8 figures, 7 tables csv+md, self-contained dashboard, results_brief.docx,
  slide_narrative.md, MANIFEST) — nothing shows the 0.82 as success or fusion as a win.

## Deferred (with reason)

- **M3 — nested-CV / frozen-test headline.** The headline RER still uses single-level
  repeated 5×5 CV (opponent selection and RER evaluation share folds). This is now
  explicitly LABELLED in the scoreboard/protocol rather than hidden. A nested-CV or
  frozen-test headline is a heavier restructure + full re-run; deferred to keep the
  asset deliverable unblocked (per the ordering rule). The result direction (fusion
  loses on every task) is not expected to flip under a stricter protocol.
- **Public cued BCI dataset (PhysioNet MI).** User authorised sourcing public data; a
  genuine cued, multi-subject motor-imagery decode (leave-one-subject-out) would be the
  strongest honest BCI headline, replacing the single-subject MC-engine story. Deferred
  as an enhancement to avoid blocking Prompt E; scoped in FIX_PLAN.md.
- **BCI raw-EEG cue relabeling.** Impossible with the committed data: no experimenter
  cues exist and the full raw recording is not committed (only band-power windows).

## Self-audit

The core dishonesty risks flagged by the review are addressed: the proposed model is
evaluated (not decorative), on non-circular real labels, with the BCI result demoted to
an honest exploratory pilot and fusion reported as losing (negative RER, CIs, Wilcoxon).
The remaining rigor gap (M3 nested CV) is labelled, not concealed. No output presents an
unsupported win; every number in `outputs/presentation/` traces to a source file via
`MANIFEST.md`.
