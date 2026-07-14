# DVXR — Multimodal Health Signal Ingestion & Modeling Pipeline

A reproducible pipeline that ingests **wearable/PHR, BCI/EEG, CGM/diabetes, and EHR**
signals into one canonical event schema, builds per-modality features, trains
auditable baseline models, and reports calibrated, explainable predictions.

The code starts with classical features and lightweight encoders because they run
reliably and make data quality visible. It keeps adapter points for larger foundation
models (see `model_choice_registry.csv`):

- **EEG/BCI:** EEG-X, then LaBraM / BENDR.
- **Wearable physiology:** BIOT for heterogeneous biosignals, MOMENT for generic windows.
- **CGM:** GluFormer if weights/data access are available, else a conformalized forecasting baseline.
- **EHR:** Med-BERT/BEHRT for structured events, NYUTron/Foresight for note/concept timelines.

## DVXR Screen — the product (`dvxr.serve` + `dvxr` CLI)

The research above is packaged into a usable, research-grade **clinical-risk screening** toolkit,
headlined by **depression screening from a short resting EEG**. Install it and score a subject:

```
pip install -e .
dvxr fit     --task mumtaz_depression --out screeners/depression   # held-out AUROC ~0.96
dvxr predict --screener screeners/depression                       # calibrated score + explanation
dvxr report                                                        # scoreboard-traced evidence
dvxr demo                                                          # self-contained HTML on real subjects
```

**Watch the pipeline run live.** The interactive app scores a real held-out subject (or your own
upload) *on the spot* — raw EEG → LaBraM embedding → calibration → risk → explanation, computed
per-Run in a fraction of a second after warmup (verified: live single-subject score reproduces the
cohort score exactly):

```
pip install -e ".[app]"                    # adds streamlit
dvxr demo --serve                          # or: streamlit run scripts/screen_app.py
dvxr screen --file resting_eeg.edf         # headless upload path (.edf/.bdf/.csv)
```

Held-out cohort subjects carry the validated benchmark AUROC; **uploads are flagged
out-of-distribution / illustrative** — the validated number applies to the research cohort, not an
arbitrary recording.

`dvxr.serve.Screener` is the missing "train once → save → load → **predict(new subject)**" path: it
wires each task to the representation that actually *wins* the benchmark — the real **LaBraM** EEG
foundation model for EEG screening, calibrated band-power for wearable stress — returns a Platt-
calibrated probability, a risk band, and a conformal interval, and carries the *same* subject-held-out
AUROC the benchmark reports. Explanations come through `dvxr.serve.explain` (grounded, always caveated).

**Validated capabilities** (AUROC, subject-held-out CV, each traced to `outputs/*scoreboard*`; see
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and `dvxr.serve.evidence`):

| Capability | Model | AUROC (95% CI) |
|---|---|---|
| **Depression (MDD vs healthy), resting EEG** | real LaBraM EEG foundation model | **0.961 (0.942–0.976)** |
| Acute stress, wearable physiology | band-power + tuned GBM | 0.955 (0.930–0.978) |
| Cognitive workload (rest vs task) | ECG autonomic; LaBraM improves the EEG path | 0.740 |
| Stress, peripheral physiology | band-power (concat) | 0.892 |

### Novelty & impact

The distinctive contribution is not another fusion architecture — it is turning a rigorously *honest*
benchmark into a **live, evidence-forward screening product**. Three grounded pieces: (1) the real
**LaBraM EEG foundation model runs as a frozen linear-probe screener** and beats hand-crafted
band-power on both EEG cohorts — decisively on depression (0.961, the single best config), a
foundation-model-as-clinical-screener result reproduced under subject-held-out CV; (2) a
**reliability-gated "do-no-harm" late fusion** (`dnh_gated`, Super-Learner provenance, van der Laan
2007) that beats the proposal's own learned cross-modal fusion on 4/6 tasks — reported *with* its
finite-sample caveat, not oversold; (3) an **honesty gate wired into CI**: every product number
resolves to a committed scoreboard, and a blocking audit (`tests/test_honesty_audit.py`) forbids the
weak/negative results (DEAP affect, CACMF-as-win, LLM-as-predictor, mortality, the diabetes leak)
from ever surfacing as a claim. The impact is a reproducible, offline, calibrated EEG-first screening
toolkit whose accuracy claims a reviewer can verify to the row of a CSV — research-grade screening,
never a diagnosis.

## CACMF — the unified multimodal fusion framework (`dvxr`)

The pipeline is now packaged as **`dvxr`** implementing **CACMF** (Cross-modal Aligned
Codebook Multimodal Fusion). `goal1_pipeline` remains importable as thin re-export
shims, so every existing script and test keeps working.

```
raw files ─▶ ingest/validate (13-col canonical schema) ─▶ per-modality features
   │
   ├─ per-modality ENCODER  f_m ─▶ z_m         (dvxr/encoders/*_adapter.py, real weights)
   ├─ VQ CODEBOOK          q_m ─▶ ê_m, code k* (dvxr/encoders/codebook.py)
   ├─ FUSION g  (early|intermediate|late|attention|cross-modal) ─▶ h  (dvxr/fusion/)
   ├─ MULTI-TASK heads + relative losses ─▶ 7 calibrated tasks   (dvxr/tasks/)
   ├─ REAL-TIME fused stream + adaptive intervention            (dvxr/realtime/)
   ├─ EXPLAIN (physio + neural saliency + attention + codebook) (dvxr/explain/)
   └─ LLM INSIGHT (explains, never predicts; offline-safe)      (dvxr/llm/)
```

**Foundation-model weights — what actually loads** (verified on CPU; see
`dvxr.config.FOUNDATION_MODELS`). The config *names* a POW-aligned primary per modality, but
only some primaries have a real loader wired here; the rest degrade to an always-runnable
baseline behind capability checks. The **"Runs here"** column is the ground truth (probe:
`make_primary_backend(modality, cfg)`), not the aspiration:

| Modality | Configured primary | Runs here (CPU, no extra deps) | Notes |
|---|---|---|---|
| Wearable | MOMENT `AutonLab/MOMENT-1-large` | ✅ **MOMENT real weights** | Loaded via `momentfm`; fed summary-stat pseudo-series (ceiling C2). |
| EHR | CEHR-BERT-style (train-local) | ✅ **Bio_ClinicalBERT real weights** | Loaded via `transformers`; fed pseudo-text of summary stats. |
| Omics | Geneformer `ctheodoris/Geneformer` | ✅ **Geneformer real weights** | Loaded via `transformers`. |
| EEG | LaBraM `braindecode/labram-pretrained` | ⚠️ **band-power + VQ baseline** | LaBraM is **not wired** — there is no `braindecode` loader, so `make_primary_backend("eeg")` returns `None`. Wiring it needs `braindecode[hug]` **and** a raw-signal path (LaBraM cannot consume the summary-stat table). |
| CGM | CGM-JEPA `CRUISEResearchGroup/CGM-JEPA` | ⚠️ **conformal Ridge baseline** | CGM-JEPA has no HF-text-loadable weights; the primary returns `None` and the baseline runs. |
| Insight LLM | Anthropic Claude API / Qwen2.5 (local) | template / Qwen | **Insight only — explains, never predicts.** |

So real pretrained weights are live for **wearable, EHR, and omics**; **EEG and CGM run the
baseline** (honestly labeled — not faked). `config.use_real_weights=True` is the intent, but
capability checks mean **the whole pipeline runs with no network and no GPU**. The BCI/EEG
modality — central to the proposal — currently rides the band-power + VQ baseline; a real EEG
foundation model (LaBraM/EEGPT) is future work gated on the raw-signal path (see
`BENCHMARK_FINDINGS.md` finding C2).

### Run CACMF (one command, offline/CPU/deterministic)

```bash
python3 scripts/run_mmf_full.py            # full pipeline -> outputs/
python3 scripts/run_mmf_full.py --profile  # profile data/ -> outputs/data_schema_report.md
python3 scripts/run_mmf_full.py --realtime # fused stream  -> outputs/realtime_fused_stream.csv
python3 scripts/run_mmf_full.py --insight  # LLM insight   -> outputs/insight_example.md (offline)
python3 scripts/run_ablation.py            # Goal-3 ablation-> outputs/ablation_table.csv
make paper                                 # build paper/tables/*.tex (PDF if pdflatex present)
make all                                   # ablation + full run + paper + tests
```

Optional real-weights setup: `pip install "braindecode[hug]"` (LaBraM); set
`ANTHROPIC_API_KEY` + `DVXR_LLM_MODEL` for the live insight layer (keys read from env only,
never logged). Architecture spec: `docs/ARCHITECTURE.md`; guardrails: `docs/MASTER_BRIEF.md`.

## Real-label benchmark — the honest scoreboard

Synthetic-fixture metrics validate plumbing, not science. For the actual evaluation —
real external labels, subject/patient-held-out CV, real baselines, CIs and significance —
run:

```bash
python3 scripts/run_benchmark.py --repeats 5 --folds 5 --ablate
# -> outputs/benchmark_scoreboard.{csv,md}
```

`src/dvxr/bench/` compares the CACMF fused model (encoder + VQ + cross-modal fusion →
shared head) against real baselines (persistence/majority, classical GBM, best single
modality, and real pretrained SOTA encoders — MOMENT, Bio_ClinicalBERT) on three
credential-free real-label tasks, under 5×5 grouped CV with bootstrap CIs, paired
Wilcoxon + Holm, and a **true retrain-without-modality** ablation.

**Result (honest):** on stress (Non-EEG annotations), glucose (Shanghai CGM future
value), and mortality (MIMIC-IV), the learned fusion does **not** beat the strongest
baseline and does not approach a 50% relative-error reduction (RER −20% / −23% / −102%).
What *does* hold up: multimodality beats the best single modality on stress (~35% via
concatenation), and a simple learned model beats glucose persistence (~17%) — both
modest, both significant. Full analysis, root causes, and the audit-by-audit status:
[`BENCHMARK_FINDINGS.md`](BENCHMARK_FINDINGS.md).

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run on synthetic fixtures (no download)

```bash
python3 scripts/run_demo.py
```

Outputs: dataset summary, schema validation, stress classification metrics, glucose
forecasting metrics, top explanations, one streaming-style prediction, plus registries
and calibrated risk bands / prediction intervals in `outputs/`.

## Real collected BCI data → decoding + dashboard (EMOTIV + Galea)

The headline tangible result. Runs the full pipeline on the **collected** EMOTIV
EPOC X (mental commands: Neutral/Left/Right/Push/Pull) and Galea recordings in
`data/*.zip`, producing a self-contained dashboard, figures, and metrics:

```bash
venv/bin/python scripts/run_bci_pipeline.py
# -> outputs/bci/dashboard.html  + PNG figures + metrics.json
```

Decodes intended cube movement from EEG (the avatarRT / MRAE / TPHATE analog):
4-class command direction at **bal-acc 0.82 trial-grouped / 0.72 drift-controlled**
(chance 0.25), with a PHATE neural manifold, leakage-controlled CV, real-time
streaming decode, and explainable channel×band biomarkers. Full writeup:
[`BCI_PIPELINE.md`](BCI_PIPELINE.md).

## End-to-end Goal 1 run (all capabilities)

```bash
python3 scripts/run_goal1_full.py
```

Exercises every Goal 1 capability on synthetic fixtures: multimodal + multi-omics
ingestion, real device/VR-AR converters, neural (torch BIOT-style) vs PCA embeddings, the
seven clinical task heads, real-time stress+glucose streaming, explainable neural +
physiological biomarkers, and per-subject personalization. See
[`GOAL1_COMPLIANCE.md`](GOAL1_COMPLIANCE.md) for the deliverable-by-deliverable map.

The neural encoder needs torch (CPU is fine):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run on real, credential-free public data

```bash
# noneeg + mimic-demo + shanghai-cgm + WESAD (Siegen) + CGMacros (PhysioNet)
python3 scripts/fetch_data.py all-free --subjects 20
python3 scripts/run_real_demo.py
```

| Stage | Real dataset | Typical result |
|---|---|---|
| Wearable stress | PhysioNet Non-EEG (20 subjects) | AUROC ~0.90, ECE ~0.14 (subject-held-out) |
| CGM / diabetes | Shanghai T1DM/T2DM (19 patients) | 30-min MAE ~11 mg/dL, 90% interval coverage ~0.92 |
| EHR ingestion | MIMIC-IV clinical demo (100 patients) | 40k events, 254 lab/demographic concepts |

All real sources download over plain HTTP without accounts. The stress labels come from
the Non-EEG `.atr` phase annotations; CGM is the open Shanghai diabetes dataset; EHR is
the open MIMIC-IV demo subset.

### Additional real datasets (WESAD, CGMacros, DEAP)

Wired in via `scripts/fetch_data.py {wesad,cgmacros,kaggle-deap}` and exposed as real-label
benchmark tasks (`scripts/run_benchmark.py --tasks wesad_stress cgmacros_diabetes cgmacros_glucose`):

| Dataset | Source | Modalities | Real task / label |
|---|---|---|---|
| WESAD | Siegen sciebo (`WESAD.zip`, ~2 GB) | wearable_phys (chest+wrist) | stress vs non-stress (protocol conditions) |
| CGMacros | PhysioNet (open) | cgm (Libre+Dexcom), wearable_phys (Fitbit), behavior (meal macros), ehr (bio labs) | glucose 30-min forecast; diabetes strata (HbA1c) |
| DEAP | Kaggle `sayuksh/deap-datasetraw-data` (needs `~/.kaggle/kaggle.json`) | eeg (32-ch) + peripheral | affect (valence/arousal) |

Honest note: on these real cohorts the CACMF fused model does **not** beat the best
non-fused baseline (see `outputs/benchmark_scoreboard.md`) — reported as-is.

## DEAP EEG/peripheral arousal benchmark

`scripts/run_deap_demo.py` runs the DEAP arousal-classification path: it loads EEG and
peripheral physiology into the canonical schema, builds 30s windows (EEG band-power +
per-channel statistics), encodes them, and trains a calibrated high/low-arousal classifier
with a subject-held-out split.

```bash
# 1. Synthetic DEAP-shaped fixture — always runnable, no download
python3 scripts/run_deap_demo.py

# 2. One official preprocessed subject file
python3 scripts/run_deap_demo.py \
  --deap-pickle /path/to/data_preprocessed_python/s01.dat \
  --max-trials 40

# 3. A directory of subject files (subject-held-out evaluation)
python3 scripts/run_deap_demo.py \
  --deap-dir /path/to/data_preprocessed_python \
  --max-subjects 10 \
  --max-trials 40
```

The preprocessed `.dat` files can be fetched with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("manh123df/deap-dataset")
# subjects land in <path>/deap-dataset/data_preprocessed_python/s01.dat ... s32.dat
```

| Mode | Data | Typical result |
|---|---|---|
| Synthetic fixture | generated in-process | AUROC ~1.0 (clean fixture, validation only) |
| Single subject (`--deap-pickle`) | one DEAP subject, within-subject split | AUROC ~0.92, ECE ~0.13 |
| Directory (`--deap-dir`) | N DEAP subjects, subject-held-out | AUROC near chance — cross-subject arousal does not transfer with a linear baseline |

The single-vs-directory gap is the expected DEAP result: within-subject splits exploit
subject-specific patterns, while whole-subject hold-out demands cross-subject generalization
that a linear model on raw features does not achieve. Per-subject normalization or a stronger
encoder is the next step.

## SOTA model comparison

`scripts/compare_sota_models.py` scores the candidate foundation models per task on
evidence, Goal-1 fit, integration effort, and calibration, and records which model is
selected for the pipeline. `run_demo.py` also emits this report.

```bash
python3 scripts/compare_sota_models.py
```

Writes `outputs/sota_comparison.csv` (all candidates) and `outputs/sota_selection.csv`
(the selected models). Selected per task: EEG-X (EEG/BCI), BIOT (wearable biosignals),
GluFormer with a conformalized Ridge fallback (CGM), Med-BERT/BEHRT (EHR timelines), and
PHIA (LLM insight layer).

## Datasets requiring credentials / access

`scripts/fetch_data.py kaggle-wesad` and `kaggle-deap` use `kagglehub` and need a Kaggle
token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). Convert official files
to the canonical schema with:

```bash
python3 scripts/convert_wesad_subject.py /path/to/WESAD/S2/S2.pkl data/sample/wesad_S2_events.csv
python3 scripts/convert_deap_subject.py  /path/to/DEAP/s01.dat       data/sample/deap_s01_events.csv
```

Real Galea / EMOTIV / VR-AR exports convert into the canonical event schema before modeling
(each accepts `--demo` to run on a synthetic sample now):

```bash
python3 scripts/convert_galea_subject.py  --demo --output outputs/galea_demo.csv
python3 scripts/convert_emotiv_subject.py --demo --device epocx --output outputs/emotiv_demo.csv
python3 scripts/ingest_vr_session.py      --demo --output outputs/vr_demo.csv
python3 scripts/convert_omics_subject.py  --demo --output outputs/omics_demo.csv
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The real-data tests auto-skip when the corresponding dataset has not been downloaded.

## Layout

```
src/goal1_pipeline/   schemas, loaders, features, encoders, models, calibration, registry, sota, explain, streaming,
                      neural_encoders (torch), omics, clinical_tasks, personalization, realtime, biomarkers
scripts/              run_demo.py, run_real_demo.py, run_deap_demo.py, run_goal1_full.py, compare_sota_models.py,
                      fetch_data.py, convert_{wesad,deap,galea,emotiv,omics}_subject.py, ingest_vr_session.py
tests/                test_pipeline_smoke.py, test_sota_selection.py, test_neural_encoders.py, test_omics.py,
                      test_device_converters.py, test_clinical_tasks.py, test_personalization.py, test_realtime.py,
                      test_biomarkers.py, test_real_data.py (real, auto-skipping)
outputs/              committed result artifacts (metrics, predictions, registries, model card, SOTA report); raw event dumps gitignored
GOAL1_COMPLIANCE.md   deliverable-by-deliverable compliance map
```

## Caveats

- Synthetic-fixture metrics are pipeline validation, not scientific evidence (the fixtures
  are intentionally clean, so stress scores near 1.0 are expected).
- Real metrics use subject/patient-held-out splits; personalized claims require per-subject
  calibration that improves held-out performance.
- LLM/agent layers should *explain* model outputs, not replace deterministic signal processing.
