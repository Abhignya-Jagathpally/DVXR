# EHR unstructured-notes pathway (POW Goal 2)

## Why this exists

Goal 2 calls for a "transformer-based clinical language modeling framework capable of
ingesting **structured and unstructured** EHR data." The repo already had the structured
side (MIMIC-IV labs/demographics → `EHRAdapter`), but the **unstructured** side was only
scaffolded: `make_primary_backend("ehr")` routes to a HuggingFace backend that embeds
*pseudo-text synthesized from numeric column name/value pairs* — never real note text.
And there is no clinical note text in the repo: MIMIC-IV's note module is credentialed and
absent (the demo ships only `admissions/labevents/patients/d_labitems`).

This adds a genuine end-to-end path: **real free-text clinical notes → a real frozen
clinical transformer → embeddings → classifier**, benchmarked honestly against classical
text floors.

## What runs

- **Corpus — MTSamples.** ~4,499 genuine de-identified transcribed medical reports
  (mtsamples.com, public domain), via the HF mirror `rungalileo/medical_transcription_40`
  (`text` free-text + a 40-way specialty `label`). Real clinical narrative, **not**
  synthesized from structured fields. Loader: `dvxr.loaders.load_clinical_notes` (fetches +
  caches under `data/real/clinical_notes/`; a 40-row real excerpt is committed at
  `data/real/clinical_notes/sample.parquet` for offline/deterministic tests). This corpus
  is distinct from — and not patient-aligned with — the MIMIC structured `ehr` path.

- **Real clinical transformer — Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`),
  cached locally and CPU-runnable here (the EEG torch/torchaudio blocker does **not** apply
  to a plain BERT). Backend: `_ClinicalNotesBackend` in `dvxr/encoders/base.py`. Because
  clinical reports routinely exceed BERT's 512-token limit, each note is tokenized into
  ≤512-token windows (capped at 4) and the per-window `[CLS]` vectors are **mean-pooled**
  into one note embedding. Modality `ehr_notes`, adapter `NotesEHRAdapter`, foundation-model
  entry `config.FOUNDATION_MODELS["ehr_notes"]`. Override the model via `DVXR_EHR_NOTES_MODEL`.

- **Floor (always-runnable).** `_TfidfSvdBackend` (TF-IDF + TruncatedSVD) and a stateless
  `HashingVectorizer` feature set. These need no torch/transformers/network and are the
  honest baseline the transformer must beat. When transformers/weights are absent the
  adapter degrades to the TF-IDF floor.

## Tasks & benchmark

Two tasks (`dvxr.bench.tasks`), evaluated under note-held-out grouped CV (each transcript is
its own CV group):

1. `clinical_notes_surgery` — binary Surgery-vs-rest. Registered in `TASK_BUILDERS`, so it is
   a first-class binary-AUROC task the standard harness (`dvxr.bench.run.run_task`) can run
   with the full opponent set.
2. `clinical_notes_specialty` — 40-way specialty (multi-class macro-F1 / accuracy /
   macro-AUROC-OvR). The shared harness metric is binary-only, so this task is **not** in
   `TASK_BUILDERS`; it is evaluated by the dedicated script with proper multi-class metrics.

Run: `python scripts/run_clinical_notes_bench.py` → `outputs/clinical_notes_scoreboard.{md,csv}`.

**Honesty.** The frozen transformer's embedding is label-free and computed once over all
rows (no transductive leak); every supervised head and the per-fold TF-IDF vocabulary are
fit on TRAIN indices only. We do **not** assume Bio_ClinicalBERT beats TF-IDF — a frozen
CLS/mean-pool feature extractor often does not out-score a tuned bag-of-words model on
MTSamples, and the scoreboard reports whatever is measured (see `outputs/clinical_notes_scoreboard.md`).

## Measured result (full corpus, 4,499 notes, 5-fold note-held-out CV)

Frozen encoder that actually ran: `clinicalnotes:emilyalsentzer/Bio_ClinicalBERT` (real
weights, not the fallback). Full table in `outputs/clinical_notes_scoreboard.{md,csv}`.

**Task 1 — Surgery vs rest (binary, AUROC ↑):**

| config | AUROC |
|---|---|
| **clinicalbert+lr** | **0.910** |
| tfidf+lr | 0.823 |
| hashing+gbm | 0.789 |
| majority | 0.500 |

**Task 2 — 40-way specialty (multi-class):**

| config | macro-F1 ↑ | accuracy ↑ | macro-AUROC ↑ |
|---|---|---|---|
| tfidf+lr | **0.437** | 0.407 | **0.961** |
| clinicalbert+lr | 0.405 | **0.472** | 0.931 |
| hashing+gbm | 0.035 | 0.114 | 0.500 |
| majority | 0.009 | 0.222 | 0.500 |

Reading it honestly: on the **binary** surgical task the frozen clinical transformer
**clearly helps** — +0.09 AUROC over the TF-IDF floor (0.910 vs 0.823). On **fine-grained
40-way specialty** it is a **split decision**: a tuned bag-of-words (TF-IDF+LR) edges it out
on macro-F1 and macro-AUROC, while ClinicalBERT wins on top-1 accuracy. This is the expected
frozen-feature-extractor picture — the transformer's contextual representation pays off most
on the coarser semantic contrast, less so against a strong lexical model on many narrow
classes.

## Fusion decision ("integrate only if it helps")

MTSamples notes are **not** the same patients as the MIMIC / EEG / wearable / CGM cohorts,
so cross-modal fusion with the other modalities is not data-aligned and is deliberately
**not** wired into the integrated model. The relevant, data-aligned measurement is
within-notes: does the real clinical transformer beat the classical text floor?

The measurement above answers it: **yes on the binary surgical task** (ClinicalBERT 0.910 vs
TF-IDF 0.823 AUROC — a real, sizable gain), **mixed on 40-way specialty** (floor-favored on
macro-F1/AUROC, FM-favored on accuracy). So the honest verdict is that the frozen clinical
transformer *does* add signal where the target is a coarse semantic contrast, and the
`ehr_notes` modality is worth promoting into a fused model **in an aligned setting with that
kind of target**. It is not promoted into the current multimodal model because no note-plus-
other-modality alignment exists here — doing so would be data-dishonest, not helpful. Notes
therefore ship as a standalone, honestly-benchmarked real-text ingestion capability, with the
binary-task gain documented as the evidence that the pathway is worth more than its floor.

## Files

- `src/dvxr/loaders.py` — `load_clinical_notes`, `CLINICAL_NOTES_SURGERY_LABEL`
- `src/dvxr/encoders/base.py` — `_ClinicalNotesBackend`, `_TfidfSvdBackend`,
  `clinical_notes_available`, `make_primary_backend` `clinical_notes` branch
- `src/dvxr/encoders/notes_adapter.py`, `encoders/__init__.py` — `NotesEHRAdapter` / `ADAPTERS`
- `src/dvxr/config.py` — `ehr_notes` `FoundationModel` + `MODALITIES`
- `src/dvxr/bench/tasks.py`, `bench/baselines.py` — task builders, `_FM_FOR_TASK`, sota text routing
- `scripts/run_clinical_notes_bench.py` — scoreboard
- `tests/test_clinical_notes.py` — offline floor tests + guarded Bio_ClinicalBERT tests
