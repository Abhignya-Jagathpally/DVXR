# DVXR — final fixes & presentation assets: orientation (Prompt 0)

Branch `fixes-and-assets`. Guardrails: offline/CPU, deterministic (seed=7), keep the
218-test suite green, one commit per prompt, never fabricate numbers, every reported
number traces to an `outputs/` file, label single-subject/proxy/exploratory results.

## Review findings — confirmed against the code

- **C1 (BCI labels are circular).** The Emotiv sample CSV header exposes `MC.Action`,
  `MC.ActionPower`, `MC.IsActive` (Emotiv's on-device Mental-Command engine) and the
  `POW.*` band-power columns the engine itself consumes. The 4-class labels are derived
  from `MC.Action`, and the "command" decoder is trained on `POW.*` features — i.e. it
  reproduces Emotiv's own engine state, not experimenter-cued neural intent. The honest
  controls confirm chance-level neural separability: engaged-vs-neutral AUROC **0.489**,
  Left-vs-Right lateralization AUROC **0.541** (chance 0.5). CONFIRMED.
- **C2 (single subject / single session).** `metrics.json.emotiv` is one recording
  (subject "AJ", serial redacted-pending, `duration_s` 1373.1, one session). Trial-grouped
  4-class balanced-acc 0.823 collapses to 0.722 under a temporal-block split — the classic
  block-design time confound. CONFIRMED.
- **M1 (not reproducible from the repo).** `scripts/run_bci_pipeline.py` reads `data/*.zip`
  that are **not committed**. `data/sample/emotiv/` holds only a 5.5 KB sample CSV, a
  60-byte **empty** `*_intervalMarker.csv` (header only → no cue onsets), and a JSON. The
  full recording that produced `outputs/bci/` is absent. CONFIRMED.
- **M2 (silent NaN swallowing).** `src/dvxr/bench/run.py` wraps each config/fold in
  `except Exception:` and appends `NaN`, with no logging or failure count. CONFIRMED
  (see FIX_PLAN update after code map).
- **M3 (headline RER has no frozen test).** `run_task` selects the best non-fused opponent
  by mean CV error and computes proposed-vs-selected RER **on the same folds** — selection
  and evaluation overlap. CONFIRMED.
- **M4 (only one task is multimodal).** `stress` uses 4 peripheral-physiology streams from
  one wearable; `glucose` is CGM-only; `mortality` is EHR-only. Multimodal-fusion claims
  rest on `stress` alone; no dataset co-registers EEG+CGM+EHR per subject. CONFIRMED.
- **M5 (transductive SOTA/PCA).** `_sota_embeddings` fits the FM adapter (incl. its PCA/
  scaler) over **all rows** once and caches it; the "computed once … without leakage"
  comment understates that the projection sees test-fold rows. CONFIRMED.

## Data actually available on this machine

- **BCI full recording: ABSENT.** No `.zip`; `data/sample/emotiv|openbci/` are small
  subsets; `data/sample/emotiv/..._intervalMarker.csv` is empty (no cues).
- **BCI processed artifacts: PRESENT & committed.** `outputs/bci/emotiv_windows.csv`
  (2744 windows × band-power/motion/PM features, with `label`/`trial_id`), `metrics.json`,
  PNGs, `emotiv_encoder.pt`, `dashboard.html`. → BCI figures/controls can be regenerated
  from the committed **windows** (derived from the FULL, uncommitted recording — flagged
  in MANIFEST). Raw-EEG Welch relabeling is **not** possible (no raw signal retained).
- **Benchmark real data: PRESENT.** `data/real/{noneeg (4.1M), shanghai_cgm (764K),
  mimic_demo (2M)}` → `benchmark_scoreboard.*` fully reproducible.

## Consequence for scope

- BCI (Prompt B): no cue markers → **reframe** branch. Demote `command_4class`, lead with
  engaged-vs-neutral / lateralization / temporal-block controls, add `labels_source:
  "emotiv_mc_engine"`, relabel as *exploratory, single-subject, single-session*.
- Assets (Prompt E) reuse `benchmark_scoreboard.*` (real, reproducible) and the committed
  BCI windows/metrics (flagged as full-recording-derived). Nothing presents 0.82 as
  success or fusion as a win.

## Ordered plan

**Public-data sourcing (user-authorized).** Where a requirement needs data we lack, source
from reliable public sets. Highest-value use: a **real cued, multi-subject** BCI dataset
(PhysioNet EEG Motor Movement/Imagery — credential-free HTTP via wfdb) to provide a genuine
cue-labeled neural-intent decode with leave-one-subject-out, replacing the single-subject
MC-engine story as the honest BCI headline. Treated as an enhancement: attempted only if it
does not block Prompt E; otherwise deferred with a note, and the honest reframe still ships.

A reproducibility (ingest zip|dir|csv + resolver, `--emotiv/--galea`, data_source stamp,
`test_bci_smoke.py`, provenance docs) → B BCI honesty (reframe + LOBO control from windows)
→ C bench rigor (M2 log+count, M3 nested CV or frozen test, M5 per-fold fit, M4 labeling)
→ D minors if time → **E presentation assets (never skipped; generated from current honest
results even if C/B deep fixes defer)** → F QA + PR. Deferrals recorded in CHANGES.md.
