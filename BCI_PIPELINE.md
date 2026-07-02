# Real-Device BCI Pipeline (Goal 1) — EMOTIV + Galea

End-to-end pipeline on the **collected** recordings in `data/*.zip`. It decodes
intended cube movement from EEG — the wearable-BCI analog of the real-time
neural-manifold avatar decoding in Busch et al. (avatarRT / MRAE / TPHATE) — and
emits figures, metrics, and a self-contained dashboard. Omics is deferred.

## Run

```bash
venv/bin/python scripts/run_bci_pipeline.py
# -> outputs/bci/dashboard.html  (+ PNG figures, metrics.json, windows CSV)
```

~60–70 s on CPU. Needs numpy/pandas/scipy/scikit-learn/matplotlib/torch (all in
`venv`). PHATE is used for the manifold if installed, else a built-in
diffusion-map fallback (`temporal_diffusion_map`).

## Data

| Device | File | Content |
|---|---|---|
| **EMOTIV EPOC X** | `EmotivBCI-AJ_EPOCX_*.zip` | 14-ch EEG @128 Hz, ~23 min; built-in Mental-Command stream (**Neutral/Left/Right/Push/Pull**), Emotiv FFT band-power (`POW.*`), Performance-Metrics (`PM.Stress`…), motion. |
| **Galea / OpenBCI** | `OpenBCISession_*.zip` | 16-ch EEG @250 Hz BrainFlow export, resting (unlabeled). Used for multi-device ingestion + signal-quality + unsupervised manifold. |

Labels come from `MC.Action` / `MC.ActionPower`. Code→label map (from the export
JSON): `1 Neutral, 2 Push, 4 Pull, 32 Left, 64 Right`.

## Stages (`src/goal1_pipeline/bci_real.py` + `scripts/run_bci_pipeline.py`)

1. **Ingest** — read the zips directly (no external paths); parse the EMOTIV title
   row for sampling rates; split EEG / MC / POW / PM / motion streams.
2. **Epoch + label** — 2 s windows, 0.5 s step; each window labeled by the
   dominant active mental command (`ActionPower > 0.05`), else Neutral. A
   `trial_id` groups temporally-contiguous same-label windows for leakage control.
3. **Features** — per-channel relative band power (theta/alpha/betaL/betaH/gamma)
   recomputed by Welch from raw EEG (`welch`), **and** Emotiv's own FFT band power
   (`pow`). Both feature sets are decoded and compared.
4. **Neural latent** — BIOT/MRAE-style masked-feature transformer autoencoder
   (`NeuralBiosignalEncoder`) → 16-d embedding (saved to `emotiv_encoder.pt`).
5. **Manifold** — PHATE (or diffusion-map) over band-power windows; full manifold
   (commands vs faint Neutral) + a command-only manifold.
6. **Decode** — multinomial logistic regression, **leakage-controlled**
   `StratifiedGroupKFold` grouped by `trial_id`:
   - 5-class (incl. Neutral), feature-set comparison.
   - **Two-stage / avatar-control framing**: Stage A engaged-vs-Neutral (AUROC);
     Stage B 4-class command direction among engaged windows.
7. **Streaming decode** — slide the decoder over time → per-command probability
   trace (the "avatar control signal").
8. **Explainability** — mean |decoder weight| as a channel × band biomarker map.
9. **Galea** — signal-quality report (railed-channel detection) + resting manifold.

## Headline results (this recording)

4-class command direction (Left/Right/Push/Pull) decoded under two CV schemes:

| CV scheme | balanced acc | macro-F1 | controls for |
|---|---|---|---|
| Trial-grouped `StratifiedGroupKFold` | **0.82** | 0.82 | window-overlap leakage |
| **Temporal-block (per-class chronological)** | **0.72** | 0.66 | + slow session drift |

Chance = 0.25. Per-class recall (trial-grouped): Left 0.75 / Right 0.80 / Push
0.82 / Pull 0.92. The drift-controlled **0.72 is the rigorous headline** — the
decode survives holding out contiguous time blocks, so it is movement geometry,
not nearby-in-time leakage.

Supporting:
- 5-class incl. Neutral: balanced acc ≈ 0.30, macro-AUROC ≈ 0.78 (chance 0.20).
- Engaged-vs-Neutral: AUROC ≈ 0.49 (≈ chance). This dissociation (direction
  decodes ≫ chance, engagement ≈ chance) shows the decoder reads the **spatial
  pattern distinguishing movement types**, not a global arousal/drift state.
- Left-vs-Right lateralization (single hemisphere-beta contrast): AUROC ≈ 0.54 —
  only weakly above chance, so the direction signal is **not** simple textbook
  contralateral mu/beta. These are Emotiv biofeedback-trained mental commands, so
  the subject learned idiosyncratic strategies that the multivariate decoder reads
  but a single hemisphere contrast does not.

## Goal 2 / Goal 3 — multimodal late fusion + ablation

`scripts/run_fusion_ablation.py` (also run automatically by the main pipeline)
decodes the EMOTIV 4-class command from three streams captured for the *same*
windows/label, then combines per-modality decoders by **confidence-weighted late
fusion** of out-of-fold probabilities (identical leakage-controlled folds across
modalities, so the ablation is apples-to-apples):

| Modality | # feat | balanced acc | macro-F1 |
|---|---|---|---|
| EEG (Welch + Emotiv FFT) | 140 | **0.78** | 0.79 |
| Motion (MOT.* acc/quat/mag) | 20 | 0.60 | 0.58 |
| PM affective (excitement) | 2 | 0.54 | 0.52 |
| **Fusion (conf-weighted)** | 162 | **0.78** | 0.78 |

Chance 0.25. Honest finding: **EEG is the dominant modality**; motion and
affective carry weaker but clearly above-chance signal; confidence-weighted fusion
*matches* the best single modality but does not exceed it here — EEG already
captures the discriminative structure for this task. (Note: in this recording only
Emotiv's *Excitement* performance-metric was active; Stress/Engagement/etc. were
off — another reason the Unity stress task below is needed.)

## Future data collection (Unity stress task)

The current EEG lacks stress labels. Planned Unity + LSL blocked design (~15–20 min):
baseline rest (eyes open/closed) → Stroop + mental-arithmetic stressors (social-
evaluative threat) → recovery, with STAI-6 / SAM self-reports as ground truth.
Unity broadcasts LSL markers recorded alongside EMOTIV (Cortex/Band-Power outlet)
and Galea (BrainFlow→LSL). This yields the stress-labeled data Goal 1's stress/
anxiety heads need.

## Data provenance & reproducibility

**What produced `outputs/bci/`:** a single full EMOTIV EPOC X recording (subject "AJ",
one ~1373 s session) plus a Galea/OpenBCI resting session. **These raw recordings are
NOT committed** (biometric data). Only a small **schema sample** is in the repo:

- `data/sample/emotiv/` — a 5-row EMOTIV CSV + an **empty** `*_intervalMarker.csv`
  (header only → no experimenter cue onsets) + the session JSON.
- `data/sample/openbci/` — small OpenBCI/BrainFlow session subsets.

**What a fresh clone can do:** `ingest_emotiv()` / `ingest_galea()` accept a `.zip`,
a directory, **or** a direct CSV. `scripts/run_bci_pipeline.py` resolves its input in
priority order — `--emotiv/--galea` CLI → `$DVXR_BCI_DATA` → a full `.zip` in `data/` →
the committed sample — and stamps `data_source: "full" | "sample"` and
`labels_source: "emotiv_mc_engine"` into `metrics.json`. On the committed sample it
**exits cleanly** (writes `metrics_sample.json`, never overwrites the full-run
`metrics.json`) because 5 rows cannot be decoded.

**To reproduce the full artifacts**, point the script at the full recording:

```bash
python3 scripts/run_bci_pipeline.py --emotiv /path/to/EmotivBCI-*.zip --galea /path/to/OpenBCISession-*.zip
# or:  DVXR_BCI_DATA=/path/to/recordings  python3 scripts/run_bci_pipeline.py
```

**Honesty note (see BENCHMARK_FINDINGS / FIX_PLAN):** labels come from Emotiv's on-device
Mental-Command engine, not experimenter cues — this is a **single-subject, single-session,
exploratory** pilot that *reproduces the MC-engine command state from raw EEG*, not
validated neural-intent decoding. Do not commit raw biometric recordings without consent.
