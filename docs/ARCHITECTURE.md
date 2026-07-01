# CACMF Architecture & Math Spec

CACMF = **Cross-modal Aligned Codebook Multimodal Fusion**. This is the implementation
spec for `src/dvxr/`. Every constant lives in `dvxr.config.CACMFConfig` (§A7), never
hard-coded. Sections §A2–A7 are implemented across Stages 2–11 of `docs/REFRACTOR_PLAN.md`.

## A1. Pipeline overview (multi-stage)

```
 raw files ─▶ ingest ─▶ validate ─▶ per-modality windowing ─▶ per-modality features
 (Galea,                (13-col canonical event table)        (band-power, HRV/GSR/resp/
  EMOTIV,                                                       motion, CGM dynamics, EHR
  CGM, EHR,                                                     code/notes, omics)
  wearables)                          │
        Stage 1: per-modality ENCODER  f_m(·)  ─▶ continuous latent z_m ∈ R^d
        Stage 2: VQ CODEBOOK           q_m(·)  ─▶ discrete token k* + quantized ê_m ∈ R^d
        Stage 3: FUSION                g(·)    ─▶ joint latent h ∈ R^{d_f}
                 (early | intermediate | late | attention | cross-modal transformer)
        Stage 4: MULTI-TASK HEADS      ─▶ 7 calibrated task outputs (softmax/logistic + forecast)
        Stage 5: PERSONALIZATION       (per-subject normalize + calibrator)
        Stage 6: REAL-TIME loop + ADAPTIVE INTERVENTION
        Stage 7: EXPLAIN               (physio biomarkers + neural saliency + attention + codebook)
        Stage 8: LLM INSIGHT           (narrate / retrieve / recommend; optional, offline-safe)
```

Modality set `M = {eeg, wearable_phys, cgm, ehr, omics, behavior}`. Any subset may be
present per subject/session; the architecture handles arbitrary missing modalities
(mask them; never impute silently into a prediction without flagging).

## A2. Per-modality encoders f_m → latent z_m

Uniform interface `encode(frame, columns) -> DataFrame[z_00..z_{d-1}]` (mirrors the
existing `NeuralBiosignalEncoder`/`FeatureEncoder` API so adapters drop in):

- **eeg** `EEGAdapter` — primary LaBraM (`braindecode/labram-pretrained`) via `from_pretrained`; fallback = windowed band-power + per-channel stats → `VQBiosignalEncoder`. Handles Galea vs EMOTIV variable channel counts.
- **wearable_phys** `BiosignalAdapter` — primary MOMENT (`AutonLab/MOMENT-1-large`) / BIOT; fallback = `NeuralBiosignalEncoder` or PCA.
- **cgm** `CGMAdapter` — primary MOMENT/Chronos (GluFormer unreleased); fallback = conformalized Ridge forecaster + latent summary (mean, CV, MAGE, time-in-range, slope).
- **ehr** `EHRAdapter` — Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) note/code embedding; fallback = tokenized-code timeline features.
- **omics** `OmicsAdapter` — Geneformer (`ctheodoris/Geneformer`) or `build_omics_features` → linear proj to d.
- **behavior** `BehaviorAdapter` — VR/AR behavior features (or MOMENT on behavioral time-series) → linear proj to d.

Every "try foundation model" path is a **capability check**: import-guarded + weight-path
guarded, and logs which encoder actually ran. No adapter hard-requires network or GPU.

## A3. VQ codebook tokenization q_m ("codebook mappings and vectors")

Per modality a learnable codebook `C_m = {e_{m,1}, …, e_{m,K}} ⊂ R^d`, `K = codebook_size`
(default 512). Mirrors LaBraM's VQ neural tokenizer / VQ-VAE.

- **Quantization:** `k* = argmin_k ‖ z_m − e_{m,k} ‖₂`, quantized `ê_m = e_{m,k*}`.
- **Straight-through estimator:** forward uses `ê_m`; backward `ê_m = z_m + stop_grad(ê_m − z_m)`.
- **VQ loss:** `L_vq(m) = ‖ stop_grad(z_m) − ê_m ‖₂² + β · ‖ z_m − stop_grad(ê_m) ‖₂²`, `β = commitment_beta = 0.25`.
- **EMA** codebook updates + **dead-code reinit** for stability. Optional **Gumbel-softmax** soft-assignment `p(k) = softmax(−‖z−e_k‖²/τ)` (`config.gumbel`).
- **Perplexity** (usage / collapse monitor): `exp(−Σ_k p̄_k log p̄_k)`, `p̄_k` = batch usage freq.
- **Reconstruction** `L_recon`: predict masked EEG/phys spectrum or standardized features from `ê_m`.

## A4. Fusion g(·) → joint latent h (implement ALL five; config flag)

With present-modality quantized latents `{ê_m : m ∈ present}`:

1. **Early** — concat aligned per-window vectors across modalities before a single encoder.
2. **Intermediate** — `h = MLP(concat_m ê_m)` with a learned missing-modality mask token per absent modality.
3. **Late (weighted)** — per-modality heads give `p_m`; `p = Σ_m w_m p_m`, `w_m = softmax(θ)_m`. Required baseline.
4. **Attention** — modality tokens + type embeddings; `α_m = softmax(a·tanh(W ê_m))`, `h = Σ_m α_m ê_m`. Export `α_m`.
5. **Cross-modal transformer (CACMF core)** — sequence `[CLS, ê_eeg, ê_wear, ê_cgm, ê_ehr, ê_omics, ê_behav]` + modality-type + temporal position embeddings; `n_fusion_layers` encoder layers, multi-head cross-attention; absent modalities masked. Readout `h = hidden[CLS] ∈ R^{d_f}`.

Three aggregation baselines (operate on head outputs):
- **weighted_late** (as 3); **ensemble_avg** `p = mean_m p_m`; **confidence_weighted** `p = Σ_m c_m p_m / Σ_m c_m`, `c_m = 1 − H(p_m)/log(2)` (normalized entropy).

## A5. Multi-task heads + softmax + latent vectors

- **Classification** (stress, anxiety, depression risk, cognitive workload, diabetes complication, clinical risk): linear head → `softmax(W h + b)`; temperature scaling + Platt (`calibration.fit_platt_calibrator`); emit `risk_band`.
- **Forecasting** (glucose): regression head + split-conformal interval (`conformal_radius`, `interval_coverage`).
- **Exported latents** for paper/explainability/LLM: `z_m`, `ê_m` + `k*`, joint `h`, attention `α_m`, fusion weights `w_m` → `outputs/latent_*.csv/.npy`.

## A6. Relative losses

```
L_total = Σ_t λ_t · L_task_t                 # 6 classification (CE) + 1 forecasting (Huber)
        + λ_vq  · Σ_m L_vq(m)                # codebook + commitment
        + λ_rec · Σ_m L_recon(m)             # masked reconstruction (self-supervision)
        + λ_alg · L_align                    # cross-modal contrastive alignment (InfoNCE, τ_a)
```

- Class-weighted cross-entropy for imbalance; `L_align` (InfoNCE) pulls same-subject/window
  latents of different modalities together.
- Defaults: `λ_task=1.0, λ_vq=1.0, λ_rec=0.5, λ_alg=0.1`.
- Optional **Kendall uncertainty weighting**: learn per-task `σ_t`, `L = Σ_t (1/(2σ_t²))L_task_t + log σ_t` (`config.uncertainty_weighting`); report learned `σ_t`.

## A7. Optimizers, schedules, hyperparameters (all in `CACMFConfig`)

- **Optimizer:** AdamW, `lr=1e-3` (encoders/codebook), `lr=5e-4` (fusion+heads), `weight_decay=1e-2`, `betas=(0.9,0.999)`.
- **Schedule:** linear warmup (`warmup_frac≈0.08`) → cosine decay. Grad clip `max_norm=1.0`. Optional weight EMA.
- **Core:** `d=64, d_f=128, K=512, β=0.25, n_fusion_layers=4, n_heads=8, dropout=0.1, window_seconds=30, window_step=30, mask_ratio=0.3, epochs=30, batch_size=64, τ=1.0, gumbel=False, uncertainty_weighting=False, fusion_strategy="cross_modal", aggregation="confidence_weighted", seed=7`.
- **Personalization:** per-subject standardization (`per_subject_normalize`) + a per-subject `PersonalizedCalibrator`; report both population and personalized held-out metrics.

## A8. Package layout

```
src/dvxr/
  config.py            # CACMFConfig + YAML/JSON load/save; all constants; FOUNDATION_MODELS
  ingest/              # profile_data_dir, canonical mapping, loaders re-export
  encoders/            # baseline (moved), codebook.py (VQ), *_adapter.py, base.py (EncoderProtocol)
  fusion/              # strategies.py, aggregate.py, model.py (CACMFModel)
  tasks/               # heads.py, losses.py, train.py
  realtime/            # base (moved RealtimeMonitor), monitor.py (Fused), intervention.py
  explain/             # linear (moved), attention_maps.py, codebook_usage.py, report.py
  llm/                 # client.py, insight.py, prompts/
  eval/                # splits.py, ablation.py, metrics.py
  <flat legacy modules: schemas, loaders, features, models, calibration, clinical_tasks,
   neural_encoders, personalization, biomarkers, omics, registry, sota, reporting,
   sample_data, streaming, bci_real>
src/goal1_pipeline/    # thin re-export shims → dvxr (backward compat)
scripts/               # run_mmf_full.py, run_ablation.py, build_paper_tables.py
paper/                 # IEEE LaTeX scaffold (Goal 4)
docs/                  # ARCHITECTURE.md (this file), MASTER_BRIEF.md, REFRACTOR_PLAN.md
configs/default.yaml
```
