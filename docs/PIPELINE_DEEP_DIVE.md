# Pipeline deep-dive — what the DVXR multimodal fLLM actually does

This is the honest, code-grounded answer to "how does the pipeline actually work?" Every claim points
at the file and line that implements it, so you can check it rather than trust it. Where a design choice
is weaker than it could be, that is stated — the same discipline the BLOCKING honesty audit enforces
(`tests/test_honesty_audit.py`).

**One-line orientation.** Real public biosignal data → frozen foundation-model / band-power embeddings →
(optional) VQ tokenization → learned cross-modal fusion **or** a frozen-LLM soft-prompt reader → a
calibrated linear head → risk + interval + decision-curve + explanation. The *validated* product path is
a single-modality frozen-encoder linear probe; the multimodal fusion + LLM paths are real and
demonstrable but **do not win on full-observation accuracy** — their honest edge is graceful degradation
under missing sensors.

---

## 1. What data is the input?

Eight–nine **real, public** cohorts, loaded by `src/dvxr/loaders.py` and assembled into benchmark tasks
by `src/dvxr/bench/tasks.py`:

| Domain | Cohort | Task builder |
|---|---|---|
| EEG (depression) | Mumtaz 2016 MDD resting EEG, 19-ch | `tasks.py` `mumtaz_depression` |
| EEG (workload) | PhysioNet EEG Mental-Arithmetic (Zyma 2019) | `eegmat_workload` |
| EEG (affect) | DEAP — *excluded*, at chance | (blocked from claims) |
| EEG (sleep) | Sleep-EDF | raw-signal benchmark |
| Wearable | WESAD (Schmidt 2018) — ECG/EDA/EMG/Resp/Temp | `wesad_stress_task` (`tasks.py:169`) |
| Wearable | PhysioNet Non-EEG (Birjandtalab 2016) | `stress` |
| CGM | Shanghai T1DM/T2DM, CGMacros | glucose tasks |
| EHR | MIMIC-IV demo | `mortality` — *excluded*, untrustworthy here |

**Omics and VR are synthetic fixtures** — never benchmarked, and firewalled from any claim by
`assert_no_fabrication`. A user can also supply their own device export at serve time; that path is
flagged out-of-distribution (see §12).

**WESAD is the demo's multimodal subject** because it is genuinely *co-registered*: one device records
ECG, EDA, EMG, respiration and temperature **on the same subject at the same time**, so cross-modal
fusion is actually testable there. No cohort co-registers EEG + CGM + EHR per subject (see §6).

---

## 2. Preprocessing, missing values, normalization

**EEG** (`loaders.py:880`, mirrored at `:728`/`:889`/`:980`/`:1041`):
band-pass **0.5–45 Hz** → **average reference** (`set_eeg_reference("average")`, `loaders.py:728`) →
**resample** to a fixed rate (`raw.resample`, `loaders.py:732`; cohorts here are 64 Hz). Peripherals are
re-typed to `misc` so the EEG filter + reference touch EEG channels only (`loaders.py:718`).

**Windowing** (`features.py`): `build_signal_windows` (`:42`) makes fixed-length windows and, for the
band-power representation, computes a **Welch PSD** (`signal.welch`, `features.py:248`) over canonical
bands + summary stats; `build_raw_windows` (`:85`) resamples each channel to a fixed sample count for the
raw-CNN / LaBraM path.

**Missing values:** intra-window gaps are dropped / zero- / median-filled at load; **missing
*modalities*** are handled structurally, not imputed — a *learned absent token* + attention masking (§7,
§11). Median-split label proxies exist for some exploratory tasks and are firewalled by
`assert_no_fabrication`.

**Normalization** — two places, both fit **on training data only** (no leakage):
- Tabular / band-power head: `StandardScaler` fit on the train fold (`screener.py:116`).
- LaBraM raw path: **per-window, per-channel z-score** (`labram_bench.py:64–66`) — scale-invariant,
  unsupervised, matching EEG-FM practice.
There is deliberately **no per-subject normalization** at train time (it would need subject-specific
statistics the deployed model won't have).

---

## 3. Multi-source integration & the join logic

- **Canonical schema:** every loader emits the same 13-column long-format event table (`subject_id`,
  `session_id`, `timestamp`, `channel`, `value`, `label…`). This is a *floor*, not an exact set — loaders
  may add dataset-specific columns.
- **Integration = vertical concat of events; feature integration = horizontal concat of modality blocks**
  (`_concat`, `representations.py:43`, an `hstack` in a fixed modality order).
- **Join key (within a dataset):** `(subject_id, session_id, time-window)`. Windows from the same
  subject/session/time align row-for-row across modalities.
- **No cross-dataset co-registration.** `subject_id`s are dataset-namespaced on purpose so a Mumtaz EEG
  subject can never be silently "joined" to a WESAD subject — they are different people. This is the
  single most important honest limitation: the proposed EEG+CGM+EHR cross-domain fusion has **no
  co-registered cohort to be tested on**, so it is scoped as future work, not a delivered result.
- **Assumptions made:** binary label mappings per task; some timestamps are synthesized for
  summary-stat cohorts; peripheral "signals" are sometimes summary-stat pseudo-series. All are explicit
  in the loaders, none are presented as raw recordings.

---

## 4. Tokenization — what, how, and better options

**Yes, there is a real tokenizer** — a full VQ-VAE codebook, not a hashing trick
(`encoders/codebook.py`, `VectorQuantizer` at `:46`):

- **K = 512** codes for CACMF (`:49`), **64** for the LLM path (`predictor.py:180`); code dim 64.
- **Nearest-neighbour** assignment by squared L2 (`codebook.py:81–86`).
- **Straight-through estimator** so gradients flow to the encoder (`:106`); an optional **Gumbel-softmax**
  soft path (`:99–103`).
- **Commitment loss** β = 0.25 trains the encoder toward its codes (`:96–97`); the codebook itself is an
  **EMA buffer**, not autograd-updated (`_ema_update`, `:124`).
- **Dead-code reinitialization** revives unused codes (`:137`), and **perplexity** monitors codebook
  usage (`:110`).
- It tokenizes **per-modality latent embeddings**, not raw samples.

**Could a better tokenizer help?** Yes, and it's a Phase-2 candidate:
- **FSQ (finite scalar quantization)** — drops the codebook/commitment machinery entirely, tends to
  reach near-100% code utilization with no dead codes. Removes exactly the failure mode
  `_ema_update`/dead-code-reinit exist to patch.
- **Residual VQ (RVQ)** — a stack of quantizers refining the residual; much higher effective capacity at
  the same code dim, standard in modern neural audio codecs.
Either is a bounded, honest experiment (tracked in `docs/IMPROVEMENT_EXPERIMENT.md`), measured against
the committed board — not assumed to win.

---

## 5. Embeddings — how, and is there a better model?

**How generated (all frozen, off-the-shelf where possible):**
- **EEG:** the real **LaBraM** foundation model — temporal-conv patch tokens → transformer → CLS
  embedding, run through a vendored forward (`encoders/labram_real.py`, wired by `labram_bench.py`), no
  braindecode dependency. This is the flagship encoder.
- **Band-power:** Welch PSD over canonical bands + stats (`features.py:248`) — the tuned tabular floor.
- **From-scratch SSL:** `NeuralBiosignalEncoder` (masked-feature reconstruction) and its VQ subclass
  (`codebook.py:204`) — a BIOT-style transformer trained here, used by the fusion/LLM path.
- Other domains use frozen MOMENT (time-series), Bio_ClinicalBERT (text), Geneformer (omics, synthetic).

**Better model possible?** For EEG, recent **EEGPT** and **CBraMod** report stronger cross-subject
transfer than LaBraM on several benchmarks; both are Phase-2 candidates. Caveat: our cohorts are 64 Hz
(≤32 Hz content) vs these models' native 200–256 Hz, so any FM is under-resourced on sampling rate here
— an honest ceiling on what a better encoder can buy.

---

## 6. Latent-space transformations

In order: per-modality **projection** (`Linear d → d_f`, e.g. `strategies.py:134`), optional **VQ
quantization** (§4), **InfoNCE cross-modal alignment** (τ ≈ 0.1) pulling paired-modality latents
together, a learned **absent token** substituted for any missing modality (`strategies.py:43–44`), and
then the fusion transformer (§7). Nothing here silently imputes real data — absence is an explicit,
learned marker.

---

## 7. Attention / self-attention

Three real implementations, all standard scaled-dot-product attention under the hood:

1. **LaBraM encoder** — multi-head self-attention with per-head QK-LayerNorm and LayerScale
   (`encoders/labram_real.py`).
2. **From-scratch encoders** — stock `nn.TransformerEncoderLayer` (`codebook.py:167`).
3. **Cross-modal fusion** (`fusion/strategies.py`):
   - `CrossModalFusion` (`:131`) — a `TransformerEncoder` (`config.n_heads`, `config.n_fusion_layers`;
     defaults 8 heads / 4 layers) over a **CLS token + one token per modality**, with a
     **key-padding mask** that ignores absent modalities (`:158`), CLS readout (`:161`), and an
     **attention-pool α exported per modality for explainability** (`:167`). This is the tensor the
     glass-box demo visualizes.
   - `AttentionFusion` (`:111`) — a lighter additive (Bahdanau-style) attention (`:125–128`).

---

## 8. LLM framework & the soft-prompt path

`src/dvxr/llm/predictor.py`, `SoftPromptReader` (`:49`):

- **Frozen** `Qwen2.5-0.5B-Instruct` by default (`DVXR_LLM_PREDICTOR` overrides), all params
  `requires_grad_(False)` (`:76–77`), CPU-runnable and deterministic.
- Each modality's **VQ vectors → soft-prompt tokens** in the LLM's embedding space: a **seeded, frozen
  random projection** by default (`_project`, `:89`), or an optional **in-distribution** mode that makes
  each soft token a convex combination of real token embeddings (`:79–87`, `:95–105`). A **missing
  modality → a learned absent token** (`_absent_token`, `:107`).
- Soft tokens are **prepended** to the embedded text prompt (`SOFT_PROMPT_PREFIX`, `:35`; concat at
  `:148`); one forward produces hidden states, which are **mean-pooled** into the multimodal feature
  (`:153`). A calibrated head predicts from that pooled vector (`rep_llm`, `:205`).
- **Separation of concerns:** the head originates the number; `llm/insight.py` only *narrates* it. The
  LLM is never the source of the calibrated probability.

**Honest status:** as a *predictor* this is the **weakest** configuration on full-observation accuracy,
so it is not a product claim. Its validated roles are (a) explanation-only narration and (b) a
missing-modality-robust reader (§10, §11).

---

## 9. Fine-tuning & inference — how they're implemented, and tweaks

**"Fine-tuning" here is a frozen-encoder linear probe — no gradient fine-tuning of the foundation
models.** In `serve/screener.py`:
- `_fit_head` (`:112`) = `StandardScaler` + **balanced `LogisticRegression`** (`max_iter=1000`) on the
  frozen embeddings.
- `fit_screener` (`:244`) runs **3×5 subject-held-out CV** (`repeated_group_folds`, `:262`) to produce
  out-of-fold probabilities, fits a **Platt calibrator** and a **conformal radius** on those OOF
  predictions (no test leakage into the head), computes a **decision curve**, then **refits the
  deployable head on all windows** (`:314`).
- The **VQ codebook and `NeuralBiosignalEncoder` *are* trained** (Adam, masked-reconstruction +
  commitment; `codebook.py:256`); **Qwen stays frozen**. A trainable-projection + **LoRA** variant of the
  LLM path is *documented but deliberately not run on CPU* (`predictor.py:16–18`).

**Inference** (`screener.py`): `predict_windows` (`:182`) = scaler → logistic → Platt calibrator (→
optional per-subject recalibrator); `score_subject` (`:194`) = **mean-pool the window probabilities** →
risk band + conformal interval. Deterministic, offline, CPU.

**Tweaks worth trying (bounded, honest):**
- **Window pooling:** plain mean → attention-pool or logit-mean (mean pooling can wash out a short
  high-risk segment).
- **Calibration:** Platt → temperature scaling or isotonic (Platt is robust at small n; isotonic needs
  more data).
- **Conformal:** single global radius → class-conditional / Mondrian conformal for asymmetric risk.
- **Probe:** logistic → a tiny MLP or a **LoRA** probe on LaBraM — with an explicit overfitting caveat
  at n ≈ 58 (likely to *hurt* held-out AUROC; must be measured, not assumed).

---

## 10. KV cache — is there one, and can it be optimized?

**There is no KV cache anywhere.** The soft-prompt reader runs **one non-autoregressive forward per
window** (`self._model(inputs_embeds=…, attention_mask=…)`, `predictor.py:151`) and mean-pools the hidden
states — no `.generate()`, no `past_key_values`, no `use_cache`. A KV cache accelerates *autoregressive
token generation*; a single encode-style forward has nothing to cache across steps, so it would buy
essentially nothing here.

**The real latency levers** (recent-literature, ordered by expected payoff for this workload):
1. **Batching** — already batched at `predictor.py:143`; larger batches amortize the fixed cost.
2. **Sequence-length reduction** — soft tokens + a short prompt; trimming the prompt or pruning soft
   tokens shrinks the quadratic attention cost directly.
3. **Weight quantization** (int8/4-bit) of the frozen LLM — smaller/faster with negligible accuracy cost
   for a frozen encoder.
4. **Module-level model cache** — one loaded reader per model id (`_READERS`, `:158`) + per-task
   embedding cache (`task.extra`, `:194`) already avoid reloading and recomputation.
A KV cache only becomes relevant if the LLM is later used to *generate* explanation text token-by-token
(today `llm/insight.py` narration is short and templated) — then standard `use_cache=True` applies.

---

## 11. Missing-modality robustness (the interoperability story)

The proposed pipeline handles arbitrary present-modality subsets at test time:
- **Fusion:** a learned per-modality absent token (`strategies.py:43`) + key-padding mask
  (`strategies.py:158`) — absent modalities are ignored by attention, never imputed.
- **LLM:** a learned absent soft token (`predictor.py:107`); `missing_modality_robustness` (`:224`)
  trains the head once on full-modality embeddings and measures how error rises as each modality is
  dropped **at test time only**.
- **Training-time modality dropout** for robustness (`representations.py:98`).
This is something a single-modality model structurally *cannot* do — and it is the basis of the honest
"how is it better" result in §12.

---

## 12. How it compares to SOTA — and how it's actually better

**The honest full-observation result:** the learned cross-modal (CACMF) fusion **loses on all 6 real
tasks** (RER −20% … −186%), and the LLM-as-predictor is the weakest config. The one validated win is a
**single-modality** frozen-LaBraM depression screener — window AUROC **0.961**, subject **0.986**
(n = 58) — a foundation model, not fusion, not an LLM. This is stated, not hidden; the audit *blocks*
selling fusion or the LLM as a win.

**Sensor dropout — graceful degradation, but *not* a win (measured).** The POW's real regime is
*streaming*, where sensors drop in and out; the CACMF fusion (absent tokens + masked attention) and the
soft-prompt LLM degrade gracefully (no imputation), while the tuned floor must impute a fixed-width
vector. We measured this honestly (`scripts/run_streaming_showdown.py`,
`outputs/streaming_showdown_wesad_stress.{json,md}`): as modalities drop 0→6 on WESAD, the proposed
model's gap to the floor **narrows** (RER −58% → −15.5%) — real graceful degradation — but the tuned
floor **still leads at every level**, so there is **no CI-backed crossover**. A win is only claimed where
the bootstrap CI excludes a tie; here none survives, and the curve says so. So the honest phrasing is
*degrades more gracefully*, **not** *wins under dropout*. (The `streaming_eval.py` docstring's more
optimistic "the story flips" was not borne out by the measurement — reported, not hidden.)

**Why it's better *for users* (all real, all demoable):**
- **Calibrated** probabilities (Platt) with **ECE** reported — a 0.7 means 0.7.
- **Conformal interval** — an honest uncertainty band, not a bare point estimate.
- **Decision-curve / net benefit** (Vickers & Elkin 2006) — whether *acting* on the screen helps, with a
  bootstrap-gated "useful" verdict.
- **Explainability** — per-modality attention (§7) + modality attribution (`predictor.py:212`) + grounded
  narration (`llm/insight.py`).
- **Missing-modality robustness** (§11) and **pluggability** (fit → save → load → predict, `screener.py`).
- **Not a diagnosis** — every surface carries the research-prototype caveat, enforced by the audit.

**Do we need LSL (Lab Streaming Layer)?** **No — not for this product.** LSL appears only in *planning*
docs (`BCI_PIPELINE.md:113`, `PAPER_DRAFT.md:175`) as a *blocked, future* Unity + Galea + EMOTIV live
stress-labeling protocol; nothing in the pipeline imports it. Screening runs offline on recorded /
exported signals, and `dvxr.realtime/` already simulates the streaming regime with rolling buffers
(`realtime/monitor.py`) — no LSL. LSL (or BrainFlow) becomes necessary *only* if you later want **live
headset acquisition**; the plug-and-play (file / sample-entry) demo does not require it.

---

## 13. Optimization roadmap — better pathways to the proposed model

Consolidated menu of literature-grounded improvements, each tagged by how much we'd trust it before
measuring. Phase-2 picks the single most promising one and measures it honestly against the committed
board **and** the dropout-crossover curve (`docs/IMPROVEMENT_EXPERIMENT.md`).

| Lever | Change | Expectation |
|---|---|---|
| **Encoder** | LaBraM → **EEGPT / CBraMod** | *honest-effort* — plausibly better cross-subject transfer; capped by 64 Hz data |
| **Tokenizer** | VQ → **FSQ / residual-VQ** | *honest-effort* — SimVQ was tested and **lost** (see below); FSQ/RVQ untried |
| **LLM path** | frozen-random projection → **learned + LoRA** soft prompts | *honest-effort on GPU* — the documented full Option-3; overfit risk at small n |
| **KV / latency** | int8/4-bit quantization, seq-len pruning, bigger batches | *safe* — speed with negligible accuracy cost |
| **Inference** | mean-pool → attention-pool; Platt → temperature/isotonic; global → Mondrian conformal | *safe-to-modest* — better use of the same embeddings |
| **Regime** | optimize the **dropout curve** (narrow the gap toward a real crossover), not full-obs accuracy | *strategic* — the gap narrows under dropout but no CI-backed crossover yet |

**Tested (Phase 2 — an honest negative).** The glass-box surfaced low VQ codebook utilization (perplexity
2.5–5.7 / 64), so we pre-registered and measured **SimVQ** (one-linear-layer codebook reparameterization).
It **underperformed** the existing EMA + dead-code VQ on every WESAD modality and across 8–120 epochs —
recorded in full in [`docs/IMPROVEMENT_EXPERIMENT.md`](IMPROVEMENT_EXPERIMENT.md), shipped as an
off-by-default flag (`DVXR_VQ=simvq`) so the negative stays visible. The one real, cheap lever the
experiment did surface: the low utilization is the LLM path's short training (`epochs=8`), not a VQ
ceiling (the same VQ reaches ~24 perplexity at 120 epochs) — a future, separately-benchmarked tweak.

**Bottom line:** the proposed multimodal fLLM is real, fully wired, and demonstrable end-to-end; its
honest value is calibrated, uncertainty-aware, explainable, missing-modality-robust screening — with a
measurably more graceful degradation under sensor dropout (the gap narrows, though the floor still
leads — no CI-backed win) — not a full-observation leaderboard win. The
glass-box demo (`dvxr glassbox`) shows every stage above running side-by-side with the winning
single-modality model, on real co-registered data, with the numbers traced to committed scoreboards.
