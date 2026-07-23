# Literature Review: Model Selection for the DVXR Multimodal Clinical-Risk Framework

_Compiled 2026-07-23. Scope: an unbiased, skeptical survey to pick the best model(s) per modality for DVXR (EEG + CGM/glucose + clinical notes + behavioral time-series), with an explicit distinction between segment-level/leaky benchmarks and subject/patient-held-out ones._

> **Selection-integrity statement (read first).** This review was written to *challenge* the framework's current choices, not defend them. Where a challenger is genuinely better I say so; where the incumbent is already competitive I say that too; and where the published evidence is non-comparable (different montages, sampling rates, splits) or likely leaky, I refuse to manufacture a ranking and instead flag it. Several headline numbers in this literature are inflated by subject-identity leakage or segment-level splits — those are called out inline. arXiv IDs beginning `26xx`/`25xx` are 2025–2026 preprints.

---

## 0. TL;DR recommendations

| Modality | Current DVXR choice | Verdict | Recommended action |
|---|---|---|---|
| EEG | LaBraM (arXiv:2405.18765) | **Keep** (competitive, MIT, CPU-friendly) | Optionally A/B against **CBraMod**; treat all resting-state depression AUROCs with suspicion (Identity Trap) |
| Glucose / CGM short-horizon (30–120 min) | Causal CGM features + small MoE; GBM ties/beats | **Keep GBM as the point-forecast baseline** | Do *not* adopt a CGM foundation model for RMSE; the evidence says trees/linear win at this horizon and patient-disjoint split |
| Clinical notes | Bio_ClinicalBERT | **Keep for now; low-cost upgrade available** | Swap to **Clinical ModernBERT** if long notes or speed matter (drop-in encoder, small gain) |
| Behavioral / wearable time-series | MOMENT-1-large (won't build) | **Drop as a hard dependency** | Use a GBM/PatchTST baseline; MOMENT/Chronos give no reliable win over trees here and MOMENT is 385M params |
| Fusion | (availability-aware) | **Keep availability-aware late fusion** | Contrastive alignment is worth a *research* try only; no robust production win yet |

The single most important finding for this project: **for both CGM short-horizon forecasting and cross-subject wearable stress, strong classical baselines (gradient boosting / linear) are not obviously beaten by any foundation model under honest patient/subject-disjoint evaluation.** Do not switch away from them on the strength of headline numbers.

---

## 1. EEG foundation models

### 1.1 Comparison table

| Model | Pretrain corpus | Params | Assumes | License | Best-at (subject-disjoint) | CPU? | Notes |
|---|---|---|---|---|---|---|---|
| **LaBraM** (arXiv:2405.18765, ICLR'24 spotlight) | ~2,500 h (TUEG-derived, multi-dataset) | 5.8M / 46M / 369M | 200 Hz, 200-sample patches, µV | **MIT** | Strong all-rounder; **wins cross-subject cognitive-load with linear probe** (arXiv:2601.21965) | Yes (Base 5.8M, ~0.48 GFLOP) | The de-facto reference FM; robust, well-supported (torcheeg/braindecode) |
| **CBraMod** (arXiv:2412.07236, ICLR'25) | ~9,000 h TUEG | ~4.0M | 200 Hz, 19–22 ch 10-20, patch=200 | **MIT** | SOTA on 10+ tasks; beats LaBraM on PhysioNet-MI (κ 0.522 vs 0.491); Mumtaz depression BAcc **0.956** | Yes (smallest, ~0.32 GFLOP) | Criss-cross (separable spatial/temporal) attention; strongest single peer to LaBraM |
| **BIOT** (Yang et al., NeurIPS'23, arXiv:2305.10351) | Mixed biosignals | ~3M | Tokenizes heterogeneous montages | MIT | Handles variable channels/rates | Yes | Older; generally below LaBraM/CBraMod now |
| **EEGPT** (Wang et al., NeurIPS'24) | Multi-dataset | ~10M (large ~101M) | Dual SSL, frozen-backbone linear probes | Open (repo) | Competitive; strong frozen-probe transfer | Yes | Good "frozen encoder" option; not clearly > CBraMod |
| **Brant / Brant-2** (arXiv:2402.10251) | ~4 TB SEEG+EEG, 15k subj | up to **1B** | Intracranial-leaning; SEEG heavy | Research | Intracranial / clinical long-recording tasks | **No (heavy)** | Corpus is SEEG-dominant → montage mismatch for scalp wearable EEG |
| **NeuroLM** (arXiv:2409.00101, ICLR'25) | ~25,000 h | up to **1.7B** | LLM-as-EEG, instruction tuning | Open (repo) | Multi-task via one model | **No (heavy)** | Impressive breadth, but 1.7B is out of scope for CPU/small compute |
| **CSBrain** (arXiv:2506.23075) | TUEG-class | mid | Cross-scale spatiotemporal | Open | **Best reported Mumtaz depression BAcc 0.9643, MentalArithmetic BAcc 0.7558** | Marginal | 2025 newcomer; strong numbers but single-paper, verify independently |
| **REVE** (NeurIPS'25) | **25,000 subjects** | large | "Any setup" adapts to montage | Open | Best avg rank under cross-subject-prioritized protocol in some benches | Marginal | Montage-agnostic — attractive if channel counts vary |
| **BrainWave / FEMBA / Uni-NTFM / LUNA** (2025) | various | various | various | mixed | Top-3 in some benches (FEMBA) | mixed | Fast-moving; none has displaced LaBraM/CBraMod as the *practical* default |

Independent benchmarks: **EEG-FM-Bench** (arXiv:2508.17742), **OmniEEG-Bench** (arXiv:2606.00815), **Brain4FMs** (arXiv:2602.11558), and the EEG-FM survey (arXiv:2601.17883).

### 1.2 What the honest benchmarks actually say

- **No single EEG FM dominates.** EEG-FM-Bench (arXiv:2508.17742) explicitly concludes there is *no universal winner*; ranking flips by task and by whether the split is segment-level or subject-disjoint. Under full fine-tuning the top-3 average ranks were **CBraMod (4.51), LaBraM (4.88), FEMBA (5.42)**. Under a cross-subject-prioritized protocol, BrainOmni/CBraMod/REVE lead. LaBraM and CBraMod are consistently in the top cluster.
- **The Identity Trap (arXiv:2606.06647) is the key caveat for DVXR's depression use-case.** High accuracy on *resting-state* clinical EEG under subject-disjoint CV can reflect **subject-identity features that happen to correlate with the label in that cohort**, not a real biomarker. The audit found the subject-variance fraction in frozen representations was **13–89× a random null in 12/12 model×dataset pairs** (LaBraM, CBraMod, REVE) and *rose* under fine-tuning. Companion work (arXiv:2606.09189, "Pretrained, Frozen, Still Leaking") reaches the same conclusion. **Implication:** DVXR's headline depression AUROC (0.961) should be treated as an *upper bound possibly contaminated by identity leakage* unless validated on a held-out cohort with within-subject label variation.
- **FMs do not reliably beat simple baselines under fair cross-subject eval.** "Are Large Brainwave Foundation Models Capable Yet?" (arXiv:2507.01196) finds that under rigorous cross-subject protocols, FMs often fail to show a clear advantage over classical/compact nets (EEGNet, EEGConformer). This matches DVXR's own honest negative results elsewhere.
- **Montage / sampling-rate mismatch is real.** LaBraM and CBraMod both assume **200 Hz** and 10-20-style montages. Brant-2 is SEEG-heavy; NeuroLM/Brant-2 are too large for CPU. For a wearable/low-channel deployment, montage-agnostic models (**REVE**, CBraMod's conditional positional encoding) reduce adaptation pain.

### 1.3 EEG recommendation

**Keep LaBraM.** It is MIT-licensed, CPU-runnable at Base size (5.8M params), well-integrated (torcheeg/braindecode), and sits in the top cluster on every independent benchmark. There is **no evidence that switching to a heavier model (Brant-2, NeuroLM) is worth it** for CPU/small-compute deployment.

**Worth a cheap A/B: CBraMod.** It is smaller (~4M), MIT-licensed, 200 Hz (drop-in with LaBraM's preprocessing), and edges LaBraM on several subject-disjoint tasks including Mumtaz depression (BAcc 0.956). This is a low-effort experiment with a plausible small gain. **CSBrain/REVE** report even higher numbers but are single-paper (2025) — verify before adopting.

**Do this regardless of model choice:** run the FMSCOPE-style identity-leakage check (arXiv:2606.06647) on the depression pipeline. If the label is resting-state and subject-disjoint, the current AUROC may be inflated. This matters more than the LaBraM-vs-CBraMod choice.

---

## 2. Time-series / biosignal transformers & foundation models (incl. CGM)

### 2.1 Comparison table

| Model | Type | License | Params | CPU? | Evidence vs strong baseline | Notes |
|---|---|---|---|---|---|---|
| **MOMENT-1-large** (arXiv:2402.03885) | TS FM (T5) | MIT | 385M | Heavy | Beats other *FMs* on classification; specialized models still win; **no CGM RMSE win shown** | DVXR's current pick; won't build in env and is 385M — poor fit |
| **Chronos / Chronos-Bolt** (arXiv:2403.07815) | Probabilistic TS FM (T5 tokens) | Apache-2.0 | Tiny→Large | Bolt-Tiny yes | Best FM zero-shot in some benches; **but "beats baseline" claims are on generic forecasting, not patient-disjoint CGM** | Bolt variants are fast |
| **TimesFM** (arXiv:2310.10688) | Decoder TS FM | Apache-2.0 | 200M–500M | Marginal | Strong zero-shot forecasting | Univariate focus |
| **Moirai / Moirai-MoE** (arXiv:2402.02592) | Any-variate TS FM | Apache-2.0 | S/B/L | Small yes | Strong on LOTSA-style benches | Good multivariate story |
| **PatchTST** (arXiv:2211.14283) | Supervised transformer | Open | small | **Yes** | Strong, cheap, trainable per-task | Excellent non-FM baseline |
| **GPT4TS / "One-Fits-All"** (arXiv:2302.11939) | Frozen-LLM TS | Open | GPT2-size | Marginal | Competitive but not clearly > PatchTST | — |
| **GluFormer** (arXiv:2408.11876; Nature 2025, s41586-025-09925-9) | CGM FM (generative) | Open (repo) | transformer | Marginal | **Wins long-term risk stratification (diabetes/CVD up to 12 yr), NOT short-horizon RMSE** | Trained on 10M+ readings, 10.8k adults; transfers across 19 cohorts |
| **Gluformer** (arXiv:2209.04526, distinct) | CGM transformer + UQ | Open | small | Yes | Best *log-likelihood/UQ*, not clearly best point RMSE | Uncertainty quantification, not a headline RMSE win |
| **GlucoBench baselines** (arXiv:2410.05780) | benchmark | Open | — | — | **ARIMA/linear best on small datasets; deep (Latent-ODE/Transformer) only wins on the largest** | The honest CGM verdict |
| **GBM (XGBoost/LightGBM/CatBoost)** | tree | Open | tiny | **Yes** | **Competitive-to-best on OhioT1DM 30–120 min, far cheaper** | DVXR's current effective baseline |

Benchmarks: **GlucoBench** (arXiv:2410.05780), **MetaboNet-Bench** (arXiv:2606.18640), **GluMind** (arXiv:2509.18457).

### 2.2 Honest read on "is a foundation model better than gradient boosting for 30–120 min CGM?"

**No — not under patient-disjoint evaluation at this horizon.** Multiple independent lines converge:

- **GlucoBench (arXiv:2410.05780):** simpler models (ARIMA, linear regression) were *best on smaller CGM datasets*; deep models only pulled ahead on the largest. Transformers helped calibration/UQ, not necessarily point RMSE.
- **OhioT1DM comparative studies (2025, ResearchSquare rs-7410777; Nature Sci Rep s41598-025-32373-4):** XGBoost/CatBoost/LightGBM are *competitive with LSTM/Bi-LSTM* while being far cheaper. Under Leave-One-Patient-Out CV, 30-min RMSE lands in the ~13–25 mg/dL range across method classes with **no consistent deep-learning advantage**.
- **GluFormer's own claim is about long-horizon *risk*, not next-hour glucose.** Its Nature 2025 result is 4–12-year diabetes/CVD stratification beating GMI/HbA1c — a *different task* from 30–120 min RMSE. Do not cite it as evidence for adopting a FM for short-horizon forecasting.
- Many high-flying deep-CGM numbers use **within-patient or segment-level splits** (train/test windows from the same person), which leak. The patient-disjoint numbers are much less flattering.

**CGM recommendation: keep gradient boosting as the production point-forecaster.** It ties or beats the neural MoE on RMSE (as DVXR already found), runs on CPU, and matches the literature consensus. A CGM foundation model is **not worth it for RMSE**. If DVXR later wants *long-term risk stratification* (a genuinely different product feature), **GluFormer** is the model to evaluate — but that is an expansion, not a replacement.

### 2.3 Behavioral / wearable time-series recommendation

**Drop MOMENT-1-large as a hard dependency.** It is 385M params, won't build in the current env, and the benchmark evidence (arXiv:2402.03885 and the multi-FM comparisons) shows FMs beat *each other* but not necessarily task-specialized models — and never demonstrated a patient-disjoint biosignal win that would justify the weight. For behavioral time-series use a **PatchTST or GBM baseline** (both CPU, both trainable). If a FM is wanted for engineering convenience (zero-shot, one model many series), **Chronos-Bolt** (Apache-2.0, tiny variants, CPU) is the most practical, but validate it against the tree baseline before trusting it.

---

## 3. Multimodal representation learning / fusion (physiological + clinical)

### 3.1 Comparison table

| Approach | Representative work | What it buys | Honest weakness |
|---|---|---|---|
| **Availability-aware late fusion** (DVXR current) | Missing-modality survey (arXiv:2409.07825) | Robust when a modality is absent; simple; interpretable | Ignores cross-modal interactions |
| **Contrastive cross-modal alignment (CLIP/InfoNCE-style)** | Uncertainty-Aware Graph Contrastive Fusion (Neural Networks 2025, S0893608025002424); EEG+AV contrastive (Bioengineering 2024, 11/10/997) | Shared latent space; can align EEG↔wearable | Needs paired data; gains are dataset-specific, rarely production-robust |
| **Self-supervised 1D physiological fusion** | Information Fusion 2025 (10.1016/j.inffus.2025.103397) | Remote-monitoring fusion without labels | Early-stage; small cohorts |
| **Incomplete/valid-context wearable reps** | VCR (arXiv:2605.18837); missing-modality survey (arXiv:2409.07825) | Handles dropped sensors gracefully | Complexity; limited external validation |
| **Unified physiological alignment** | Brant-X (arXiv:2409.00122) | Aligns EEG with other physio signals | Ties you to Brant family (heavy) |

Curated resource: `awesome-mmps` (github.com/willxxy/awesome-mmps).

### 3.2 Fusion recommendation

**Keep availability-aware late fusion.** For a clinical-risk product where modalities are frequently missing and interpretability/auditability matter, it is the right default and matches the missing-modality literature (arXiv:2409.07825). **Contrastive alignment (InfoNCE/graph-contrastive) is worth a research spike only** — the 2025 results (S0893608025002424, inffus 2025) are real but dataset-specific and not yet shown to beat a well-tuned availability-aware fusion on an external cohort. Do not put it on the production path until it clears a patient-disjoint holdout.

---

## 4. Stress / affect detection from wearables & EEG (WESAD / DEAP-class)

| Setting | Reported | Protocol | Honest caveat |
|---|---|---|---|
| WESAD subject-*dependent* | RF/ExtraTrees/XGB **~99% F1** | within-subject / windowed | **Leaky** — windows from the same subject in train and test; not a real generalization number |
| WESAD original baseline | RF **88.3% acc** (binary), LDA **87.4% balanced** | subject-independent | The honest reference point |
| WESAD cross-subject (LOSO) | CNN ~**92.85%** 3-class; RNN ~93% F1; CNN+FFN 91.7% F1 | LOSO | Deep nets ~ classical; **no decisive FM/deep win** |
| Cross-domain (WESAD+ScikitSST-MOVE+DREAMER) | varies | cross-dataset | Generalization drops sharply across datasets |

Sources: WESAD (Schmidt et al., ICMI 2018); improved subject-independent (arXiv:2203.09663); cross-modality WESAD (arXiv:2502.18733); personalized-vs-generalized (arXiv:2308.14245); hybrid feature learning (Sensors 2026, 10.3390/s26113451).

**Read:** The famous "99%" WESAD numbers are **subject-dependent and effectively leaky**. Under LOSO, tree ensembles and compact CNNs land in the **~85–93%** band with **no clear foundation-model advantage**. For DVXR's stress/affect signal, a **well-tuned GBM on HRV/EDA/temperature features is the honest SOTA-competitive baseline**; report LOSO numbers, not windowed ones.

---

## 5. Diabetes monitoring / glucose forecasting systems

| System / method | Task | Metric (honest protocol) | Source |
|---|---|---|---|
| XGBoost/CatBoost/LightGBM | 30 min ahead, OhioT1DM | RMSE competitive w/ LSTM, ~13–25 mg/dL LOPO | rs-7410777 (2025) |
| LSTM / Bi-LSTM | 30 min, OhioT1DM | RMSE ~13.65 (LSTM) / 21.73 (Bi-LSTM) | rs-7410777 |
| ARIMA / linear | small CGM datasets | **best** on small data | GlucoBench (arXiv:2410.05780) |
| Latent-ODE / Transformer / TFT | large CGM datasets | wins only on largest; best UQ/calibration | GlucoBench |
| Gluformer (arXiv:2209.04526) | personalized forecast + UQ | best log-likelihood, not clearly best RMSE | arXiv:2209.04526 |
| **GluFormer** (arXiv:2408.11876) | **long-term risk (4–12 yr)** | beats GMI/HbA1c for diabetes/CVD | Nature 2025 s41586-025-09925-9 |
| GluMind | cross-population forecasting | robustness gains | arXiv:2509.18457 |
| MetaboNet-Bench | multimodal T1D forecasting bench | patient-disjoint 30/60/120 min | arXiv:2606.18640 |

**SOTA verdict:** For **short-horizon CGM forecasting under patient-disjoint eval, there is no method that decisively beats gradient boosting**; RMSE differences across method classes are small and dataset-dependent. Deep/foundation models earn their keep on (a) very large datasets, (b) uncertainty quantification, and (c) **long-term risk stratification** (GluFormer) — none of which is DVXR's current 30–120 min point-RMSE task. **Keep GBM.**

---

## 6. Wearable-based affective computing (leading approaches)

| Model | Modality | Params | License | Note |
|---|---|---|---|---|
| **PaPaGei** (arXiv:2410.20542) | PPG | — | Open | First open PPG FM; 57k h / 20M segments; beats engineered features on HTN/BP/HR |
| **Pulse-PPG** (arXiv:2502.01108) | PPG (field) | — | Open | Field-trained (100-day study); better lab→field transfer |
| **SensorLM** (Zhang et al., 2025) | PPG+accel, language-aligned | large | — | Sensor–language FM |
| **Apple SSL PPG/ECG** (Abbaspourazad et al., 2024) | PPG/ECG | large | Closed | Strong but proprietary |
| **AnyPPG** (arXiv:2511.01747) | ECG-guided PPG | — | Open | 100k+ h; holistic profiling |
| Classical HRV/EDA + GBM | wrist wearable | tiny | Open | **Still the honest cross-subject baseline for affect** |

**Read:** PPG foundation models (PaPaGei, Pulse-PPG) are a *real* advance for PPG-derived tasks and are the direction to watch for wearable affect — but their demonstrated wins are on physiological targets (BP, HTN, HR), not decisively on cross-subject *affect/stress* over a tuned classical baseline. If DVXR's wearable channel is PPG-heavy, **PaPaGei is the most defensible open FM to evaluate**; otherwise a GBM on HRV/EDA features remains competitive and CPU-cheap.

---

## 7. "What would actually be worth trying next" — ranked by effort ÷ expected gain

1. **Run an identity-leakage audit (FMSCOPE-style, arXiv:2606.06647) on the EEG depression pipeline.** _Effort: low. Gain: high (credibility)._ This is the highest-value action in the whole review — it tells you whether the 0.961 AUROC is a biomarker or a cohort artifact. Do this before any model swap.
2. **A/B CBraMod vs LaBraM on the EEG tasks.** _Effort: low (both MIT, both 200 Hz, ~4–6M params). Gain: small-but-real._ CBraMod edges LaBraM on several subject-disjoint tasks; drop-in preprocessing.
3. **Keep GBM for CGM; add proper uncertainty (quantile/conformal) instead of a FM.** _Effort: low. Gain: medium._ The literature says trees win RMSE; the missing piece is calibrated intervals, which conformal prediction gives cheaply — a better use of effort than a CGM FM.
4. **Swap Bio_ClinicalBERT → Clinical ModernBERT** if notes are long or latency matters. _Effort: low (drop-in HF encoder). Gain: small (≈+1 pt AUROC, 2.3× faster, longer context)._ (arXiv Clinical ModernBERT; Clinical-Longformer arXiv:2201.11838 as the long-context alternative.)
5. **Replace the MOMENT dependency with PatchTST or Chronos-Bolt for behavioral series.** _Effort: low–medium. Gain: unblocks the build._ MOMENT (385M) is the wrong tool; either fixes the "won't build" problem with a CPU-friendly model.
6. **Evaluate PaPaGei _if_ the wearable channel is PPG.** _Effort: medium. Gain: uncertain._ Real FM progress for PPG, but validate on a patient-disjoint affect holdout first.
7. **(Research-only) Contrastive EEG↔wearable alignment.** _Effort: high. Gain: speculative._ Interesting, not production-ready; keep availability-aware fusion in production.
8. **(Expansion, not replacement) GluFormer for long-term risk stratification.** _Effort: medium–high. Gain: new capability, not a forecasting upgrade._ Only if DVXR wants to add a multi-year risk product.

**Explicitly NOT worth it:** switching EEG to Brant-2/NeuroLM (1B–1.7B params, CPU-hostile, corpus mismatch); adopting any CGM foundation model to improve 30–120 min RMSE; putting cross-modal contrastive fusion on the production path now.

---

## 8. Selection-integrity note (expanded)

- **I did not favor incumbents.** LaBraM, Bio_ClinicalBERT, and GBM survived scrutiny on evidence, not by default; MOMENT did *not* survive and I recommend dropping it despite it being the current choice.
- **Where evidence was weak or non-comparable, I said so** rather than ranking: EEG FM leaderboards flip by task and split (EEG-FM-Bench, arXiv:2508.17742), so I give a *top cluster* (LaBraM/CBraMod/EEGPT/REVE) rather than a false #1. CSBrain/REVE numbers are single-paper and unverified.
- **Leaky vs honest benchmarks are separated throughout:** WESAD "99%" (subject-dependent, leaky) vs ~88–93% LOSO (honest); segment-level CGM vs patient-disjoint; and the resting-state EEG "Identity Trap" that can inflate depression AUROC under subject-disjoint CV.
- **The strongest, most transferable conclusion** is a negative one that favors no vendor: **classical baselines (gradient boosting, linear/ARIMA, compact CNNs) remain competitive-to-best for CGM short-horizon forecasting and cross-subject wearable stress**, and no foundation model in this survey has honestly dethroned them at those specific tasks and protocols.

---

## References (primary)

**EEG:** LaBraM arXiv:2405.18765 (ICLR'24); CBraMod arXiv:2412.07236 (ICLR'25); BIOT arXiv:2305.10351 (NeurIPS'23); EEGPT (NeurIPS'24); Brant-2 arXiv:2402.10251; NeuroLM arXiv:2409.00101 (ICLR'25); CSBrain arXiv:2506.23075; REVE (NeurIPS'25); EEG-FM-Bench arXiv:2508.17742; OmniEEG-Bench arXiv:2606.00815; Brain4FMs arXiv:2602.11558; EEG-FM survey arXiv:2601.17883; "Capable Yet?" arXiv:2507.01196; Identity Trap arXiv:2606.06647; Frozen-Still-Leaking arXiv:2606.09189; Cognitive-load FM arXiv:2601.21965.

**Time-series/CGM:** MOMENT arXiv:2402.03885; Chronos arXiv:2403.07815; TimesFM arXiv:2310.10688; Moirai arXiv:2402.02592; PatchTST arXiv:2211.14283; GPT4TS arXiv:2302.11939; GluFormer arXiv:2408.11876 / Nature s41586-025-09925-9; Gluformer(UQ) arXiv:2209.04526; GlucoBench arXiv:2410.05780; MetaboNet-Bench arXiv:2606.18640; GluMind arXiv:2509.18457; OhioT1DM comparative ResearchSquare rs-7410777; T2D interpretable Sci Rep s41598-025-32373-4.

**Fusion:** Missing-modality survey arXiv:2409.07825; VCR arXiv:2605.18837; Graph Contrastive Fusion Neural Networks 2025 (S0893608025002424); Brant-X arXiv:2409.00122; SSL 1D physio fusion Information Fusion 2025 (10.1016/j.inffus.2025.103397); awesome-mmps (github.com/willxxy/awesome-mmps).

**Stress/affect & wearables:** WESAD (Schmidt et al., ICMI 2018); subject-independent stress arXiv:2203.09663; cross-modality WESAD arXiv:2502.18733; personalized-vs-generalized arXiv:2308.14245; hybrid feature learning Sensors 2026 (10.3390/s26113451); PaPaGei arXiv:2410.20542; Pulse-PPG arXiv:2502.01108; AnyPPG arXiv:2511.01747.

**Clinical notes:** Bio_ClinicalBERT (Alsentzer et al., arXiv:1904.03323); Clinical-Longformer/BigBird arXiv:2201.11838; "Do We Still Need Clinical Language Models?" arXiv:2302.08091; comparative clinical LLMs arXiv:2503.23281; GatorTron (Yang et al., npj Digit Med 2022); Clinical ModernBERT (2025).
