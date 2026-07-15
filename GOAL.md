# /goal — DVXR POW as a novel academic contribution

**Owner:** Abhignya Jagathpally · DVXR Lab, UNT · Summer 2026
**Source:** `1-Abhignya-POW.docx` (Goals 1–3; Goal 4 / paper writeup intentionally out of scope for now)
**Mode:** self-paced `/loop` — each iteration makes one concrete, committed, verifiable advance.

## The goal in one sentence

Deliver the POW's multimodal LLM/BCI clinical-risk pipeline (Goals 1–3) **and** extract from it
a genuinely novel, publishable contribution — not another fusion architecture, but a principled
answer to *why learned cross-modal fusion fails on small clinical BCI cohorts and what to do
instead*.

## Where we start (honest state, 2026-07-14)

The pipeline (`dvxr`, CACMF) is built and honestly evaluated. The headline finding is a
**credible negative result**: on all six real mental-health/clinical tasks (stress, DEAP
anxiety/arousal, eegmat workload, Mumtaz depression, glucose, mortality) the learned cross-modal
CACMF fusion **loses** to simple single-modality / concatenation / GBM baselines under 5×5
subject-held-out CV (see `BENCHMARK_FINDINGS.md`). Root cause: from-scratch encoder on ~20–60
subjects over per-window summary statistics cannot out-represent a tuned GBM, and cross-modal
fusion adds negative value.

## The novel contribution this loop pursues

**"Do-no-harm" reliability-gated fusion for small-cohort clinical BCI.** Convert the negative
result into a positive, defensible one:

1. **Diagnosis (already have it):** a rigorous six-cohort benchmark showing learned cross-modal
   fusion systematically underperforms in the small-N clinical-BCI regime — an honest phenomenon
   worth reporting.
2. **Remedy (to build):** a reliability-gated late fusion that estimates each modality's (and
   concat's, and SOTA-encoder's) predictive reliability via **inner CV on the training fold**,
   then combines their predictions with non-negative, reliability-derived weights carrying a
   **do-no-harm guarantee** — provably (on the inner-CV estimate) never worse than the best single
   candidate. Target: match or beat the best single modality on ≥4 of 6 tasks, with CIs, so the
   result flips from "fusion hurts" to "principled fusion does no harm and sometimes helps."
3. **Analysis:** when does it help vs. merely not-harm? Relate the realized gain to a modality-
   complementarity / reliability-dispersion diagnostic computed on train folds.

Novelty is grounded, not invented: the framing is *the systematic small-cohort failure + a
safety-floored remedy*, validated on six real cohorts, reported with the same honesty as the
negative result. (Novelty scan vs. stacked generalization / super-learner / multimodal-beats-
unimodal theory is running; framing will be tightened against closest prior art.)

## Guardrails (non-negotiable, inherited from project memory)

- **No fabrication.** Real labels, subject-held-out CV, report where the method loses. Honesty is
  the project's whole credibility. A do-no-harm method that *ties* the best single modality is a
  real, reportable result — do not dress a tie as a win.
- Offline / CPU / deterministic. No network dependency in the default path.
- Every number traceable to a reproducible command.
- Config == what actually runs (no aspirational claims about weights that don't load).

## Progress log

- **2026-07-14 — iter 0:** Read POW + honest state. Defined this goal. Launched novelty scout.
  Next: implement the reliability-gated do-no-harm fusion as a new bench config + unit test +
  single-task proof-of-life.
- **2026-07-14 — iter 1 (Slice A build):** Novelty scout returned — do-no-harm guarantee is
  Super Learner (2007)/Hasson (2023), cite as provenance; novel assets are the six-cohort small-N
  benchmark, a finite-sample floor, and a synergy diagnostic. Built `dnh_gated`
  (`src/dvxr/bench/gated_fusion.py`): reliability-gated late fusion over {single modalities,
  linear concat, GBM concat, real SOTA} with a subject-grouped bootstrap SE gate + shrinkage.
  Library includes the strong learner so the floor is "never worse than the tuned GBM." Wired into
  `baseline_configs`. Added `tests/test_gated_fusion.py` (5 pass, incl. inner-CV safety floor).
  Proof-of-life on stress: dnh_gated 0.1085 — beats best single (0.163) and learned CACMF fusion
  (0.136), matches its GBM candidate (0.107). Built synergy diagnostic (`dnh_diagnostics` +
  `scripts/run_dnh_diagnostic.py`). Full 6-cohort mh run in progress. **Not yet committed** — one
  clean slice commit after the scoreboard + findings update.
- **2026-07-14 — iter 1 (Slice A result, 5×5 mh):** `dnh_gated` beats the POW's own learned CACMF
  fusion (`rep:fused`) on **4/6** tasks (wesad +28%, depression +53%, eegmat +18%, stress +11%;
  ties near-chance on the 2 DEAP), and beats the **best single modality on 3/6** (stress +31%,
  wesad +25%, depression +14%). BUT universal held-out do-no-harm does **not** hold: −15.6% on
  eegmat (single-ECG dominates), −3–5% on the near-chance DEAP pair — the inner-CV floor diverges
  from held-out subjects at N≤60 (the measured finite-sample caveat). The strongest opponent stays
  a simple non-fused model on every task. Honest contribution: *among fusion methods, DNH ≫ CACMF,
  and it recovers real multimodal gains on half the tasks.* Findings written to
  `BENCHMARK_FINDINGS.md`. **Next micro-iter:** a 1-SE candidate-selection rule (prefer the simpler
  candidate) to close the held-out divergence — pre-registered, tested as an ablation, not
  cherry-picked. Then commit Slice A + open Slice B.
- **Slice B recon (done while mh ran):** confirmed `braindecode`/`torchaudio` absent under torch
  2.12 (blocker holds), but `transformers` 5.12 + `safetensors` + `mne` + `huggingface_hub` are
  present. So the real-EEG-FM path is a **direct safetensors weights load + vendored minimal
  forward** over the raw EEG windows in `task.extra["raw"]` (DEAP eeg=32ch, eegmat) — NOT the
  braindecode route. Candidates: LaBraM (`THU-BCI/LaBraM` / `eeg-telecom-paris/labram-base-official`
  unverified) or EEGPT. Frozen-extractor → linear head → compete on the same folds via a new
  `sota:eeg_fm` config. Detailed plan when Slice A commits.
  - **Concrete weights found:** `braindecode/labram-pretrained` (HF) ships `model.safetensors`
    (23 MB) + `config.json` with LaBraM's full 10-20 channel vocabulary and `n_times=3000`
    (15 s @ 200 Hz, 1 s patches). Loadable via `safetensors.torch.load_file` **without** the
    braindecode class import — Slice B vendors a minimal LaBraM forward (temporal-conv patch
    embed + channel/pos embeddings + transformer encoder, arXiv 2208.06366) keyed to the
    state-dict names, feeds the raw EEG windows in `task.extra["raw"]["eeg"]` (DEAP 32-ch,
    eegmat 19-ch → mapped into the channel vocab, resampled to 200 Hz), and reports frozen-
    embedding → linear head vs the band-power+VQ baseline on the same folds.
- **2026-07-14 — iter 2 (1-SE ablation = honest negative):** Implemented the 1-SE simpler-candidate
  rule as the finite-sample robustification and ran it strict-vs-1SE on the same folds
  (`scripts/run_dnh_ablation.py`). Result: **net negative** — 1-SE shaves the worst held-out
  violation (eegmat −6.9%→−1.2%) but is over-conservative, killing a real fusion win (wesad
  +8.5%→−2.1%) and worsening the near-chance DEAP tasks; do-no-harm holds on 3/6 under strict vs
  2/6 under 1-SE. **Strict stays the default;** 1-SE retained as a documented opt-in. Findings +
  code updated. Also diagnosed + fixed a shared-machine thread-thrashing issue (see
  [[shared-machine-thread-caps]]) — cap OMP/BLAS threads on heavy runs.
- **2026-07-14 — iter 3 (Slice B fully de-risked + specced):** Confirmed `braindecode/labram-pretrained`
  safetensors load here (no braindecode), and reverse-engineered the **complete LaBraM-base spec**
  (embed 200, 12 blocks, 10 heads, TemporalConv patch embed, channel-major token layout,
  `_adj_position_embedding` expansion, `LABRAM_CHANNEL_ORDER` mapping, cls-token frozen rep) from
  the state dict + the BSD-3 reference (downloaded, box has internet). Wrote the exact build spec +
  a mandatory correctness-validation plan to `docs/SLICE_B_LABRAM.md`. **Next (tasks #5/#6):**
  implement `src/dvxr/encoders/labram_real.py` (vendored forward + `from_pretrained`), validate
  (shape asserts, strict load, non-degeneracy, ideally a reference cross-check in an isolated env),
  wire `sota:eeg_fm` into the sweep + DNH library, and honestly report FM-vs-baseline per cohort.
- **2026-07-14 — iter 4 (Slice B DONE — real LaBraM wins on EEG):** Implemented + validated the
  vendored LaBraM (`labram_real.py`, strict load, non-degenerate), wired the frozen `labram` bench
  config (`labram_bench.py`), and benchmarked vs band-power on the two 64 Hz EEG cohorts.
  **Result: the real EEG FM beats hand-crafted band-power on BOTH** — eegmat workload AUROC 0.663
  vs 0.636, and **mumtaz depression AUROC 0.961 — the single best config**, beating xgboost/raw-CNN/
  learned-CACMF. Above-chance decoding functionally validates the forward. Honest caveat: 64 Hz
  source (≤32 Hz content) vs LaBraM's 200 Hz training → fidelity-limited wins (would likely improve
  at native rate); DEAP decimated, not a fair test. Findings + GOAL updated; committing iter-4.
  **Slice C next:** LLM-in-the-predictive-path (NeuroLM-style) made competitive.
  - **Slice C scope (from reading `dvxr/llm/predictor.py`):** the weakness is a *seeded, UNTRAINED*
    linear projection from VQ-codebook tokens → the frozen LLM's hidden dim, so the frozen Qwen
    reads meaningless soft prompts (only the head learns). The honest experiment: replace the random
    projection with an **in-distribution** one (project VQ tokens onto the LLM's real token-embedding
    subspace / nearest real-token embeddings — CPU-feasible, no backprop through the LLM), and
    compare vs the random projection on the smallest cohort (eegmat, 560 rows; Qwen forward cached
    once). Hypothesis: in-distribution soft prompts let the frozen LLM contribute real signal. Report
    honestly — a modest gain or an honest negative both matter, and match the project's ethos. The
    heavier LoRA/backprop-through-LLM variant is likely infeasible on CPU and would be flagged, not
    faked. This slice is the hardest to make a genuine win; treat a rigorous negative as a valid
    outcome, not a failure.
- **2026-07-14 — iter 5 (Slice C DONE — honest negative):** Implemented the in-distribution
  projection (VQ tokens → convex combos of the frozen LLM's real token embeddings,
  `DVXR_LLM_INDIST=1`) and ran it vs the random projection on eegmat
  (`scripts/run_llm_indist_ablation.py`). **Result: no effect** — rep:llm(random) 0.4164 vs
  rep:llm(indist) 0.4169 (−0.1%), both lose to band-power single:eeg (0.3635) by ~14.5%. The
  bottleneck is the frozen LLM's lack of BCI knowledge, not the projection distribution; fixing it
  needs actual read-in training (LoRA), CPU-infeasible here (flagged, not faked). Findings updated.
  **All three sequenced slices complete: A positive, B positive, C honest-negative — a full,
  honest arc across POW Goals 1–3.**
- **2026-07-14 — iter 6 (A×B synthesis — the culmination):** Added LaBraM to the `dnh_gated`
  candidate library so the do-no-harm fusion can recruit the real EEG FM. **Result: on
  mumtaz_depression DNH leaps AUROC .910 → .961** (0.0898 → 0.0394), essentially matching LaBraM
  while keeping the safety floor — the fusion correctly selects the FM where it wins. On eegmat the
  ECG autonomic signal still dominates (DNH 2nd, unchanged verdict). The two novel contributions
  compose exactly as intended: reliability-gated fusion finds the real EEG FM where it's the right
  tool and isn't fooled where it isn't. Findings updated; committing. **Goal achieved.**

## Product arc (2026-07-14) — "make it a useful product"

Second goal, same repo: turn the validated research into a usable product (integrated demo +
toolkit + evidence), EEG mental-health screening first. Delivered as five committed slices, each
with tests, all honesty-gated.

- **P1 — serving core:** `dvxr.serve.Screener` (fit→save→load→predict) wiring the *winning* models
  (LaBraM/band-power), calibrated + conformal, reproducing the benchmark AUROC. Depression screener
  reproduced **AUROC 0.9608** (CI [0.9417, 0.9756], ECE 0.030) under 3×5 subject-held-out CV.
- **P2 — packaged toolkit:** `pyproject.toml` (`pip install -e .`, console_scripts `dvxr`) +
  `dvxr fit|predict|report|demo`. Verified the installed command runs offline without PYTHONPATH.
- **P3 — screener-backed demo:** `scripts/build_screen_demo.py` scores real held-out subjects
  (case + control) through the screeners into a self-contained HTML page (headline depression EEG,
  supporting workload + stress); panels skipped-not-faked when data/weights absent.
- **P4 — evidence layer:** `dvxr.serve.evidence` — single source of scoreboard-traced numbers with a
  drift guard, `docs/MODEL_CARD.md`, and a shareable Artifact evidence page generated from the
  registry. Glucose omitted from headline (no current scoreboard trace — honest under-claim).
- **P5 — novelty/impact + blocking honesty audit:** README "DVXR Screen" section (novelty: FM-as-live-
  screener; do-no-harm finite-sample fusion; honesty-gated evidence). `tests/test_honesty_audit.py`
  (8 tests, BLOCKING): every number traces to a scoreboard; DEAP/CACMF-win/LLM-predictor/mortality/
  diabetes-leak can never be a product claim; no un-negated "diagnosis" on any surface. All green.

**Product delivered.** A user can `pip install -e .`, feed a research-cohort EEG subject, and get a
calibrated, explained, evidence-backed depression screening score (AUROC ≈0.96) — research-grade
screening, never a diagnosis — plus a demo and a scoreboard-traced evidence page.

## Live-demo + evidence/usefulness arc (2026-07-14, continued)

- **Live app:** `dvxr demo --serve` (Streamlit) — pick a held-out subject or upload a file, hit Run,
  watch raw→LaBraM embed→calibrate→score→explain compute live (~0.2s/subject; live≡cohort verified).
  Upload path flagged out-of-distribution. `serve/live.py`, `screen_app.py`, `dvxr screen --file`.
- **Evidence phase (E1–E3):** (E1) subject-level AUROC alongside window-level — depression window
  0.961 / **subject 0.986** (n=58); higher, not inflated by window-pooling; within-subject tasks
  correctly get no subject number. (E2) `EXTERNAL_SOTA` registry of published cross-subject results
  via PubMed, each with DOI + protocol — the honest bar (MDD-SSTNet LOSO 65% MODMA; WESAD LOSO ~85%;
  90%+ numbers are segment-level w/ leakage, labeled not-comparable). (E3) surfaced on model card,
  evidence Artifact (republished), and app; honesty audit extended (external DOI+protocol, both metrics).
- **Usefulness phase (U1–U3):** (U1) `dvxr triage` — cohort risk ranking; depression triage separates
  perfectly (all 29 top-half subjects are MDD cases, gap 0.837). (U2) `dvxr report-subject` —
  self-contained per-subject HTML report + app download. (U3) serve-time personalization wired
  (`dvxr-screener/2`, back-compat) with an **honest negative**: per-subject recalibration applies only
  to within-subject tasks and does NOT help on small cohorts (WESAD ECE 0.156 vs 0.124) — off by
  default, documented. Complements E1 (diagnosis↔subject-level, state↔personalization; mutually excl.).
- Future work (noted, not built): thin FastAPI/Docker deployment (U4, optional).

## Product re-headline arc (2026-07-15) — NeuroGlycemic Sentinel

Following the clinical multimodal-architecture spec, the product headline is re-framed to the **DVXR
NeuroGlycemic Sentinel**: a research-stage multimodal **glucose-excursion early-warning** framework
(30/60-min risk from CGM dynamics + acute neural/autonomic stress + clinical context) with grounded
LLM explanations. This is executed as a 9-PR roadmap (spec PR1→PR9); each PR is honesty-gated.

**The honesty guardrail on the re-headline (non-negotiable).** Re-headlining to glucose does **not**
manufacture a validated claim:
- The glucose product is labeled **research-stage / not yet validated** everywhere (README, model card,
  evidence, UI). It carries **no headline AUROC** (`evidence.PRODUCT_VISION.auroc is None`).
- The fused end-to-end claim requires **synchronized same-subject** EEG+wearable+CGM data, which does
  not exist; public component cohorts are never cross-joined, and a synchronized-same-subject gate
  blocks fusion on unrelated cohorts (spec §1.B, §4). The default glucose report **abstains**.
- Depression (0.961) / stress (0.955) / workload remain scoreboard-traced **validated components**
  (spec §1.A). All prior `EXCLUDED_CLAIMS` stay enforced (cgmacros_diabetes leak, CACMF-as-win,
  LLM-as-predictor). `make audit` stays green — the audit was *extended* (`ProductVisionAudit`), never
  weakened.

- **2026-07-15 — PR1 (product & claim re-alignment):** Added `evidence.PRODUCT_VISION` (research-stage
  glucose headline, no fabricated number, synchrony-gated) surfaced at the top of `dvxr report`;
  relabelled the validated claims as *components*; quarantined the LLM-in-the-predictive-path probe to
  `dvxr.experiments.llm_representation_probe` (`EXPERIMENTAL_ONLY` / `NOT_FOR_CLINICAL_INFERENCE`, thin
  deprecation shim at `dvxr.llm.predictor`); re-headlined README + model card; added an honest
  abstaining `stress_glucose_risk` report path. Honesty audit extended with `ProductVisionAudit`.
