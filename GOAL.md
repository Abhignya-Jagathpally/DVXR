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
  [[shared-machine-thread-caps]]) — cap OMP/BLAS threads on heavy runs. **Next:** commit iter-2,
  then build Slice B (LaBraM EEG FM, now unblocked — weights confirmed loadable).
