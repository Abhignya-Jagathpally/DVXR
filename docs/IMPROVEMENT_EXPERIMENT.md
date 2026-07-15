# Bounded improvement experiment — pre-registration

Phase 2 of the glass-box /goal. The rule from Phase 1 stands: the proposed multimodal path loses on
full-observation accuracy, and we will not sell a loss as a win. This document **pre-registers** one
bounded, literature-grounded change, the exact bars it must clear, and the decision rule — *before*
measuring — so the result (win **or** honest negative) is credible either way.

## The observation that motivates the lever

The glass-box trace surfaced a concrete, real weakness. On WESAD the per-modality VQ codebooks use only
a sliver of their capacity — measured batch perplexity **2.5–5.7 out of K = 64 codes**
(`outputs/product/glassbox/glassbox_wesad_stress.html`, and reproducible via
`dvxr glassbox --task wesad_stress`). That is textbook **codebook collapse / underutilization**: a few
codes dominate, most are dead, so the discrete tokens the frozen LLM reads carry far less information
than the codebook could. Our tokenizer already patches this reactively (EMA + dead-code reinit,
`encoders/codebook.py:124,137`) but still collapses in practice.

## Candidate levers (literature)

| Lever | What it changes | Why | Invasiveness |
|---|---|---|---|
| **SimVQ** (Zhu et al. 2024, [2411.02038](https://hf.co/papers/2411.02038)) | reparameterize the codebook through **one linear layer**, optimizing the whole code space instead of individual vectors | directly targets representation collapse; preserves the codebook API (indices, perplexity) | **minimal** — one `nn.Linear` in `VectorQuantizer` |
| FSQ (Mentzer et al. 2023, [2309.15505](https://hf.co/papers/2309.15505)) | drop the codebook; quantize each latent dim to fixed levels | eliminates collapse by construction | moderate — changes code semantics + downstream dims |
| CVQ-VAE / VQBridge ([2307.15139](https://hf.co/papers/2307.15139), [2509.10140](https://hf.co/papers/2509.10140)) | online-clustered / annealed codebook updates for ~100% utilization | stronger than our dead-code reinit | moderate |

**Chosen primary lever: SimVQ.** It is the smallest change that attacks the observed failure mode
head-on, it keeps every existing interface (`quantize` still returns code indices; `perplexity` still
works; the LLM soft-prompt path and CACMF fusion consume it unchanged), and it is toggled behind a flag —
matching the "minimal, necessary, expert" constraint. FSQ is the documented fallback if SimVQ does not
lift utilization.

## Hypotheses (pre-registered, ordered by how much we'd believe a win)

- **H1 (mechanism — most likely):** SimVQ raises codebook perplexity/utilization vs the current VQ on
  the same latents (a real, honest tokenizer-quality win, independent of downstream accuracy).
- **H2 (representation):** the frozen-LLM soft-prompt representation built on SimVQ tokens does **not
  lose** downstream AUROC vs the current VQ, and ideally gains, on a multimodal task.
- **H3 (robustness — the strategic bar):** SimVQ improves the sensor-dropout curve (`streaming_eval.py`)
  — a lower crossover level or a larger CI-backed margin — since richer tokens should degrade more
  gracefully.

We explicitly expect that **H1 is achievable and H2/H3 may well stay negative**: our fusion loses partly
because the WESAD modalities are *redundant* (low joint-gain in `dnh_diagnostics`), a data property no
tokenizer fixes. A negative on H2/H3 is a real result and will be reported as such.

## Baselines it must beat (exact, committed)

1. **Codebook perplexity** of the current `VectorQuantizer` on the same modality latents (measured, not
   assumed) — H1's bar.
2. **LLM-rep held-out AUROC** on `wesad_stress` (and/or `eegmat_workload`) under the existing 3×5
   subject-held-out protocol (`repeated_group_folds`) — H2's bar; must not regress.
3. **do-no-harm gated fusion + best single modality** on the committed board
   (`outputs/benchmark_scoreboard.csv`: WESAD best `xgboost` err 0.0453; proposed fusion err 0.1294) —
   the product bar; a full-obs win must clear this, and we do not expect it to.
4. **Sensor-dropout crossover** from `streaming_eval.py` for the task — H3's bar.

## Decision rule

- Report a **win** on a hypothesis only if the metric improves with a margin that survives its own
  noise: perplexity across seeds (H1), a bootstrap/repeat CI on AUROC that excludes a tie (H2), or a
  CI-backed crossover shift (H3). Same discipline as the honesty audit.
- If SimVQ lifts H1 but not H2/H3, report exactly that: "better tokenizer utilization, no downstream
  accuracy gain — consistent with modality redundancy, not tokenizer quality, bounding fusion here."
- Any genuine, scoreboard-traced gain is folded into the glass-box demo + registry (Slice 6); otherwise
  the negative is recorded in this file, `docs/PIPELINE_DEEP_DIVE.md`, and the model card, and the loop
  closes. **No fabricated wins under any branch.**

## Mechanics

- Implement SimVQ behind a flag on `VectorQuantizer` (`encoders/codebook.py`) — a learned linear
  reparameterization of the codebook lookup; default off (current VQ unchanged). Wire a
  `VQBiosignalEncoder(..., quantizer="simvq")` / env toggle so `llm.predictor` and `fusion.model` can opt
  in without edits.
- Measure H1 directly (perplexity, current vs SimVQ, ≥3 seeds); H2 via the existing
  `serve.screener._fit_head` + `repeated_group_folds` on the LLM rep; H3 via `streaming_eval.py`.
- Thread-capped (shared host). Deterministic seeds. Results appended below.

## Results

**H1 (SimVQ raises codebook utilization): REFUTED — a robust negative.** Measured on all 7 real WESAD
modalities (mean batch perplexity over seeds 7/13/21, K = 64, the LLM path's 8-epoch setting):

| | mean perplexity |
|---|---|
| current VQ (EMA + dead-code reinit) | **6.66** |
| SimVQ (one-linear reparameterization) | **3.03** (−3.63) |

SimVQ was *worse* on every modality. An epoch sweep on `ecg` shows the gap is not an under-training
artifact — it widens with training, because the existing VQ's dead-code reinit keeps reviving unused
codes while SimVQ (no reinit, one shared linear layer over a 24-d latent) does not:

| epochs | current VQ | SimVQ |
|---|---|---|
| 8   | 6.5  | 3.0 |
| 30  | 10.9 | 3.9 |
| 60  | 16.2 | 2.3 |
| 120 | 24.2 | 5.4 |

**Interpretation (honest).** SimVQ's shared-linear reparameterization is designed for *large* image
codebooks trained long; at this repo's scale (K = 64, tiny biosignal latents, few epochs) it does **not**
fix collapse and in fact utilizes fewer codes than the EMA + dead-code baseline already in
`encoders/codebook.py`. Our existing anti-collapse machinery is the stronger choice here.

**Incidental finding (a real, cheap lever).** The low utilization the glass-box surfaced (perplexity
2.5–5.7) is **not** a VQ ceiling — it is the LLM path's short training (`_modality_quant` uses
`epochs=8`, `predictor.py:180`). The *same* VQ reaches perplexity ~24 at 120 epochs. So the honest,
minimal way to enrich the discrete tokens is simply **more VQ epochs in the LLM path**, not a new
tokenizer.

**Decision (per the pre-registered rule).**
- H1 failed, so H2/H3 are not pursued: SimVQ does not improve the mechanism it was chosen to improve, so
  there is no basis to expect a downstream (H2) or robustness (H3) gain from it. Reported as a negative.
- No fabricated win. Nothing is folded into the product (Slice 6 records the negative and closes).
- **What ships:** SimVQ stays as an **off-by-default, documented experimental flag** (`DVXR_VQ=simvq` /
  `VQBiosignalEncoder(simvq=True)`) with this negative recorded — the repo's standing practice of keeping
  tested negatives visible rather than hiding them. The actionable recommendation (raise the LLM-path VQ
  epochs) is noted for a future, separately-benchmarked change; we do not alter the committed LLM-rep
  numbers here.

This is the honest outcome the pre-registration anticipated: a bounded lever, measured against committed
baselines, that did not help — recorded, not spun.
