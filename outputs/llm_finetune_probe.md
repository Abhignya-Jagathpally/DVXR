# LLM classification finetuning — honest probe (rasbt ch6 / appendix-E recipe)

We applied the rasbt *LLMs-from-scratch* classification-finetuning recipe (Chapter 6 +
Appendix E LoRA) to our soft-prompt predictor — the **full Option 3**: trainable per-modality
soft-prompt projections + LoRA adapters on the last transformer block(s) + a classification
head, last-token pooling, class-weighted cross-entropy, AdamW. LoRA (not raw unfreezing) keeps
the shared base LLM frozen and leakage-free across CV folds.

## Result on `cgmacros_diabetes` (subject-held-out, 1×4 CV, 1-AUROC ↓)

| model | 1-AUROC | note |
|---|---|---|
| xgboost floor | **0.0000** | HbA1c-derived label is near-separable from labs |
| rep:llm frozen probe | 0.1961 | frozen LLM + linear head |
| **rep:llm LoRA-finetuned** | 0.3911 | **worse** — overfits |

Per-fold finetuned error: 0.750, 0.214, 0.500, 0.100 — high variance, several folds worse
than chance. **Finetuning made it worse, not better.**

## Why (honest diagnosis)

This is not a bug in the recipe — it is a **data-regime** mismatch:
- rasbt's recipe is validated on ~25k labelled text examples; here N = **45 subjects**.
  Finetuning a 0.5B model (even with low-rank LoRA) on 45 held-out-subject points overfits
  catastrophically.
- The features are **per-window summary statistics**, not raw signal sequences, so there is
  little sequential structure for an LLM to exploit over a tuned GBM.
- The label (diabetes stratum from HbA1c) is near-deterministic from the lab features, so a
  gradient-boosted tree is already essentially perfect (0.000) — there is no headroom.

**Conclusion:** the rasbt techniques are sound, but on these small, tabular, summary-stat
datasets no LLM finetuning beats the floor — reported honestly, not faked. The legitimate
path to a win is a **different data regime**: larger N and raw-signal sequences (see the
data-acquisition step). The code (`dvxr.llm.finetune`) is retained and correct so it can be
re-run when such data is wired in.
