# CACMF — real-label benchmark findings (honest)

This is the scientific evaluation the earlier synthetic-fixture numbers were **not**.
It answers a sharp audit: make the proposed model actually predict, evaluate on
non-circular real labels, try to beat a genuine baseline by a significant
relative-error-reduction (RER) ≥ 50%, and **report where it doesn't**.

Reproduce: `python3 scripts/run_benchmark.py --repeats 5 --folds 5 --ablate`
→ `outputs/benchmark_scoreboard.{csv,md}`. Harness: `src/dvxr/bench/`.

## Headline: the fused model does NOT win

Repeated subject/patient-held-out CV (5×5), error metrics (lower is better),
RER = (base_err − prop_err)/base_err, 95% bootstrap CI, paired one-sided Wilcoxon,
Holm across tasks. Proposed = CACMF fused (encoder + VQ + cross-modal transformer)
→ shared head. Baseline = the single strongest **non-fused** opponent on the same
folds (trivial floor, classical GBM, best single modality, or a real pretrained
SOTA encoder).

| task | metric | best baseline | base err | fused err | RER% | 95% CI | Wilcoxon p | Holm p | meets ≥50%? |
|---|---|---|---|---|---|---|---|---|---|
| stress | 1−AUROC | rep:pca (concat) | 0.108 | 0.129 | **−19.9%** | −28.6 … −13.5 | 1.000 | 1.0 | **No** |
| glucose | MAE (mg/dL) | rep:raw (concat) | 10.66 | 13.09 | **−22.8%** | −25.7 … −20.1 | 1.000 | 1.0 | **No** |
| mortality | 1−AUROC | rep:pca | 0.178 | 0.360 | **−101.7%** | −157 … −61 | 1.000 | 1.0 | **No** |

**None of the three tasks meets the ≥50% RER bar.** On every task the learned
fusion is *worse* than a strong baseline, and the CIs exclude zero on the losing
side — i.e. the loss is statistically real, not noise. This is the credible,
Brain2Qwerty-style outcome: reporting where a method does not help is more useful
than seven suspicious 0.99s.

## What DOES hold up (real, if modest)

- **Combining modalities beats the best single modality (stress).** Concatenating
  all four physiology streams (`rep:pca` 1−AUROC 0.108) improves on the best single
  modality (`single:motion` 0.167) by ~35% RER. Multimodality helps — but *naive
  concatenation* captures it; the learned CACMF fusion (0.129) adds nothing over
  concat and in fact regresses.
- **A simple learned model beats persistence (glucose).** `rep:raw`/`rep:pca`
  (MAE 10.66) beat the 30-min persistence baseline (12.88) by ~17% RER. A real,
  significant win — but it is a linear model on history features, not CACMF.
- **True modality ablation is sensible.** Retraining the fused model *without* each
  modality (not zero-filling) shows motion dominates stress (Δ1−AUROC +0.086, CI
  [0.067, 0.106]), then ppg > temp > eda — each contributes, motion most.

## Why fusion loses here (root cause, not excuses)

1. **Features are per-window summary statistics, not raw signal** (audit C2). The
   ceiling is set by the features; a cross-modal transformer over 8–24 summary stats
   per modality cannot out-represent what a tuned GBM already extracts from them. A
   fair test of the *architecture* needs raw windowed signal (and a SOTA encoder fed
   raw sequences, not summaries).
2. **From-scratch encoder, ~20 subjects.** The CACMF encoder is trained per fold on a
   few hundred windows from ≤16 training subjects. That cannot beat a tuned
   HistGradientBoosting or PCA→logistic on the same features, let alone by 50%.
3. **Strong baselines.** Concat→PCA hits ~0.89 AUROC on stress; persistence is a hard
   glucose floor. ≥50% error reduction over those would require near-perfect
   cross-subject performance that current stress/CGM literature does not reach.
4. **Tiny mortality set.** 100 patients / 15% positives → the fused model overfits and
   the CIs are wide; no method should be trusted to a 50% claim here.

## SOTA opponents actually run

`sota:` uses a **real pretrained foundation model as a frozen feature extractor**
(computed once over all rows, so no leakage; only the shared head refits per fold):
- stress → **MOMENT-1-large** (real weights) — 1−AUROC 0.274 (weak on summary features)
- mortality → **Bio_ClinicalBERT** (real weights) — 1−AUROC 0.313
- glucose → **CGM-JEPA fell back** to a summary encoder. CGM-JEPA operates on raw CGM
  *sequences* and needs custom loading; our summary-feature pipeline can't feed it
  meaningfully. Integrating it on raw sequences is future work — flagged, not hidden.

## How each audit finding is addressed

| # | finding | status |
|---|---|---|
| B1 | encoder/fusion never fed the heads | **Fixed** — `rep:fused` = trained encoder+VQ+fusion latent → shared head; `cacmf_e2e` also reported (own head). |
| B2 | 6/7 labels were circular median-splits | **Fixed** — only real external labels (Non-EEG annotations, real future glucose, MIMIC mortality); proxies excluded from the benchmark path. |
| B3 | label fabrication (class-flip, subject-dup) | **Fixed** — `assert_no_fabrication()`; the bench path never calls those helpers. |
| B4 | synthetic-fixture 0.99s as the story | **Replaced** — headline is now real held-out numbers; fixtures are validation-only. |
| M1 | single split, no CIs/significance | **Fixed** — 5×5 grouped CV, bootstrap CIs, paired Wilcoxon, Holm, Cliff's δ. |
| M2 | no real baselines / SOTA | **Fixed** — persistence/majority, classical GBM, best single modality, real MOMENT/Bio_ClinicalBERT. |
| M3 | fusion absent | **Present** — CACMF cross-modal fusion is the proposed model and is evaluated. |
| M4 | overlapping-window leakage | **Fixed** — non-overlapping windows; subject-disjoint folds. |
| C1 | zero-fill "ablation" | **Fixed** — true retrain-without-modality ablation. |
| C2 | summary-stat features, weak SSL | **Confirmed** as the ceiling; documented as the main limitation. |

## Bottom line

The plumbing is now honest and the proposed model is genuinely evaluated. On these
real, credential-free tasks CACMF's learned fusion does not beat strong baselines and
does not approach a 50% relative-error reduction. The defensible positive claims are
narrower and real: multimodality (via concatenation) beats the best single modality on
stress, and a simple learned model beats glucose persistence — both modest, both
significant. Closing the gap for the *architecture* would require raw-signal inputs and
SOTA encoders fed raw sequences, not summary statistics.
