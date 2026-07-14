# CACMF — real-label benchmark findings (honest)

This is the scientific evaluation the earlier synthetic-fixture numbers were **not**.
It answers a sharp audit: make the proposed model actually predict, evaluate on
non-circular real labels, try to beat a genuine baseline by a significant
relative-error-reduction (RER) ≥ 50%, and **report where it doesn't**.

Reproduce (mental-health concentration — the current committed scoreboard):
`python3 scripts/run_benchmark.py --profile mh --repeats 5 --folds 5 --ablate`.
Clinical profile: `--profile clinical` (stress/glucose/mortality). Legacy default:
`--tasks stress glucose mortality`. → `outputs/benchmark_scoreboard.{csv,md}`. Harness: `src/dvxr/bench/`.

## Mental-health concentration: fusion does NOT win (and DEAP is near-chance)

The `mh` profile is the real-label mental-health emphasis: stress (peripheral physiology),
the DEAP EEG+peripheral **affective/BCI** tasks with genuine self-report (SAM) labels —
`deap_anxiety` (high-arousal + low-valence quadrant) and `deap_arousal` — and `eegmat_workload`,
a **real cognitive-workload** cohort (PhysioNet EEG mental-arithmetic: resting baseline vs
serial-subtraction, 19-ch EEG + ECG @ 64 Hz), and `mumtaz_depression`, a **real depression**
cohort (Mumtaz 2016 MDD-vs-healthy resting EEG, 19-ch @ 64 Hz, subject-level diagnosis). **Every
mental-health target the proposal names now has a real labeled benchmark** — the median-split
proxies in `clinical_tasks.py` are superseded and no longer cited. 5×5 subject-held-out CV:

| task | metric | best baseline | base err | fused err | RER% | 95% CI | meets ≥50%? |
|---|---|---|---|---|---|---|---|
| stress | 1−AUROC | rep:pca (concat) | 0.108 | 0.129 | **−19.9%** | −28.6 … −13.5 | **No** |
| wesad_stress | 1−AUROC | xgboost | 0.045 | 0.129 | **−185.9%** | −441.6 … −80.5 | **No** |
| deap_anxiety | 1−AUROC | single:physiology | 0.466 | 0.469 | **−0.6%** | −6.4 … 5.4 | **No** |
| deap_arousal | 1−AUROC | single:physiology | 0.452 | 0.458 | **−1.2%** | −7.0 … 4.7 | **No** |
| eegmat_workload | 1−AUROC | single:physiology (ECG) | 0.260 | 0.365 | **−40.4%** | −56.7 … −26.8 | **No** |
| mumtaz_depression | 1−AUROC | sota (MOMENT) | 0.082 | 0.205 | **−148.2%** | −224.4 … −86.7 | **No** |

Two honest observations. (1) The learned CACMF fusion never beats the strongest floor on
any MH task — same negative result as the clinical profile. (2) On DEAP, **every** config —
floor, MOMENT-SOTA, and fused alike — sits near chance (1−AUROC 0.45–0.49 ⇒ AUROC ≈ 0.51–0.55):
cross-subject affective decoding from *per-window summary statistics* is essentially at the
noise floor here.

### The raw-signal lever (Slice H): tested, and it does not rescue DEAP

The `raw_cnn` config (a multimodal 1D-CNN over the raw EEG+peripheral windows, wired into the
sweep on the same subject-held-out folds — `src/dvxr/bench/raw_seq.py`) is the honest test of
whether waveform structure, not summary stats, carries the affective signal. Result: it lands
**at chance** — 1−AUROC **0.504** (deap_anxiety) and **0.504** (deap_arousal) — *worse* than the
summary-stat `single:physiology` floor (0.466 / 0.452). So raw signal does **not** rescue DEAP
here. The reason is concrete and disclosed: DEAP's canonical events are the loader's *decimated*
preprocessed signal (~8 Hz effective), which aliases away the EEG oscillations (α/β) that carry
affect — so the raw window is a coarse waveform, not the 128 Hz signal.

**The fidelity contrast is the point.** On `eegmat_workload`, loaded at a proper 64 Hz (δ/θ/α/β
intact), the *same* `raw_cnn` reaches **AUROC ≈ 0.66** (1−AUROC 0.343) — far above its chance
0.50 on decimated DEAP, on par with MOMENT-SOTA (0.338) and ahead of the fused CACMF (0.365),
though still below the ECG floor (0.260). So the raw lever demonstrably extracts real biosignal
**when the sampling supports it**; DEAP's decimation, not the method, is the ceiling there.
Full-rate raw DEAP is future work.

### Cognitive workload is real-decodable — and fusion still loses

`eegmat_workload` is the honest replacement for the beta/alpha proxy, and unlike DEAP it carries
clear signal: the best single modality — **ECG** (`single:physiology`, the cardiac response to
mental arithmetic) — reaches **AUROC ≈ 0.74** (1−AUROC 0.260), and `raw_cnn` on the EEG+ECG
waveform reaches AUROC ≈ 0.66. Yet the learned CACMF fusion (0.365) **loses to the single ECG
modality by −40.4%** (CI −56.7…−26.8) — the same verdict as every other task: concatenative/
single-modality baselines beat the learned cross-modal fusion. Physiologically sensible, too:
autonomic (heart-rate) response to arithmetic load is a stronger, lower-variance workload signal
than cross-subject EEG band-power.

### Depression: highly decodable, yet learned fusion still loses hardest

`mumtaz_depression` is the last mental-health proxy converted to a real cohort (58 subjects,
28 healthy / 30 MDD, subject-held-out). Depression is the *most* separable of the six: the
xgboost floor and MOMENT-SOTA both reach **AUROC ≈ 0.92** (1−AUROC ≈ 0.082) from resting
band-power — high (the Mumtaz set is known to be comparatively separable, so read this as
dataset-specific, not a universal MDD-decoding claim). And it is the task where the learned CACMF
fusion loses **hardest**: `rep:fused` 0.205 vs the 0.082 floor → **−148.2% RER** (CI −224 … −87).
The pattern is now unmistakable across all six real tasks: where the signal is weak (DEAP) fusion
ties near chance; where it is strong (workload, depression) simple GBM / single-modality / MOMENT
baselines capture it and the cross-modal fusion *adds negative value*. A credible, honest negative
result on learned fusion — on real mental-health labels, not proxies.

## The remedy: do-no-harm reliability-gated late fusion (`dnh_gated`)

The negative result above motivates a concrete question: *if learned cross-modal fusion is the
wrong tool in this small-cohort regime, what is the right one?* `dnh_gated` (in
`src/dvxr/bench/gated_fusion.py`) is the answer we test — a reliability-gated **late** fusion that
sidesteps joint training entirely. It builds out-of-fold predictions for a candidate library
{each single modality, linear concat, gradient-boosted concat, real SOTA encoder} via
subject-grouped **inner** CV on the train fold, then combines them with non-negative
Super-Learner-style weights, **shrunk toward the single best candidate** and **accepted over it
only when the inner-CV advantage clears one subject-grouped bootstrap standard error** — otherwise
it falls back. Provenance is cited, not claimed: the do-no-harm floor is the Super-Learner oracle
inequality (van der Laan et al. 2007; Hasson et al. ICML 2023); reliability-weighted late fusion of
physiological signals also predates us (Wei et al. 2018; Han et al. TMC 2021/22). What is new here
is the **finite-sample** treatment for N≤60 and the six-cohort characterization below.

5×5 subject-held-out CV, same folds as everything above (`--profile mh`). `dnh_gated` error vs. the
best single modality, vs. the proposal's own learned CACMF fusion (`rep:fused`), and vs. the single
strongest **non-fused** opponent on each task:

| task | dnh_gated | best single | RER vs best single | rep:fused | RER vs rep:fused | strongest opponent | RER vs it |
|---|---|---|---|---|---|---|---|
| stress | 0.1154 | single:motion 0.167 | **+30.9%** | 0.1294 | **+10.8%** | rep:pca 0.1079 | −7.0% |
| wesad_stress | 0.0929 | single:resp 0.1243 | **+25.3%** | 0.1294 | **+28.2%** | xgboost 0.0453 | −105% |
| deap_anxiety | 0.4825 | single:physiology 0.4658 | −3.6% | 0.4688 | −2.9% | single:physiology | −3.6% |
| deap_arousal | 0.4726 | single:physiology 0.4522 | −4.5% | 0.4575 | −3.3% | single:physiology | −4.5% |
| eegmat_workload | 0.3003 | single:physiology 0.2598 | −15.6% | 0.3649 | **+17.7%** | single:physiology | −15.6% |
| mumtaz_depression | 0.0964 | single:eeg 0.1121 | **+14.0%** | 0.2046 | **+52.9%** | sota (MOMENT) 0.0824 | −17.0% |

Read this honestly — it is a nuanced positive, not a clean sweep:

1. **`dnh_gated` reliably beats the proposal's own learned cross-modal fusion.** On 4 of 6 tasks it
   improves on `rep:fused` (stress +11%, wesad **+28%**, eegmat **+18%**, depression **+53%**); on
   the two DEAP tasks it ties within a few points, and DEAP sits at chance for *every* config
   (1−AUROC ≈ 0.47–0.52). Where the POW's CACMF loses −20% to −186% vs. strong baselines, the
   late-fusion remedy is *far* better. **If you are going to fuse modalities in this regime,
   reliability-gated late fusion strictly dominates the learned cross-modal transformer here.**
2. **It beats the best single modality on 3 of 6** (stress +31%, wesad +25%, depression +14%) —
   real multimodal gains where the modalities are complementary.
3. **But universal do-no-harm does NOT hold on held-out subjects.** On `eegmat_workload` it loses
   −15.6% to single-ECG, and on the two near-chance DEAP tasks it slips −3–5%. The safety floor is
   guaranteed on the *inner-CV* estimate; at N≤60 that estimate **diverges** from held-out subjects
   (on eegmat the inner CV preferred a concat/GBM candidate that generalized worse than ECG-alone).
   This is not a bug hidden — it is the central finite-sample caveat, now measured: the asymptotic
   Super-Learner floor is not a held-out guarantee at this cohort size. A more conservative
   candidate-selection rule (a 1-SE rule preferring the simpler candidate) is the natural next test.
4. **The practical ceiling is still a simple non-fused model.** On every task the single strongest
   opponent is non-fused — tuned GBM/PCA on concat, single-ECG, or a frozen MOMENT encoder — and no
   fusion here reaches it. Multimodality is not the winner; but *among fusion methods*, DNH ≫ CACMF.

So the contribution is honest and specific: reliability-gated late fusion converts the POW's
learned-fusion losses into wins-or-ties against that fusion and recovers real multimodal gains on
half the tasks, while its own failure to guarantee held-out do-no-harm at N≤60 is itself a
measured, reportable finding (the synergy/redundancy diagnostic in `outputs/dnh_diagnostic.md`
characterizes *when* it helps vs. merely not-harms). Reproduce:
`python3 scripts/run_benchmark.py --profile mh --repeats 5 --folds 5` (config `dnh_gated`).

### Ablation: the 1-SE candidate-selection rule — tested, and it does NOT help (kept off)

The one held-out failure above (eegmat −15.6%) is the inner-CV floor over-trusting a
concat/GBM candidate that generalized worse than single-ECG. The natural robustification is a
**1-SE rule**: among candidates within one subject-grouped bootstrap SE of the best inner-CV
error, pick the *simplest* (a single modality over concat over a boosted tree), so at N≤60 the
noisy inner-CV estimate cannot promote an over-capacity candidate on a tie. We implemented it
(`strict=False` / `DVXR_DNH_1SE`) and ran it head-to-head against the strict argmin selection on
the same folds (`scripts/run_dnh_ablation.py`, 3×5, no-SOTA candidate set for speed on a shared
host). RER vs. best single modality (>0 = do-no-harm holds on held-out subjects):

| task | strict | 1-SE | effect |
|---|---|---|---|
| stress | +33.5% | +36.3% | ~tie |
| wesad_stress | **+8.5%** | **−2.1%** | 1-SE kills a real fusion win |
| deap_anxiety | −2.3% | −6.2% | worse |
| deap_arousal | −3.7% | −5.3% | worse |
| eegmat_workload | **−6.9%** | **−1.2%** | 1-SE closes the target gap |
| mumtaz_depression | +10.2% | +11.5% | ~tie |

**The 1-SE rule is a net negative.** It does exactly what it was designed to do on eegmat (shaves
the worst held-out violation, −6.9%→−1.2%), but it is over-conservative: it falls back off a
genuinely-helpful blend on wesad (+8.5%→−2.1%, a win turned into a loss) and worsens the
near-chance DEAP tasks. Aggregate: do-no-harm holds on **3/6 tasks under strict vs. 2/6 under
1-SE**, and mean RER is higher for strict (+6.6% vs. +5.5%). So **strict argmin stays the default**;
the 1-SE rule is retained only as an opt-in for this documented ablation. Honest negative on a
plausible fix — reported, not buried. (The deeper lesson: at N≤60 there is no free lunch between
worst-case safety and average multimodal gain; a per-task, not global, selection rule is the open
question.)

## Slice B: a REAL EEG foundation model (LaBraM) — and it wins on EEG

The proposal names an EEG foundation model; earlier findings recorded that LaBraM was **not
wireable** here (braindecode won't import under torch 2.12 — no torchaudio). That blocker is now
resolved a different way: `braindecode/labram-pretrained` ships plain **safetensors**, so
`src/dvxr/encoders/labram_real.py` loads the real pretrained weights through a **vendored
LaBraM-base forward** (no braindecode import) — validated by a strict state-dict load (all 221
keys consumed), non-degenerate embeddings, and the above-chance decoding below (the functional
proof the token layout is correct). Wired as the frozen `labram` config (200-d CLS embedding →
shared head, computed once over all rows, leak-free) competing on the same folds as the
band-power `single:eeg` baseline and `raw_cnn`. 3×5 subject-held-out CV, error = 1−AUROC:

| task | labram | single:eeg (band-power) | best non-labram | verdict |
|---|---|---|---|---|
| eegmat_workload | 0.3373 (AUROC 0.663) | 0.3635 (0.636) | single:physiology/ECG 0.2565 | LaBraM **> band-power**; ECG still best |
| mumtaz_depression | **0.0392 (AUROC 0.961)** | 0.1112 (0.889) | xgboost 0.070 | LaBraM is the **single best config** |

Two honest positives, one caveat:

1. **The real EEG FM beats hand-crafted band-power features on both EEG cohorts** — modestly on
   workload (0.663 vs 0.636) and decisively on depression (0.961 vs 0.889). On `mumtaz_depression`
   (EEG-only, the cleanest FM test) LaBraM is the **best config overall**, beating the tuned GBM
   (0.93), the raw-CNN (0.84), and the learned CACMF fusion (0.79). A foundation model extracting
   more from resting/task EEG than band-power is exactly the expected win — and here it is real,
   frozen (linear-probe, no fine-tuning), and reproduced under subject-held-out CV.
2. **On workload, EEG is still not the best modality** — the ECG autonomic response
   (`single:physiology` 0.74 AUROC) beats every EEG method including LaBraM. The FM improves the
   *EEG* path; it does not overturn the task-level finding that autonomic signal dominates for
   arithmetic load.
3. **Fidelity caveat (stated up front, not after the fact):** both cohorts are **64 Hz**, so real
   content is ≤32 Hz, while LaBraM was pretrained on 200 Hz EEG (≤100 Hz). We resample 64→200 Hz to
   give LaBraM its patch structure, but not its bandwidth — so these are **fidelity-limited** wins;
   LaBraM would plausibly do *better* on native ≥200 Hz data. DEAP is worse still (decimated ~8 Hz
   effective — see Slice H) and is not a fair FM test; it is left for a full-rate re-export. So the
   result reads: the real EEG FM already beats band-power **even under-resourced on sampling rate**.

Reproduce: `python3 scripts/run_benchmark.py --tasks mumtaz_depression --repeats 3 --folds 5`
(config `labram`; needs the cached weights or `DVXR_LABRAM_ALLOW_DOWNLOAD=1`).

### The LLM-in-the-predictive-path (`rep:llm`): present, off by default, weakest

The proposal's title is "LLM-Based … Prediction," and the repo does implement an LLM **in the
predictive path** (`dvxr/llm/predictor.py`: VQ-codebook tokens → soft prompts into a frozen Qwen,
NeuroLM-style). It is **excluded from the default sweep** (enable with `--llm`) because on CPU it
runs Qwen2.5-0.5B with an *untrained* seeded projection (the LoRA/trainable variant is documented
but not run), and it lands among the **weakest** configs. The working predictions come from the
classical/linear shared head; the LLM's *validated* role is `dvxr/llm/insight.py`, which explains
a calibrated prediction and never makes one. So "LLM-Based Prediction" is, today, aspirational —
the LLM narrates reliably or predicts poorly untrained. Reported here, not hidden.

#### Slice C ablation: in-distribution soft tokens do NOT rescue the LLM predictor (honest negative)

We tested the most plausible cheap fix for *why* the frozen-LLM predictor is weak: its VQ→LLM
projection is an untrained random matrix, so the soft prompts are out-of-distribution for the
frozen Qwen. Slice C replaces it with an **in-distribution** projection — each VQ token becomes a
convex combination of the LLM's own real token embeddings (`DVXR_LLM_INDIST=1`,
`src/dvxr/llm/predictor.py`), so soft prompts live inside the model's embedding hull by
construction. Head-to-head on `eegmat_workload` (3×5 subject-held-out,
`scripts/run_llm_indist_ablation.py`):

| config | 1−AUROC | AUROC |
|---|---|---|
| single:eeg (band-power ref) | 0.3635 | 0.636 |
| rep:llm (random projection) | 0.4164 | 0.584 |
| rep:llm (in-distribution) | 0.4169 | 0.583 |

**In-distribution changes nothing (−0.1%);** both LLM variants lose to band-power by ~14.5%. The
bottleneck is not the projection *distribution* — it is that a **frozen** general-purpose 0.5B LLM
has no learned mapping from EEG/BCI VQ tokens to the clinical label; placing the tokens in the
right magnitude/subspace does not confer decoding ability the model never acquired. Closing this
would require actually *training* the read-in (LoRA / backprop-through-LLM), which is CPU-
infeasible here and flagged, not faked. So the honest verdict stands and is now *sharper*: the
LLM's validated role is insight/explanation, not prediction — and we ruled out the obvious
projection fix rather than leaving it as hand-waving. A clean negative on the third novel slice.

## Headline (clinical profile): the fused model does NOT win

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
| B2 | 6/7 labels were circular median-splits | **Fixed** — only real external labels (Non-EEG annotations, real future glucose, MIMIC mortality, DEAP self-report, CGMacros A1c strata); proxies excluded from the benchmark path. |
| B2a | `cgmacros_diabetes` feature/target leak | **Fixed** — the label is a real ADA threshold (`int(HbA1c ≥ 6.5)`), *not* a median-split, so it was not a B2 violation — but the defining glycemic labs (`hba1c`, `fasting_glucose`, `fasting_insulin`) were still emitted as `ehr` features, handing the model its own label. Now excluded via `DIABETES_EHR_DENYLIST` (`bench/tasks.py`) with an assertion guard. Effect: XGBoost floor AUROC 0.98→0.80 and `single:ehr` 0.89→0.58 (near chance), confirming the prior EHR "signal" was the leaked label. The honest task predicts A1c-defined status from CGM glucose dynamics + non-defining covariates; fusion still loses (single:cgm is the strongest opponent). |
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
