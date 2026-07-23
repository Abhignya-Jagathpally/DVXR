# Decisions Q&A — every choice, questioned and answered with evidence

This document answers, one by one, the questions posed for this round. Each answer links to a
committed experiment or artifact. The through-line: **claims are earned by measurement, and
where a simpler option wins we say so.**

### Why *this* model? How is it better than a simple MLP, CNN, decision tree, or linear model on the same data?

Honestly, on point accuracy **it isn't** — and we ran the experiment to find out. The
same-split model ladder (`docs/MODEL_JUSTIFICATION.md`, `outputs/_r2/glucose_model_ladder.csv`)
scores persistence/linear/decision-tree/random-forest/gradient-boosting/MLP/NeuroGlycemicNet
on the *same held-out patients*. **Gradient boosting is the best point forecaster at every
horizon** (RMSE 12.48 vs the deep net's 12.99 @30 min); MLP and even linear ridge edge it out
at 30 min too. The deep model earns its place only through **calibrated prediction intervals,
availability-aware abstention, and multimodal fusion** — not point accuracy. Dual-track
recommendation: ship gradient boosting for a point forecast; the deep model for calibrated,
abstaining, multimodal serving.

### If you give every model the same data, how do they compare?

That is exactly what the ladder does (same features, same split). Key finding: **every**
reasonable model beats persistence by ~25% (MASE 0.72–0.89). The **causal CGM-history
representation is the win, not the architecture.** Features > model class here.

### Why only this many epochs? Why not before / after?

Not a magic number — **early stopping decided it.** The run capped at 120 epochs, patience 15;
it **stopped at epoch 61** because validation NLL was best at **epoch 46** and did not improve
for 15 more (`neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1/training_losses.csv`,
`figures/training_loss.png`). Training longer would overfit; stopping earlier would underfit.

### What optimizer, and why?

AdamW with cosine LR warmup + decoupled weight decay (`weight_decay=1e-4`,
`lr_warmup_epochs`, `lr_min_ratio` in `config/cgmacros_glucose.json`). Rationale in
`docs/MODEL_JUSTIFICATION.md`: standard for this small late-fusion transformer-style net; the
decoupled decay regularizes without shrinking the residual-over-persistence head (the
inductive bias that makes it reliably beat persistence). A full optimizer sweep is a bounded
future ablation — the ladder already justifies the *architecture class*, which matters more.

### How is the glucose model actually good, not leaky?

RMSE 12.99 @30 min beats persistence (17.40) under **patient-clustered 95% CIs**
(`training_acceptance.json`: superiority gate passed). Leakage guards: causal features only
(past-only shift/rolling), and the 30-min target differs from current glucose by ~11 mg/dL on
average — a genuine forecast, not persistence relabeled. Deterministic reproduction confirmed
(max |diff| 5.7e-14). The forecast scatter (`figures/held_out_forecasts.png`) shows realistic
horizon-dependent widening, not a suspicious perfect line.

### Interoperable devices with interpretation — how is that addressed?

`docs/INTEROPERABILITY.md`: LSL stream contracts for Galea/EMOTIV/CGM/wearables in;
FHIR-Observation forecast export to an EHR; the rt-demo contract out. Each surface carries an
**interpretation payload** — risk band + grounded reason + explicit abstention + research-only
flag — so no device receives a bare number it could misread.

### How is explainability handled? Is there a better way?

`docs/EXPLAINABILITY.md`: three layers — signed per-feature attributions, tree/GBM native
importances (a faithfulness cross-check), and a grounded narrative that only restates frozen
values (`predicts:False`). A better/next step (documented, not a gap): exact SHAP on the GBM
point model — the models are already fit.

### How do you deal with latency and hallucinations?

- **Latency** (`outputs/latency_report.md`): every serving path <3 ms (direct 0.15, agentic
  2.31, RT frame ~0), glucose point models <1.5 ms — real-time safe.
- **Hallucinations**: the numeric body is frozen before any explanation; the LLM/narrative may
  only restate it. Enforced by `neuroglycemic/health_agent.py::_validate_llm_payload` and
  `tests/test_no_hallucinated_numbers.py` (asserts every number in the text appears in the
  body). The model **abstains** rather than guessing when inputs are missing.

### Are the mental-health heads on par with SOTA?

`docs/HEADS_SOTA.md`: depression **0.961/0.986 AUROC** (clears the LOSO/external published
bars), WESAD stress **0.955**, stress **0.892** — all subject-held-out, protocol-labeled. DEAP
anxiety/arousal is an honest negative: at chance even at full 128 Hz resolution
(`outputs/_r2/deap_fullrate_probe.md`), so the decimation hypothesis was refuted and the heavy
raw re-export was not chased.

### What else could be done? (honest roadmap)

- **Prospective external validation** of the glucose model (the one remaining release
  blocker) — validate on a held-out cohort/site before any clinical claim.
- **Raw-sequence deep model** for glucose (1-D CNN/transformer over the CGM series) — the one
  place a deep model might overtake gradient boosting, since the ladder shows tabular features
  already saturate the simple models.
- **SHAP** local explanations on the point model; **more CGM cohorts** (Shanghai, real
  DiaTrend via Synapse DUA) for cross-cohort generalization.
- **Cued, multi-session BCI** for the avatar skills — the current control signal is a
  single-subject EMOTIV engine label, demo-viable but not validated intent.

None of these change a current claim; they are the honest next steps, not hidden caveats.
