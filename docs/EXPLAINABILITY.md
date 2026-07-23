# Explainability — how every prediction is explained

Three complementary, machine-checked layers. None invents a number
(`tests/test_no_hallucinated_numbers.py` enforces it).

## 1. Signed feature attributions (per prediction)

Every response carries per-feature signed contributions from the shared linear-attribution
surface (`dvxr.serve.research_predict._linear_attribution` → `serve/explain.py`). Worked
example (`run_agentic_prediction`, glucose-instability outcome):

```
selected: probability 0.9998, risk band "high", evidence_status "simulation",
          validated_for_clinical_use false
contributions:
  time_above_range   +3.75  raises   (linear)
  cgm_std            +2.475 raises   (linear)
  hba1c             +1.575 raises   (linear)
  fasting_glucose    +0.84  raises   (linear)
```

The user sees *which* inputs pushed the estimate up or down and by how much — not a black box.
(`evidence_status: "simulation"` here because the committed research artifacts are absent in
this environment; the labelled illustrative model is used and says so.)

## 2. Model-intrinsic importances (from the ladder)

The gradient-boosting and tree models in the glucose ladder
(`neuroglycemic-sentinel/src/neuroglycemic/model_ladder.py`) expose native feature
importances, and the CGM lag/slope features they rank highest are the same causal-history
signals the deep model consumes — a cross-check that the explanation is faithful to the data,
not post-hoc storytelling.

## 3. Grounded natural-language explanation (explains, never predicts)

The explanation node (`dvxr/serve/agents/nodes.py::_grounded_explanation`, the seat where an
LLM/`HealthAgent` can be plugged in) restates only values already frozen in the body:

> "Research-stage estimate for glucose_instability: probability 0.9998 (risk band high). Not
> validated for clinical use; this explains the model output and makes no independent
> prediction."

`explanation.predicts == False`, and `tests/test_no_hallucinated_numbers.py` asserts every
numeric token in the text already appears in the prediction body. The LLM is confined to this
leaf and gated by the numeric-grounding validator
(`neuroglycemic/health_agent.py::_validate_llm_payload`) — see docs/INTEROPERABILITY.md and
the hallucination section of docs/DECISIONS_QA.md.

## Is there a better way?

For richer local attributions on the tree/GBM point model, SHAP values are a natural next
step (the models are already fit); we use exact linear attributions + native importances here
because they are dependency-light and audit-stable. Documented as a bounded enhancement, not a
gap that changes any claim.
