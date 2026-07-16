# Incident vs. persistence: the early-warning target (PR35 / P0-3)

An early-warning product must predict a **new** glucose excursion for a participant who is currently in
range — not merely notice that an already-hyperglycaemic participant is *still* high. The original
target labelled an anchor positive whenever **any** future sample in `(t, t+h]` was out of range, with
no requirement that the participant be in range at `t`. That is a **persistence detector**: a person
sitting at 220 mg/dL scores a "positive 60-minute excursion" just for staying high, inflating apparent
performance with the easiest possible cases.

## The taxonomy (`dvxr.targets.excursion`)

Every example now carries its state at `t` and an outcome class:

| state at `t` | future in `(t, t+h]` | `outcome_class` | in incident model? |
|---|---|---|---|
| in range | new excursion | `incident_excursion` (label 1) | ✅ |
| in range | stays in range | `no_excursion` (label 0) | ✅ |
| out of range | still out at `t+h` | `persistent_excursion` | ❌ excluded |
| out of range | back in range at `t+h` | `recovery` | ❌ excluded |

`build_excursion_labels(..., label_definition="incident")` reports **only in-range-at-`t` anchors**,
labelled by incident onset; anchors already out of range at `t` are censored
(`out_of_range_at_anchor`) so they can never masquerade as early warning. `label_definition="any"`
preserves the original (persistence-inclusive) definition for back-compat.

The deployed CGM risk artifact is now built on the **incident** target by default
(`scripts/build_cgm_artifact.py`, model version `cgm-only-incident/pilot-v1`); the target definition is
stamped into the model version for provenance.

## Per-horizon reporting (P0-4)

Pooling 30- and 60-minute examples into one AUROC hides that they are two different conditional tasks.
`dvxr.eval.glucose_ablation` now reports a **per-horizon** breakdown (AUROC/AUPRC/sensitivity/FA-rate)
alongside the pooled figure, so 30- and 60-minute performance are never conflated.

## Honesty

This is a *harder, more honest* target — the trivial persistence positives are removed, so reported
numbers reflect genuine early-warning skill, not staying-power. Every EEG/fused arm remains
`cannot_evaluate`; the fused product still abstains.
