"""dvxr.prediction — the predictive boundary (Gate 3).

The Generate lifecycle depends on a ``RiskPredictionService`` Protocol, never on a concrete model, so
the honest research-stage default (``AbstainingPredictionService``) and an in-scope single-modality
baseline (``CgmOnlyExcursionService``) are interchangeable. The FUSED EEG+CGM+wearable claim has no
synchronized dataset, so it stays behind the abstaining default; only a single-cohort CGM-only forecast
ever returns a number, and it is labelled ``modality_scope="cgm_only"`` — never the fused headline.
"""
from dvxr.prediction.service import (  # noqa: F401
    AbstainingPredictionService,
    AdequacyConfig,
    CgmOnlyExcursionService,
    PredictionBundle,
    PredictionInputs,
    RiskPredictionService,
    ScientificValidityError,
    build_cgm_feature_matrix,
    cgm_history_features,
)
