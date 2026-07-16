"""dvxr.sentinel — the curated NeuroGlycemic Sentinel PRODUCT surface (spec §23 separation).

This package is the single public entry point for the delivered product. It re-exports exactly the
components on the product path — ingestion contracts, the prospective target, the cohort synchrony
gate, the prediction service boundary, the Generate lifecycle, the reproducible snapshot, the policy
engine, grounded explanation, and patient-isolated retrieval — and NOTHING experimental. The three
overlapping surfaces the repo grew are now separated by namespace:

  * ``dvxr.sentinel``     — the product (this package).
  * ``dvxr.bench``        — depression/stress/public-cohort benchmarks (validated *components*).
  * ``dvxr.experiments``  — non-product research probes (frozen-LLM predictor, glassbox, generic fusion).

Rather than physically relocate ~470 import sites (which would risk the green-per-change discipline),
this is a re-export facade over the existing modules: importing from ``dvxr.sentinel`` gives the clean
product API, while every existing ``dvxr.*`` import keeps working. The honesty audit still forbids any
experimental predictor from reaching the product path.
"""
# --- domain contracts & schema ---
from dvxr.contracts import (  # noqa: F401
    ActionDecision,
    GenerateRequest,
    ModelEvidence,
    PatientSnapshot,
    RiskPrediction,
)
from dvxr.cohort import (  # noqa: F401
    GLUCOSE_FUSION_MODALITIES,
    SynchronyError,
    can_fuse,
    require_synchronized_for_fusion,
)
from dvxr.targets import (  # noqa: F401
    ExcursionThresholds,
    build_excursion_labels,
)

# --- prediction boundary (fused product abstains; CGM-only is single-modality) ---
from dvxr.prediction import (  # noqa: F401
    AbstainingPredictionService,
    CgmOnlyExcursionService,
    PredictionBundle,
    PredictionInputs,
    RiskPredictionService,
)

# --- Generate lifecycle, snapshot, policy, grounded explanation, retrieval ---
from dvxr.serve.orchestrate import ConsentError, generate_risk_report  # noqa: F401
from dvxr.serve.snapshot import build_patient_snapshot  # noqa: F401
from dvxr.serve.panels import build_report_panels  # noqa: F401
from dvxr.safety.policy import POLICY_ID, select_action  # noqa: F401
from dvxr.llm.grounded import grounded_explanation  # noqa: F401
from dvxr.retrieval import LocalKeywordTextIndex, RetrievalRepository  # noqa: F401


def create_product_api(**kwargs):
    """The Sentinel HTTP product. Delegates to the serving app; the product routes are the /v1
    lifecycle (POST /v1/risk-reports, GET /v1/predictions/{id}). Secure-by-default (see
    ``dvxr.serve.api.create_app``)."""
    from dvxr.serve.api import create_app
    return create_app(**kwargs)


#: The product's HTTP surface — only the Sentinel lifecycle routes are part of the product contract.
PRODUCT_ROUTES = ("/v1/risk-reports", "/v1/predictions/{prediction_id}")

__all__ = [
    "GenerateRequest", "RiskPrediction", "ModelEvidence", "ActionDecision", "PatientSnapshot",
    "GLUCOSE_FUSION_MODALITIES", "SynchronyError", "can_fuse", "require_synchronized_for_fusion",
    "ExcursionThresholds", "build_excursion_labels",
    "RiskPredictionService", "AbstainingPredictionService", "CgmOnlyExcursionService",
    "PredictionBundle", "PredictionInputs",
    "generate_risk_report", "ConsentError", "build_patient_snapshot", "build_report_panels",
    "select_action", "POLICY_ID", "grounded_explanation",
    "LocalKeywordTextIndex", "RetrievalRepository",
    "create_product_api", "PRODUCT_ROUTES",
]
