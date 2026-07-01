"""dvxr.fusion — five fusion strategies + three aggregation baselines + CACMFModel."""
from .aggregate import (  # noqa: F401
    AGGREGATORS,
    confidence_weighted,
    ensemble_avg,
    normalized_entropy_confidence,
    weighted_late,
)
from .strategies import FusionOutput, get_fusion_strategy  # noqa: F401
from .model import build_cacmf_model  # noqa: F401
