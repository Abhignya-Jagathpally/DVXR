"""dvxr.eval — held-out splits, metrics, and the Goal-3 ablation harness."""
from .splits import subject_holdout_split  # noqa: F401
from .metrics import classification_metrics, forecast_metrics  # noqa: F401
from .ablation import (  # noqa: F401
    ablation_summary,
    make_synthetic_dataset,
    run_ablation,
)
