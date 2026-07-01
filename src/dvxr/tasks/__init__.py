"""dvxr.tasks — multi-task heads, relative losses, and the joint training loop."""
from .heads import (  # noqa: F401
    CLASSIFICATION_TASKS,
    FORECAST_TASK,
    build_task_module,
    calibrate_probabilities,
    forecast_interval_coverage,
)
from .losses import (  # noqa: F401
    build_uncertainty_weighting,
    class_weighted_ce,
    huber_forecast,
    info_nce,
    mse_recon,
    total_loss,
)
from .model import build_multitask_model  # noqa: F401
from .train import (  # noqa: F401
    population_and_personalized_metrics,
    train_multitask,
)
