"""dvxr.tasks.heads — multi-task heads (ARCHITECTURE §A5).

Six softmax/logistic classification heads (the six mental-health/clinical tasks) +
one conformal-interval forecasting head (glucose). Task names/proxies are REUSED from
``clinical_tasks.CLINICAL_TASKS`` — no invented labels. Calibration/conformal helpers
delegate to the existing ``dvxr.calibration`` utilities.
"""
from __future__ import annotations

from typing import List

import numpy as np

from dvxr.calibration import conformal_radius, fit_platt_calibrator, interval_coverage
from dvxr.clinical_tasks import CLINICAL_TASKS

# glucose is handled as the forecasting head; the other six are classification.
FORECAST_TASK = "glucose_instability"
CLASSIFICATION_TASKS: List[str] = [t.name for t in CLINICAL_TASKS if t.name != FORECAST_TASK]


def build_task_module(config, d_f: int, classification_tasks: List[str]):
    """Return a torch TaskHeads module (torch imported lazily)."""
    import torch
    from torch import nn

    class TaskHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_tasks = list(classification_tasks)
            self.cls = nn.ModuleDict({t: nn.Linear(d_f, 2) for t in classification_tasks})
            self.forecast = nn.Linear(d_f, 1)

        def forward(self, h):
            logits = {t: self.cls[t](h) for t in self.classification_tasks}
            yhat = self.forecast(h).squeeze(-1)
            return logits, yhat

        def probabilities(self, h):
            logits, _ = self.forward(h)
            return {t: torch.softmax(v, dim=1) for t, v in logits.items()}

    torch.manual_seed(config.seed)
    return TaskHeads()


# ---- calibration / conformal wrappers (numpy, reuse dvxr.calibration) ----

def calibrate_probabilities(probs_pos: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Platt-calibrate positive-class probabilities; returns values in [0, 1]."""
    cal = fit_platt_calibrator(np.asarray(probs_pos), np.asarray(truth))
    return np.clip(cal.predict(np.asarray(probs_pos)), 0.0, 1.0)


def forecast_interval_coverage(pred: np.ndarray, truth: np.ndarray,
                               cal_pred: np.ndarray, cal_truth: np.ndarray,
                               alpha: float = 0.10):
    """Split-conformal interval + coverage for the forecast head.

    Radius is set from calibration-set absolute residuals; coverage is measured on
    the (held-out) test residuals. Returns (radius, coverage, lower, upper).
    """
    cal_resid = np.asarray(cal_truth, dtype=float) - np.asarray(cal_pred, dtype=float)
    radius = conformal_radius(cal_resid, alpha=alpha)
    pred = np.asarray(pred, dtype=float)
    lower, upper = pred - radius, pred + radius
    cov = interval_coverage(np.asarray(truth, dtype=float), lower, upper)
    return radius, cov, lower, upper
