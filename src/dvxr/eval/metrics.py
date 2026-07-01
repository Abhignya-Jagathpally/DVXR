"""dvxr.eval.metrics — classification + forecasting metrics (ARCHITECTURE §A7).

Reuses dvxr.calibration for ECE and conformal interval coverage. AUROC/AUPRC are
NaN (not an error) when a split is single-class — reported honestly, never faked.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from dvxr.calibration import conformal_radius, expected_calibration_error, interval_coverage


def classification_metrics(y_true, prob_pos) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score, roc_auc_score,
    )
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_pos, dtype=float)
    pred = (p >= 0.5).astype(int)
    single_class = len(np.unique(y)) < 2
    return {
        "auroc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "auprc": float("nan") if single_class else float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "ece": float(expected_calibration_error(y, p)),
    }


def forecast_metrics(y_true, pred, cal_true, cal_pred, alpha: float = 0.10) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(y - p)))
    radius = conformal_radius(np.asarray(cal_true, float) - np.asarray(cal_pred, float),
                              alpha=alpha)
    cov = interval_coverage(y, p - radius, p + radius)
    return {"mae": mae, "interval_radius": float(radius), "coverage": float(cov)}
