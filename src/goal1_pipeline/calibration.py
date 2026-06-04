from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class BinaryCalibrator:
    model: LogisticRegression | None

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float).reshape(-1, 1)
        if self.model is None:
            return probability.ravel()
        return self.model.predict_proba(probability)[:, 1]


def fit_platt_calibrator(probability: np.ndarray, truth: np.ndarray) -> BinaryCalibrator:
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float).reshape(-1, 1)
    if len(np.unique(truth)) < 2:
        return BinaryCalibrator(model=None)
    model = LogisticRegression(random_state=7)
    model.fit(probability, truth)
    return BinaryCalibrator(model=model)


def classification_calibration_metrics(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    truth = np.asarray(truth, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "ece": float(expected_calibration_error(truth, probability, bins=bins)),
    }


def expected_calibration_error(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probability)
    if total == 0:
        return float("nan")

    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right)
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        if not mask.any():
            continue
        confidence = probability[mask].mean()
        accuracy = truth[mask].mean()
        ece += mask.mean() * abs(confidence - accuracy)
    return float(ece)


def risk_band(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "watch"
    if probability < 0.75:
        return "elevated"
    return "high"


def add_risk_bands(frame: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["risk_band"] = out[probability_col].map(lambda x: risk_band(float(x)))
    return out


def conformal_radius(calibration_error: np.ndarray, alpha: float = 0.10) -> float:
    errors = np.abs(np.asarray(calibration_error, dtype=float))
    if len(errors) == 0:
        return float("nan")
    quantile = np.ceil((len(errors) + 1) * (1 - alpha)) / len(errors)
    quantile = min(1.0, max(0.0, quantile))
    return float(np.quantile(errors, quantile, method="higher"))


def interval_coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((truth >= lower) & (truth <= upper)))