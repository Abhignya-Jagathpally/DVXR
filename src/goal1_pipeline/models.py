from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import (
    BinaryCalibrator,
    add_risk_bands,
    classification_calibration_metrics,
    conformal_radius,
    fit_platt_calibrator,
    interval_coverage,
)
from .features import feature_columns


@dataclass
class TrainedModel:
    model: Pipeline
    feature_columns: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    calibrator: BinaryCalibrator | None = None
    interval_radius: float | None = None


def train_stress_classifier(windows: pd.DataFrame) -> TrainedModel:
    return train_binary_classifier(
        windows,
        positive_label="stress",
        negative_label="non_stress",
        probability_col="stress_probability",
        raw_probability_col="raw_stress_probability",
    )


def train_arousal_classifier(windows: pd.DataFrame) -> TrainedModel:
    return train_binary_classifier(
        windows,
        positive_label="high_arousal",
        negative_label="low_arousal",
        probability_col="high_arousal_probability",
        raw_probability_col="raw_high_arousal_probability",
    )


def train_binary_classifier(
    windows: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    probability_col: str,
    raw_probability_col: str,
) -> TrainedModel:
    features = feature_columns(windows)
    y = (windows["target"] == positive_label).astype(int)
    groups = windows["subject_id"].astype(str)
    train_idx, calibration_idx, test_idx = _group_train_calibration_test_split(windows, groups)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)),
        ]
    )
    model.fit(windows.iloc[train_idx][features], y.iloc[train_idx])

    calibration_probability = model.predict_proba(windows.iloc[calibration_idx][features])[:, 1]
    calibrator = fit_platt_calibrator(calibration_probability, y.iloc[calibration_idx].to_numpy())

    raw_probability = model.predict_proba(windows.iloc[test_idx][features])[:, 1]
    probability = calibrator.predict(raw_probability)
    predicted = (probability >= 0.5).astype(int)
    truth = y.iloc[test_idx].to_numpy()

    calibration_metrics = classification_calibration_metrics(truth, probability)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "auroc": _safe_auroc(truth, probability),
        "brier": calibration_metrics["brier"],
        "log_loss": calibration_metrics["log_loss"],
        "ece": calibration_metrics["ece"],
        "test_windows": float(len(test_idx)),
    }
    predictions = windows.iloc[test_idx][["subject_id", "session_id", "window_start", "window_end", "target"]].copy()
    predictions[raw_probability_col] = raw_probability
    predictions[probability_col] = probability
    predictions["predicted_label"] = np.where(predicted == 1, positive_label, negative_label)
    predictions = add_risk_bands(predictions, probability_col)
    return TrainedModel(model=model, feature_columns=features, metrics=metrics, predictions=predictions, calibrator=calibrator)


def train_glucose_forecaster(glucose_frame: pd.DataFrame) -> TrainedModel:
    features = [col for col in glucose_frame.columns if col.startswith("glucose_") or col == "time_in_range_fraction"]
    features = [col for col in features if col != "target_glucose"]
    groups = glucose_frame["subject_id"].astype(str)
    train_idx, calibration_idx, test_idx = _group_train_calibration_test_split(glucose_frame, groups)

    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(glucose_frame.iloc[train_idx][features], glucose_frame.iloc[train_idx]["target_glucose"])

    calibration_prediction = model.predict(glucose_frame.iloc[calibration_idx][features])
    calibration_truth = glucose_frame.iloc[calibration_idx]["target_glucose"].to_numpy(dtype=float)
    interval_radius = conformal_radius(calibration_truth - calibration_prediction, alpha=0.10)

    prediction = model.predict(glucose_frame.iloc[test_idx][features])
    truth = glucose_frame.iloc[test_idx]["target_glucose"].to_numpy(dtype=float)
    lower = prediction - interval_radius
    upper = prediction + interval_radius
    metrics = {
        "mae_mg_dl": float(mean_absolute_error(truth, prediction)),
        "rmse_mg_dl": float(np.sqrt(mean_squared_error(truth, prediction))),
        "interval_radius_mg_dl": float(interval_radius),
        "interval_coverage": interval_coverage(truth, lower, upper),
        "test_rows": float(len(test_idx)),
    }
    predictions = glucose_frame.iloc[test_idx][["subject_id", "session_id", "timestamp_utc", "target_glucose"]].copy()
    predictions["predicted_glucose"] = prediction
    predictions["lower_90"] = lower
    predictions["upper_90"] = upper
    predictions["residual_mg_dl"] = truth - prediction
    predictions["in_interval"] = (truth >= lower) & (truth <= upper)
    return TrainedModel(
        model=model,
        feature_columns=features,
        metrics=metrics,
        predictions=predictions,
        interval_radius=float(interval_radius),
    )


def _group_train_calibration_test_split(
    frame: pd.DataFrame,
    groups: pd.Series,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/calibration/test, keeping subjects disjoint when possible.

    With three or more subjects the split is by whole subject (no subject appears
    in more than one fold), which is the honest setting for personalized health
    models. With fewer subjects we fall back to a deterministic row-level split so
    the demo still runs end to end.
    """
    group_values = np.asarray(groups).astype(str)
    positions = np.arange(len(frame))
    unique = np.array(sorted(np.unique(group_values)))
    rng = np.random.default_rng(seed)

    if len(unique) >= 3:
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = min(max(1, n // 5), n - 2)
        n_cal = min(max(1, n // 5), n - 1 - n_test)
        test_groups = shuffled[:n_test]
        cal_groups = shuffled[n_test : n_test + n_cal]
        train_groups = shuffled[n_test + n_cal :]
        train_idx = positions[np.isin(group_values, train_groups)]
        calibration_idx = positions[np.isin(group_values, cal_groups)]
        test_idx = positions[np.isin(group_values, test_groups)]
        return train_idx, calibration_idx, test_idx

    shuffled_positions = positions.copy()
    rng.shuffle(shuffled_positions)
    n = len(shuffled_positions)
    n_train = max(1, int(n * 0.6))
    n_cal = max(1, int(n * 0.2))
    train_idx = shuffled_positions[:n_train]
    calibration_idx = shuffled_positions[n_train : n_train + n_cal]
    test_idx = shuffled_positions[n_train + n_cal :]
    if len(test_idx) == 0:
        test_idx = shuffled_positions[-1:]
    return train_idx, calibration_idx, test_idx


def _safe_auroc(truth: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=int)
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(roc_auc_score(truth, probability))