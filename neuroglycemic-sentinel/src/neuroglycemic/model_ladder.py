"""Empirical model-selection ladder for glucose forecasting.

Answers "why this model?" by running, on the **same patient-disjoint split** the neural
model used, a ladder from trivial to complex — persistence, linear, decision tree, random
forest, gradient boosting, MLP — and reporting per-horizon RMSE/MAE + MASE-vs-persistence.
The deep ``NeuroGlycemicNet`` earns its complexity only if it beats the best simple model
on the *same* held-out patients; if a simpler model wins, that is reported and shipped.

Uses the exact train/test patient partition recorded in the neural run's
``patient_split.csv`` so the comparison is apples-to-apples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cgmacros_data import CGMACROS_CGM_FEATURES, CGMACROS_EVENT_FEATURES
from .neural_dataset import target_column

FEATURE_COLUMNS = [*CGMACROS_CGM_FEATURES, *CGMACROS_EVENT_FEATURES]


@dataclass(frozen=True)
class LadderResult:
    table: pd.DataFrame

    def to_markdown(self) -> str:
        lines = ["# Glucose model-selection ladder (same patient-disjoint split)\n"]
        lines.append(
            "Per-horizon test RMSE/MAE (mg/dL) and MASE vs persistence "
            "(MAE_model / MAE_persistence; <1 beats persistence). "
            "Same held-out patients as the neural model.\n"
        )
        lines.append(self.table.to_markdown(index=False))
        return "\n".join(lines)


def _load_split(run_dir: Path) -> tuple[list[str], list[str]]:
    split = pd.read_csv(run_dir / "patient_split.csv")
    col = "split" if "split" in split.columns else split.columns[-1]
    id_col = "patient_id" if "patient_id" in split.columns else split.columns[0]
    # Fit classical models on every non-test patient (they need no early-stopping
    # holdout); the test partition is held out identically to the neural model.
    is_test = split[col].astype(str).eq("test")
    test = split.loc[is_test, id_col].astype(str).tolist()
    non_test = split.loc[~is_test, id_col].astype(str).tolist()
    return non_test, test


def _build_models():
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    return {
        "linear_ridge": lambda: Ridge(alpha=1.0),
        "decision_tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=1, random_state=0
        ),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=0
        ),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=0
        ),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_glucose_ladder(
    windows_path: Path,
    run_dir: Path,
    horizons_minutes: tuple[int, ...] = (30, 60, 90, 120),
) -> LadderResult:
    """Fit the ladder on the neural run's train patients; score on its test patients."""
    frame = pd.read_csv(windows_path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    train_ids, test_ids = _load_split(run_dir)
    train = frame[frame["patient_id"].isin(train_ids)]
    test = frame[frame["patient_id"].isin(test_ids)]
    if train.empty or test.empty:
        raise ValueError("Split produced an empty train or test partition.")

    x_train_raw = train[FEATURE_COLUMNS]
    x_test_raw = test[FEATURE_COLUMNS]
    medians = x_train_raw.median(numeric_only=True)
    x_train = x_train_raw.fillna(medians).to_numpy(float)
    x_test = x_test_raw.fillna(medians).to_numpy(float)

    # Load the deep model's per-horizon metrics (same test patients) if present.
    proposed = {}
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.is_file():
        by_h = json.loads(metrics_path.read_text()).get("by_horizon", {})
        for h, v in by_h.items():
            reg = v.get("neural_regression_metrics", v)
            proposed[int(h)] = (reg.get("rmse_mg_dl"), reg.get("mae_mg_dl"))

    models = _build_models()
    rows = []
    for horizon in horizons_minutes:
        target = target_column(horizon)
        if target not in frame.columns:
            continue
        tr_mask = train[target].notna().to_numpy()
        te_mask = test[target].notna().to_numpy()
        y_tr = train[target].to_numpy(float)[tr_mask]
        y_te = test[target].to_numpy(float)[te_mask]
        current_te = test["cgm_current_mg_dl"].to_numpy(float)[te_mask]

        # persistence baseline: future == current
        persist_rmse, persist_mae = _rmse(y_te, current_te), _mae(y_te, current_te)
        rows.append(_row("persistence", horizon, persist_rmse, persist_mae, persist_mae))

        for name, factory in models.items():
            model = factory()
            model.fit(x_train[tr_mask], y_tr)
            pred = model.predict(x_test[te_mask])
            rows.append(
                _row(name, horizon, _rmse(y_te, pred), _mae(y_te, pred),
                     _mae(y_te, pred) / persist_mae if persist_mae else float("nan"))
            )

        if horizon in proposed and proposed[horizon][0] is not None:
            p_rmse, p_mae = proposed[horizon]
            rows.append(
                _row("neuroglycemic_net", horizon, p_rmse, p_mae,
                     p_mae / persist_mae if persist_mae else float("nan"))
            )

    table = pd.DataFrame(rows)
    return LadderResult(table=table)


def _row(model: str, horizon: int, rmse: float, mae: float, mase: float) -> dict:
    return {
        "model": model,
        "horizon_minutes": int(horizon),
        "rmse_mg_dl": round(float(rmse), 3),
        "mae_mg_dl": round(float(mae), 3),
        "mase_vs_persistence": round(float(mase), 3),
    }
