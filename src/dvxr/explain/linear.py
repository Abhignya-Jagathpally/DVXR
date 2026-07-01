from __future__ import annotations

import pandas as pd


def top_linear_contributors(trained, top_n: int = 10) -> pd.DataFrame:
    """Return top coefficient-based explanations for linear models."""
    estimator = trained.model.steps[-1][1]
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        values = coef[0] if coef.ndim > 1 else coef
    else:
        raise ValueError("Estimator does not expose coefficients")

    frame = pd.DataFrame(
        {
            "feature": trained.feature_columns,
            "weight": values,
            "abs_weight": abs(values),
        }
    )
    return frame.sort_values("abs_weight", ascending=False).head(top_n).drop(columns=["abs_weight"])
