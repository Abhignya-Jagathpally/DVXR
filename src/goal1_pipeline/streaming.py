from __future__ import annotations

import pandas as pd

from .features import latest_stress_feature_row
from .schemas import validate_events


def predict_latest_stress(events: pd.DataFrame, trained, window_seconds: int = 30) -> dict:
    """Run a streaming-style prediction on the most recent window."""
    events = validate_events(events)
    latest = latest_stress_feature_row(events, window_seconds=window_seconds)
    aligned = latest.reindex(columns=trained.feature_columns, fill_value=0.0)
    raw_probability = trained.model.predict_proba(aligned)[0, 1]
    if trained.calibrator is not None:
        probability = float(trained.calibrator.predict([raw_probability])[0])
    else:
        probability = float(raw_probability)
    return {
        "window_start": str(latest.iloc[0]["window_start"]),
        "window_end": str(latest.iloc[0]["window_end"]),
        "stress_probability": probability,
        "predicted_label": "stress" if probability >= 0.5 else "non_stress",
    }