"""CGMacros ingestion builds causal, leak-free CGM-autoregressive windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neuroglycemic.cgmacros_data import (
    CGMACROS_CGM_FEATURES,
    CGMACROS_EVENT_FEATURES,
    CGMacrosBuildConfig,
    build_cgmacros_patient_windows,
)
from src.neuroglycemic.neural_dataset import target_column


def _synthetic_subject(minutes: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2020-05-01T00:00:00Z")
    times = [start + pd.Timedelta(minutes=i) for i in range(minutes)]
    # a smooth glucose curve with a meal bump so lags/targets differ from current
    glucose = 110 + 30 * np.sin(np.arange(minutes) / 60.0) + np.linspace(0, 10, minutes)
    cgm = pd.DataFrame({"date": times, "mg/dL": glucose})
    meals = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(minutes=120)],
            "carbohydrate_g": [45.0], "protein_g": [20.0],
            "fat_g": [15.0], "fiber_g": [5.0], "calories": [400.0],
        }
    )
    return cgm, meals


def test_windows_are_causal_and_have_all_features():
    cgm, meals = _synthetic_subject()
    cfg = CGMacrosBuildConfig(source_timezone="UTC", horizons_minutes=(30, 60))
    windows = build_cgmacros_patient_windows("CGMacros-999", cgm, meals, config=cfg)
    assert not windows.empty
    for col in (*CGMACROS_CGM_FEATURES, *CGMACROS_EVENT_FEATURES):
        assert col in windows.columns
    # cgm_current is present and finite on every eligible window
    assert windows["cgm_current_mg_dl"].notna().all()


def test_no_target_leakage_into_current():
    cgm, meals = _synthetic_subject()
    cfg = CGMacrosBuildConfig(source_timezone="UTC", horizons_minutes=(30, 60))
    windows = build_cgmacros_patient_windows("CGMacros-999", cgm, meals, config=cfg)
    # a genuine forecast: 30-min target should differ from current glucose on average
    diff = (windows["cgm_current_mg_dl"] - windows[target_column(30)]).abs().mean()
    assert diff > 1.0
    # every target timestamp is strictly after the anchor (no past/simultaneous leakage)
    tt = pd.to_datetime(windows[f"target_glucose_30m_time"], utc=True)
    anchor = pd.to_datetime(windows["anchor_time"], utc=True)
    assert (tt > anchor).all()


def test_empty_on_out_of_range_glucose():
    times = [pd.Timestamp("2020-05-01T00:00:00Z") + pd.Timedelta(minutes=i) for i in range(120)]
    cgm = pd.DataFrame({"date": times, "mg/dL": [5.0] * 120})  # all below validity floor
    cfg = CGMacrosBuildConfig(source_timezone="UTC")
    windows = build_cgmacros_patient_windows("CGMacros-000", cgm, pd.DataFrame(), config=cfg)
    assert windows.empty


def test_config_rejects_bad_timezone():
    with pytest.raises(ValueError):
        CGMacrosBuildConfig(source_timezone="   ")
