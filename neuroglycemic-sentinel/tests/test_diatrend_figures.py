"""The DiaTrend-style overview figures render from recorded artifacts only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neuroglycemic.diatrend_figures import (
    DEFAULT_GLUCOSE_COLUMN,
    build_overview_suite,
    save_time_in_range_figure,
)


def _windows(n_patients: int = 3, per_patient: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(1, n_patients + 1):
        base = pd.Timestamp("2020-02-13T00:00:00Z")
        for i in range(per_patient):
            rows.append(
                {
                    "patient_id": patient,
                    "anchor_time": base + pd.Timedelta(minutes=15 * i),
                    DEFAULT_GLUCOSE_COLUMN: float(
                        110 + 30 * rng.standard_normal() + 5 * patient
                    ),
                }
            )
    return pd.DataFrame(rows)


def _audit(n_patients: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": list(range(1, n_patients + 1)),
            "aligned_windows": [40] * n_patients,
            "cgm_rows": [500] * n_patients,
            "start_utc": ["2020-02-13T00:00:00Z"] * n_patients,
            "stop_utc": ["2020-02-21T00:00:00Z"] * n_patients,
        }
    )


def test_build_overview_suite_writes_all_five(tmp_path):
    outputs = build_overview_suite(
        _windows(), _audit(), tmp_path, cohort_label="Test cohort"
    )
    assert set(outputs) == {
        "cgm_traces",
        "time_in_range",
        "glucose_distribution",
        "data_availability",
        "cohort_summary",
    }
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_time_in_range_requires_glucose_column(tmp_path):
    frame = _windows().drop(columns=[DEFAULT_GLUCOSE_COLUMN])
    with pytest.raises(ValueError):
        save_time_in_range_figure(frame, tmp_path / "tir.png")
