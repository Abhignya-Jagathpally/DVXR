"""CGMacros ingestion and causal CGM-autoregressive window construction.

CGMacros (PhysioNet) ships one CSV per participant with a dense (~1-min) CGM trace
(Libre + Dexcom), heart rate, and logged meals (carbohydrate/protein/fat/fiber/calories).
This adapter produces the same same-patient aligned-window contract the neural model
consumes, using the **identical causal CGM feature block** as the DiaTrend adapter
(``DIATREND_CGM_FEATURES``) so CGM-autoregressive forecasting is directly comparable
across cohorts. The event modality here is **meals** (no insulin is recorded in CGMacros).

No interpolation is used to create labels; every CGM history feature is past-only
(``glucose.shift`` / trailing rolling windows) and every target is a real future CGM
observation matched within tolerance — no future leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .diatrend import (
    DIATREND_CGM_FEATURES,
    _localize_timestamp,
    _nearest_future_cgm,
    _rolling_slope,
    _sum_events_on_grid,
)
from .neural_dataset import target_column, target_time_column

# Identical causal CGM features as DiaTrend — the autoregressive core is cohort-agnostic.
CGMACROS_CGM_FEATURES = DIATREND_CGM_FEATURES

# CGMacros logs meals (no insulin); the event modality carries nutrition macros.
CGMACROS_EVENT_FEATURES = (
    "events_carbohydrate_g_30m",
    "events_carbohydrate_g_60m",
    "events_carbohydrate_g_120m",
    "events_protein_g_60m",
    "events_fat_g_60m",
    "events_fiber_g_60m",
    "events_calories_60m",
    "events_minutes_since_meal",
)

_MEAL_COLUMNS = {
    "Carbs": "carbohydrate_g",
    "Protein": "protein_g",
    "Fat": "fat_g",
    "Fiber": "fiber_g",
    "Calories": "calories",
}


@dataclass(frozen=True)
class CGMacrosBuildConfig:
    source_timezone: str
    glucose_source: str = "libre"  # 'libre' (100% coverage) or 'dexcom'
    horizons_minutes: tuple[int, ...] = (30, 60, 90, 120)
    grid_minutes: int = 5
    history_minutes: int = 120
    anchor_stride_minutes: int = 15
    minimum_history_coverage: float = 0.75
    glucose_min_mg_dl: float = 20.0
    glucose_max_mg_dl: float = 600.0
    cohort_id: str = "cgmacros"

    def __post_init__(self) -> None:
        if not self.source_timezone.strip():
            raise ValueError("source_timezone is required; CGMacros timestamps are naive.")
        if self.glucose_source not in {"libre", "dexcom"}:
            raise ValueError("glucose_source must be 'libre' or 'dexcom'.")
        if not self.horizons_minutes or any(v <= 0 for v in self.horizons_minutes):
            raise ValueError("At least one positive forecast horizon is required.")
        if self.grid_minutes <= 0 or self.history_minutes < self.grid_minutes:
            raise ValueError("The grid and history durations are invalid.")
        if self.anchor_stride_minutes <= 0 or self.anchor_stride_minutes % self.grid_minutes:
            raise ValueError("anchor_stride_minutes must be a positive grid multiple.")
        if not 0 < self.minimum_history_coverage <= 1:
            raise ValueError("minimum_history_coverage must be in (0, 1].")
        if self.glucose_min_mg_dl <= 0 or self.glucose_min_mg_dl >= self.glucose_max_mg_dl:
            raise ValueError("Glucose validity bounds must be positive and ordered.")


@dataclass(frozen=True)
class CGMacrosPatientAudit:
    patient_id: str
    source_file: str
    raw_cgm_rows: int
    valid_cgm_rows: int
    meal_rows: int
    aligned_windows: int
    cgm_start_utc: str | None
    cgm_stop_utc: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def read_cgmacros_subject(
    path: Path, *, source_timezone: str, glucose_source: str = "libre"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one CGMacros CSV into (cgm[date, mg/dL], meals[date, macros])."""
    path = Path(path)
    frame = pd.read_csv(path)
    if "Timestamp" not in frame.columns:
        raise ValueError(f"{path.name} is missing the Timestamp column.")
    glucose_column = "Libre GL" if glucose_source == "libre" else "Dexcom GL"
    if glucose_column not in frame.columns:
        raise ValueError(f"{path.name} is missing the {glucose_column} column.")
    timestamps = _localize_timestamp(frame["Timestamp"], source_timezone, name="Timestamp")

    cgm = pd.DataFrame(
        {"date": timestamps, "mg/dL": pd.to_numeric(frame[glucose_column], errors="coerce")}
    ).dropna(subset=["mg/dL"])

    meal_frame = pd.DataFrame({"date": timestamps})
    has_meal = pd.Series(False, index=frame.index)
    for source_col, canonical in _MEAL_COLUMNS.items():
        if source_col in frame.columns:
            values = pd.to_numeric(frame[source_col], errors="coerce")
            meal_frame[canonical] = values
            has_meal = has_meal | values.notna()
        else:
            meal_frame[canonical] = np.nan
    meals = meal_frame.loc[has_meal.to_numpy()].reset_index(drop=True)
    return cgm, meals


def build_cgmacros_patient_windows(
    patient_id: str,
    cgm: pd.DataFrame,
    meals: pd.DataFrame,
    *,
    config: CGMacrosBuildConfig,
) -> pd.DataFrame:
    """Causal, multi-horizon CGM-autoregressive windows for one CGMacros participant."""
    raw = cgm.dropna(subset=["date", "mg/dL"]).copy()
    raw = raw.loc[raw["mg/dL"].between(config.glucose_min_mg_dl, config.glucose_max_mg_dl)]
    if raw.empty:
        return pd.DataFrame()
    raw = raw.sort_values("date")
    raw["grid_time"] = raw["date"].dt.ceil(f"{config.grid_minutes}min")
    glucose = raw.groupby("grid_time")["mg/dL"].median().sort_index()
    glucose_available_time = raw.groupby("grid_time")["date"].max().sort_index()
    grid_index = pd.date_range(
        glucose.index.min(), glucose.index.max(), freq=f"{config.grid_minutes}min"
    )
    glucose = glucose.reindex(grid_index)
    observed = glucose.notna().astype(float)
    history_periods = config.history_minutes // config.grid_minutes + 1
    history_coverage = observed.rolling(history_periods, min_periods=history_periods).mean()

    # --- CGM history features (identical construction to DiaTrend) ---
    feature = pd.DataFrame(index=grid_index)
    feature["cgm_current_mg_dl"] = glucose
    for minutes in (5, 15, 30, 60, 120):
        feature[f"cgm_lag_{minutes}m_mg_dl"] = glucose.shift(minutes // config.grid_minutes)
    for minutes in (5, 15, 30):
        feature[f"cgm_delta_{minutes}m_mg_dl"] = glucose - glucose.shift(
            minutes // config.grid_minutes
        )
    for minutes in (30, 60, 120):
        periods = minutes // config.grid_minutes + 1
        feature[f"cgm_mean_{minutes}m_mg_dl"] = glucose.rolling(periods, min_periods=periods).mean()
        feature[f"cgm_sd_{minutes}m_mg_dl"] = glucose.rolling(periods, min_periods=periods).std(ddof=0)
    for minutes in (30, 60):
        periods = minutes // config.grid_minutes + 1
        feature[f"cgm_slope_{minutes}m_mg_dl_per_min"] = _rolling_slope(
            glucose, periods, config.grid_minutes
        )

    # --- Meal event features (rolling causal macro sums) ---
    carbs = _sum_events_on_grid(meals, "carbohydrate_g", grid_index, config.grid_minutes)
    protein = _sum_events_on_grid(meals, "protein_g", grid_index, config.grid_minutes)
    fat = _sum_events_on_grid(meals, "fat_g", grid_index, config.grid_minutes)
    fiber = _sum_events_on_grid(meals, "fiber_g", grid_index, config.grid_minutes)
    calories = _sum_events_on_grid(meals, "calories", grid_index, config.grid_minutes)
    for minutes in (30, 60, 120):
        periods = minutes // config.grid_minutes + 1
        feature[f"events_carbohydrate_g_{minutes}m"] = carbs.rolling(periods, min_periods=1).sum()
    feature["events_protein_g_60m"] = protein.rolling(60 // config.grid_minutes + 1, min_periods=1).sum()
    feature["events_fat_g_60m"] = fat.rolling(60 // config.grid_minutes + 1, min_periods=1).sum()
    feature["events_fiber_g_60m"] = fiber.rolling(60 // config.grid_minutes + 1, min_periods=1).sum()
    feature["events_calories_60m"] = calories.rolling(60 // config.grid_minutes + 1, min_periods=1).sum()

    meal_times = pd.Series(pd.NaT, index=grid_index, dtype="datetime64[ns, UTC]")
    if not meals.empty:
        logged = meals.dropna(subset=["date"]).copy()
        logged["grid_time"] = logged["date"].dt.ceil(f"{config.grid_minutes}min")
        meal_times.update(logged.groupby("grid_time")["date"].max().reindex(grid_index))
    last_meal = meal_times.ffill()
    minutes_since = (pd.Series(grid_index, index=grid_index) - last_meal).dt.total_seconds() / 60.0
    feature["events_minutes_since_meal"] = minutes_since.where(minutes_since.le(240.0))

    # An unobserved meal window is unknown, not zero. The event expert is available
    # only where at least one recorded macro contributes to the causal lookback.
    event_frame = feature[list(CGMACROS_EVENT_FEATURES)]
    event_available = event_frame.notna().any(axis=1)
    feature.loc[~event_available, list(CGMACROS_EVENT_FEATURES)] = np.nan

    event_available_time = pd.Series(pd.NaT, index=grid_index, dtype="datetime64[ns, UTC]")
    if not meals.empty:
        latest = pd.merge_asof(
            pd.DataFrame({"grid_time": grid_index}),
            meals[["date"]].sort_values("date").rename(columns={"date": "meal_available_time"}),
            left_on="grid_time",
            right_on="meal_available_time",
            direction="backward",
            tolerance=pd.Timedelta(minutes=240),
        )["meal_available_time"]
        event_available_time.update(pd.Series(latest.to_numpy(), index=grid_index))
    event_available_time = event_available_time.where(event_available)

    result = feature.copy()
    result.insert(0, "anchor_time", grid_index)
    result.insert(0, "cohort_id", config.cohort_id)
    result.insert(0, "patient_id", str(patient_id))
    result["session_id"] = f"{config.cohort_id}:{patient_id}"
    result["cgm_available"] = glucose.notna().to_numpy()
    result["cgm_quality"] = history_coverage.fillna(0.0).to_numpy()
    cgm_times = glucose_available_time.reindex(grid_index)
    result["cgm_staleness_minutes"] = (
        (pd.Series(grid_index, index=grid_index) - cgm_times).dt.total_seconds().div(60.0).to_numpy()
    )
    result["cgm_clock_uncertainty_ms"] = 0.0
    result["cgm_patient_id"] = str(patient_id)
    result["cgm_cohort_id"] = config.cohort_id
    result["cgm_anchor_time"] = result["anchor_time"]
    result["cgm_available_time"] = cgm_times.to_numpy()
    result["events_available"] = event_available.to_numpy()
    result["events_quality"] = event_frame.notna().mean(axis=1).where(event_available, 0.0).to_numpy()
    result["events_staleness_minutes"] = (
        (pd.Series(grid_index, index=grid_index) - event_available_time)
        .dt.total_seconds().div(60.0).where(event_available, 0.0).to_numpy()
    )
    result["events_clock_uncertainty_ms"] = 0.0
    result["events_patient_id"] = np.where(event_available, str(patient_id), None)
    result["events_cohort_id"] = np.where(event_available, config.cohort_id, None)
    result["events_anchor_time"] = result["anchor_time"].where(event_available)
    result["events_available_time"] = event_available_time.to_numpy()

    for horizon in config.horizons_minutes:
        target, actual_time = _nearest_future_cgm(
            grid_index, raw, horizon_minutes=horizon,
            tolerance_minutes=float(config.grid_minutes),
        )
        result[target_column(horizon)] = target.to_numpy()
        result[target_time_column(horizon)] = actual_time.to_numpy()

    stride = config.anchor_stride_minutes // config.grid_minutes
    eligible = (
        result["cgm_available"]
        & result["cgm_quality"].ge(config.minimum_history_coverage)
        & (np.arange(len(result)) % stride == 0)
    )
    target_names = [target_column(v) for v in config.horizons_minutes]
    eligible &= result[target_names].notna().any(axis=1)
    result = result.loc[eligible].reset_index(drop=True)

    ordered = [
        "patient_id", "cohort_id", "session_id", "anchor_time",
        *CGMACROS_CGM_FEATURES, *CGMACROS_EVENT_FEATURES,
        *[
            value
            for modality in ("cgm", "events")
            for value in (
                f"{modality}_available", f"{modality}_quality",
                f"{modality}_staleness_minutes", f"{modality}_clock_uncertainty_ms",
                f"{modality}_patient_id", f"{modality}_cohort_id",
                f"{modality}_anchor_time", f"{modality}_available_time",
            )
        ],
        *[
            value
            for horizon in config.horizons_minutes
            for value in (target_column(horizon), target_time_column(horizon))
        ],
    ]
    return result[ordered]


def discover_cgmacros_subjects(source_directory: Path) -> list[Path]:
    """Find each ``CGMacros-0NN/CGMacros-0NN.csv`` participant file."""
    source_directory = Path(source_directory)
    if not source_directory.is_dir():
        raise NotADirectoryError(source_directory)
    files = sorted(
        subdir / f"{subdir.name}.csv"
        for subdir in source_directory.iterdir()
        if subdir.is_dir() and subdir.name.startswith("CGMacros-")
        and (subdir / f"{subdir.name}.csv").is_file()
    )
    if not files:
        raise FileNotFoundError(f"No CGMacros participant CSVs found in {source_directory}.")
    return files


def build_cgmacros_dataset(
    subjects: Iterable[Path], *, config: CGMacrosBuildConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path_value in subjects:
        path = Path(path_value)
        patient_id = path.stem  # e.g. CGMacros-001
        cgm, meals = read_cgmacros_subject(
            path, source_timezone=config.source_timezone, glucose_source=config.glucose_source
        )
        valid = cgm["mg/dL"].between(config.glucose_min_mg_dl, config.glucose_max_mg_dl)
        patient_windows = build_cgmacros_patient_windows(patient_id, cgm, meals, config=config)
        windows.append(patient_windows)
        valid_times = cgm.loc[valid, "date"]
        audits.append(
            CGMacrosPatientAudit(
                patient_id=patient_id,
                source_file=path.name,
                raw_cgm_rows=int(len(cgm)),
                valid_cgm_rows=int(valid.sum()),
                meal_rows=int(len(meals)),
                aligned_windows=int(len(patient_windows)),
                cgm_start_utc=(valid_times.min().isoformat() if not valid_times.empty else None),
                cgm_stop_utc=(valid_times.max().isoformat() if not valid_times.empty else None),
            ).as_dict()
        )
    combined = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    if combined.empty:
        raise ValueError("CGMacros ingestion produced no eligible causal windows.")
    return combined, pd.DataFrame(audits)


def cgmacros_feature_registry() -> dict[str, tuple[str, ...]]:
    return {"cgm": CGMACROS_CGM_FEATURES, "events": CGMACROS_EVENT_FEATURES}
