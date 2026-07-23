"""DiaTrend-style cohort-overview figures generated only from recorded artifacts.

These reproduce the dataset-overview panels of the DiaTrend paper's Figure 1
(per-participant CGM traces, time-in-range distribution, glucose distribution,
data-availability/coverage, and a cohort summary) from the aligned window table and the
ingestion audit that the sentinel CLI already wrote. No value is synthesized here — every
panel is computed directly from the passed DataFrames. Every figure is titled with the
real cohort label so a substitute cohort is never mistaken for DiaTrend.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Default observed-glucose column in the aligned window tables.
DEFAULT_GLUCOSE_COLUMN = "reference_current_glucose_mg_dl"
DEFAULT_TIME_COLUMN = "anchor_time"
DEFAULT_PATIENT_COLUMN = "patient_id"
HYPO_MG_DL = 70.0
HYPER_MG_DL = 180.0


def _pyplot(cache_directory: Path):
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _glucose_series(frame: pd.DataFrame, glucose_column: str) -> pd.Series:
    return pd.to_numeric(frame[glucose_column], errors="coerce").dropna()


def save_cgm_trace_figure(
    windows: pd.DataFrame,
    destination: Path,
    *,
    glucose_column: str = DEFAULT_GLUCOSE_COLUMN,
    time_column: str = DEFAULT_TIME_COLUMN,
    patient_column: str = DEFAULT_PATIENT_COLUMN,
    max_participants: int = 6,
    hypo: float = HYPO_MG_DL,
    hyper: float = HYPER_MG_DL,
    cohort_label: str = "Cohort",
) -> Path:
    """Per-participant observed-glucose traces vs. time (DiaTrend Fig. 1-style)."""
    _require_columns(
        windows, {glucose_column, time_column, patient_column}, label="Window table"
    )
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    patients = sorted(windows[patient_column].dropna().unique())[:max_participants]
    n = len(patients)
    if n == 0:
        raise ValueError("No participants available for CGM trace figure.")
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 2.6 * nrows), squeeze=False,
        constrained_layout=True,
    )
    flat = axes.flatten()
    for axis, patient in zip(flat, patients):
        group = windows.loc[windows[patient_column].eq(patient)].copy()
        time = pd.to_datetime(group[time_column], errors="coerce", utc=True)
        glucose = pd.to_numeric(group[glucose_column], errors="coerce")
        order = time.argsort()
        axis.axhspan(hypo, hyper, color="#12b76a", alpha=0.08)
        axis.plot(time.iloc[order], glucose.iloc[order], linewidth=0.8, color="#1d4ed8")
        axis.axhline(hypo, color="#b42318", linewidth=0.8, linestyle="--")
        axis.axhline(hyper, color="#dc6803", linewidth=0.8, linestyle="--")
        axis.set(title=f"Participant {patient}", ylabel="Glucose (mg/dL)")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", labelrotation=30, labelsize=7)
    for axis in flat[n:]:
        axis.set_visible(False)
    fig.suptitle(f"Observed CGM traces — {cohort_label}", fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_time_in_range_figure(
    windows: pd.DataFrame,
    destination: Path,
    *,
    glucose_column: str = DEFAULT_GLUCOSE_COLUMN,
    patient_column: str = DEFAULT_PATIENT_COLUMN,
    hypo: float = HYPO_MG_DL,
    hyper: float = HYPER_MG_DL,
    cohort_label: str = "Cohort",
) -> Path:
    """Per-participant time-in-range stacked bars (%<70 / 70-180 / >180)."""
    _require_columns(windows, {glucose_column, patient_column}, label="Window table")
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    rows = []
    for patient, group in windows.groupby(patient_column, sort=True):
        glucose = _glucose_series(group, glucose_column)
        if glucose.empty:
            continue
        total = len(glucose)
        rows.append(
            {
                "patient": patient,
                "below": 100.0 * (glucose < hypo).sum() / total,
                "in_range": 100.0 * ((glucose >= hypo) & (glucose <= hyper)).sum() / total,
                "above": 100.0 * (glucose > hyper).sum() / total,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No glucose values available for time-in-range figure.")
    labels = table["patient"].astype(str).to_list()
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(6.0, 0.7 * len(labels) + 2), 4.5),
                             constrained_layout=True)
    axis.bar(x, table["below"], label=f"< {hypo:g} (hypo)", color="#b42318")
    axis.bar(x, table["in_range"], bottom=table["below"],
             label=f"{hypo:g}-{hyper:g} (in range)", color="#12b76a")
    axis.bar(x, table["above"], bottom=table["below"] + table["in_range"],
             label=f"> {hyper:g} (hyper)", color="#dc6803")
    axis.set(
        title=f"Time in range by participant — {cohort_label}",
        xlabel="Participant", ylabel="% of observations", ylim=(0, 100),
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=0)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    axis.grid(axis="y", alpha=0.25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_glucose_distribution_figure(
    windows: pd.DataFrame,
    destination: Path,
    *,
    glucose_column: str = DEFAULT_GLUCOSE_COLUMN,
    patient_column: str = DEFAULT_PATIENT_COLUMN,
    hypo: float = HYPO_MG_DL,
    hyper: float = HYPER_MG_DL,
    cohort_label: str = "Cohort",
) -> Path:
    """Cohort glucose histogram + per-participant violin distribution."""
    _require_columns(windows, {glucose_column, patient_column}, label="Window table")
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    cohort = _glucose_series(windows, glucose_column)
    if cohort.empty:
        raise ValueError("No glucose values available for distribution figure.")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].hist(cohort, bins=60, color="#1d4ed8", alpha=0.85)
    axes[0].axvline(hypo, color="#b42318", linestyle="--", linewidth=1)
    axes[0].axvline(hyper, color="#dc6803", linestyle="--", linewidth=1)
    axes[0].set(title="Cohort glucose distribution", xlabel="Glucose (mg/dL)",
                ylabel="Observations")
    axes[0].grid(alpha=0.2)
    patients = sorted(windows[patient_column].dropna().unique())
    data = [
        _glucose_series(windows.loc[windows[patient_column].eq(p)], glucose_column).to_numpy()
        for p in patients
    ]
    data = [series for series in data if series.size]
    positions = np.arange(1, len(data) + 1)
    axes[1].violinplot(data, positions=positions, showmedians=True)
    axes[1].axhline(hypo, color="#b42318", linestyle="--", linewidth=0.8)
    axes[1].axhline(hyper, color="#dc6803", linestyle="--", linewidth=0.8)
    axes[1].set(title="Per-participant glucose distribution", xlabel="Participant",
                ylabel="Glucose (mg/dL)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(p) for p in patients], rotation=0, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(f"Glucose distribution — {cohort_label}", fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_data_availability_figure(
    audit: pd.DataFrame,
    destination: Path,
    *,
    patient_column: str = DEFAULT_PATIENT_COLUMN,
    cohort_label: str = "Cohort",
) -> Path:
    """Per-participant data availability: aligned windows + recording span (days)."""
    _require_columns(
        audit, {patient_column, "aligned_windows", "start_utc", "stop_utc"},
        label="Ingestion audit",
    )
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    table = audit.copy()
    start = pd.to_datetime(table["start_utc"], errors="coerce", utc=True)
    stop = pd.to_datetime(table["stop_utc"], errors="coerce", utc=True)
    table["span_days"] = (stop - start).dt.total_seconds() / 86400.0
    table = table.sort_values(patient_column)
    labels = table[patient_column].astype(str).to_list()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].bar(x, table["aligned_windows"], color="#1d4ed8")
    axes[0].set(title="Aligned forecast windows", xlabel="Participant",
                ylabel="Window count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, table["span_days"], color="#12b76a")
    axes[1].set(title="Recording span", xlabel="Participant", ylabel="Days")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Data availability ({len(labels)} participants) — {cohort_label}", fontsize=13
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def save_cohort_summary_figure(
    windows: pd.DataFrame,
    audit: pd.DataFrame,
    destination: Path,
    *,
    glucose_column: str = DEFAULT_GLUCOSE_COLUMN,
    patient_column: str = DEFAULT_PATIENT_COLUMN,
    hypo: float = HYPO_MG_DL,
    hyper: float = HYPER_MG_DL,
    cohort_label: str = "Cohort",
) -> Path:
    """A compact cohort summary table rendered as a figure."""
    _require_columns(windows, {glucose_column, patient_column}, label="Window table")
    destination = Path(destination)
    plt = _pyplot(destination.parent / ".matplotlib-cache")
    glucose = _glucose_series(windows, glucose_column)
    n_patients = int(windows[patient_column].nunique())
    total_windows = int(len(windows))
    in_range = 100.0 * ((glucose >= hypo) & (glucose <= hyper)).mean() if len(glucose) else float("nan")
    span_days = float("nan")
    if {"start_utc", "stop_utc"} <= set(audit.columns):
        start = pd.to_datetime(audit["start_utc"], errors="coerce", utc=True)
        stop = pd.to_datetime(audit["stop_utc"], errors="coerce", utc=True)
        span_days = float(((stop - start).dt.total_seconds() / 86400.0).median())
    rows = [
        ("Participants", f"{n_patients}"),
        ("Aligned windows", f"{total_windows:,}"),
        ("Glucose observations", f"{len(glucose):,}"),
        ("Median glucose (mg/dL)", f"{glucose.median():.0f}" if len(glucose) else "n/a"),
        ("IQR glucose (mg/dL)",
         f"{glucose.quantile(0.25):.0f}-{glucose.quantile(0.75):.0f}" if len(glucose) else "n/a"),
        ("Cohort time-in-range", f"{in_range:.1f}%" if len(glucose) else "n/a"),
        ("Median recording span", f"{span_days:.1f} days" if span_days == span_days else "n/a"),
    ]
    fig, axis = plt.subplots(figsize=(7.0, 0.6 * len(rows) + 1.2), constrained_layout=True)
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=["Metric", "Value"], loc="center",
                       cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axis.set_title(f"Cohort summary — {cohort_label}", fontsize=13, pad=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def build_overview_suite(
    windows: pd.DataFrame,
    audit: pd.DataFrame,
    figure_dir: Path,
    *,
    cohort_label: str = "Cohort",
    glucose_column: str = DEFAULT_GLUCOSE_COLUMN,
    hypo: float = HYPO_MG_DL,
    hyper: float = HYPER_MG_DL,
) -> dict[str, Path]:
    """Write all five DiaTrend-style overview figures; return {name: path}."""
    figure_dir = Path(figure_dir)
    outputs = {
        "cgm_traces": save_cgm_trace_figure(
            windows, figure_dir / "diatrend_cgm_traces.png",
            glucose_column=glucose_column, hypo=hypo, hyper=hyper, cohort_label=cohort_label,
        ),
        "time_in_range": save_time_in_range_figure(
            windows, figure_dir / "diatrend_time_in_range.png",
            glucose_column=glucose_column, hypo=hypo, hyper=hyper, cohort_label=cohort_label,
        ),
        "glucose_distribution": save_glucose_distribution_figure(
            windows, figure_dir / "diatrend_glucose_distribution.png",
            glucose_column=glucose_column, hypo=hypo, hyper=hyper, cohort_label=cohort_label,
        ),
        "data_availability": save_data_availability_figure(
            audit, figure_dir / "diatrend_data_availability.png", cohort_label=cohort_label,
        ),
        "cohort_summary": save_cohort_summary_figure(
            windows, audit, figure_dir / "diatrend_cohort_summary.png",
            glucose_column=glucose_column, hypo=hypo, hyper=hyper, cohort_label=cohort_label,
        ),
    }
    return outputs
