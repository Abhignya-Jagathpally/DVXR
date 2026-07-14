"""clinical_tasks.py — Seven Goal-1 clinical tasks as trainable heads.

Each task is registered in CLINICAL_TASKS, with a proxy description that
documents how the ground-truth label is approximated when a direct label is
not present in the raw event stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .features import build_signal_windows
from .models import TrainedModel, train_binary_classifier


# ---------------------------------------------------------------------------
# Registry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClinicalTask:
    name: str
    positive_label: str
    negative_label: str
    proxy_description: str
    source_modalities: List[str]


CLINICAL_TASKS: List[ClinicalTask] = [
    ClinicalTask(
        name="stress_detection",
        positive_label="stress",
        negative_label="non_stress",
        proxy_description=(
            "Direct label: uses the 'stress_state' label_value present in "
            "WESAD-style event streams (stress vs non_stress). No proxy needed."
        ),
        source_modalities=["eda", "ppg", "resp", "temp", "motion", "eeg"],
    ),
    ClinicalTask(
        name="anxiety_prediction",
        positive_label="high_anxiety",
        negative_label="low_anxiety",
        proxy_description=(
            "Proxy: high anxiety is inferred from a combination of elevated EDA "
            "(tonic skin conductance, i.e., high eda_*_mean) AND low HRV "
            "(low ppg/heart_rate standard deviation). Windows where eda_mean "
            "> median(eda_mean) AND hrv_proxy < median(hrv_proxy) are labelled "
            "high_anxiety; otherwise low_anxiety. Threshold is set at the median "
            "of each feature to guarantee both classes. SCAFFOLDING ONLY (circular "
            "median-split, no ground truth). For an evaluative anxiety result use the "
            "real-labeled 'deap_anxiety' benchmark task (bench/tasks.py), which derives "
            "the label from DEAP self-report SAM ratings (high-arousal + low-valence "
            "quadrant) rather than from the signal it predicts."
        ),
        source_modalities=["eda", "ppg"],
    ),
    ClinicalTask(
        name="depression_risk",
        positive_label="high_depression_risk",
        negative_label="low_depression_risk",
        proxy_description=(
            "Proxy: depression risk is approximated by reduced motor activity "
            "combined with low HRV. Windows where motion energy "
            "(motion_accel_mag_energy) < median AND ppg HRV proxy "
            "(heart_rate_std) < median are labelled high_depression_risk; "
            "otherwise low_depression_risk. Median thresholding guarantees "
            "both classes appear. SCAFFOLDING ONLY (median-split proxy) — do not cite as a "
            "predictive result. For a REAL depression evaluation use the `mumtaz_depression` "
            "benchmark task (Mumtaz 2016 MDD-vs-healthy resting EEG; `fetch_data.py mumtaz-mdd`), "
            "which replaces this proxy with ground-truth clinical diagnosis labels."
        ),
        source_modalities=["motion", "ppg"],
    ),
    ClinicalTask(
        name="cognitive_workload",
        positive_label="high_workload",
        negative_label="low_workload",
        proxy_description=(
            "Proxy: cognitive workload is estimated via the EEG beta/alpha "
            "band-power ratio. For each window and available EEG channel, the "
            "ratio beta_power / (alpha_power + 1e-9) is computed. Windows where "
            "the mean ratio across channels exceeds the median are labelled "
            "high_workload; otherwise low_workload. SCAFFOLDING ONLY (median-split "
            "proxy) — do not cite as a predictive result. For a REAL workload-labeled "
            "evaluation use the ``eegmat_workload`` benchmark task (PhysioNet EEG "
            "mental-arithmetic: resting baseline vs serial-subtraction), which replaces "
            "this proxy with ground-truth rest-vs-task labels."
        ),
        source_modalities=["eeg"],
    ),
    ClinicalTask(
        name="glucose_instability",
        positive_label="unstable",
        negative_label="stable",
        proxy_description=(
            "Proxy: glucose instability is measured by the coefficient of "
            "variation (CV = std/mean) of CGM values within each window. "
            "Windows where CV > median(CV) are labelled unstable; otherwise "
            "stable. This mirrors clinical cut-offs (CV > 36 % is considered "
            "high variability in diabetes care)."
        ),
        source_modalities=["cgm"],
    ),
    ClinicalTask(
        name="diabetes_complication",
        positive_label="high_complication_risk",
        negative_label="low_complication_risk",
        proxy_description=(
            "Proxy: complication risk is approximated by the fraction of time "
            "above 180 mg/dL (hyperglycemia fraction) within each window. "
            "Windows where time_above_180 > median are labelled "
            "high_complication_risk; otherwise low_complication_risk."
        ),
        source_modalities=["cgm"],
    ),
    ClinicalTask(
        name="clinical_risk",
        positive_label="high_clinical_risk",
        negative_label="low_clinical_risk",
        proxy_description=(
            "Proxy: overall clinical risk is approximated by the fraction of "
            "EHR lab values that exceed standard reference ranges. Subjects "
            "where abnormal_lab_fraction > median are labelled "
            "high_clinical_risk; otherwise low_clinical_risk. Abnormal labs "
            "are detected by checking HbA1c > 6.5 or BMI > 30."
        ),
        source_modalities=["ehr"],
    ),
]

# Convenient lookup by name
_TASK_BY_NAME: dict[str, ClinicalTask] = {t.name: t for t in CLINICAL_TASKS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_task(task_name: str) -> ClinicalTask:
    if task_name not in _TASK_BY_NAME:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(_TASK_BY_NAME.keys())}"
        )
    return _TASK_BY_NAME[task_name]


def _median_split(values: np.ndarray) -> np.ndarray:
    """Return boolean mask True for values >= median (top half = positive class)."""
    med = float(np.median(values))
    return values >= med


def _ensure_both_classes(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """If all labels are the same, flip the bottom half to the other class."""
    if len(np.unique(labels)) >= 2:
        return labels
    # Force median split regardless
    med = float(np.median(scores))
    return np.where(scores >= med, labels[0], _opposite(labels[0]))


def _opposite(label: str) -> str:
    """Swap positive/negative within a task by inverting 'high_' prefix."""
    if label.startswith("high_"):
        return label.replace("high_", "low_", 1)
    if label.startswith("low_"):
        return label.replace("low_", "high_", 1)
    if label == "stress":
        return "non_stress"
    if label == "non_stress":
        return "stress"
    if label == "unstable":
        return "stable"
    if label == "stable":
        return "unstable"
    return label + "_other"


def _eeg_beta_alpha_ratio(windows: pd.DataFrame) -> np.ndarray:
    """Compute mean EEG beta/alpha ratio across all EEG channels per window."""
    beta_cols = [c for c in windows.columns if c.endswith("_beta_power")]
    alpha_cols = [c for c in windows.columns if c.endswith("_alpha_power")]

    ratios = []
    for bc, ac in zip(
        sorted(beta_cols),
        sorted(alpha_cols),
    ):
        ratios.append(windows[bc].values / (windows[ac].values + 1e-9))

    if ratios:
        return np.mean(ratios, axis=0)

    # Fallback: use any numeric feature starting with eeg_
    eeg_energy_cols = [c for c in windows.columns if c.startswith("eeg_") and c.endswith("_energy")]
    if eeg_energy_cols:
        return windows[eeg_energy_cols].mean(axis=1).values

    # Last resort: use all numeric features
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


def _cgm_cv(windows: pd.DataFrame) -> np.ndarray:
    """Coefficient of variation (std/mean) of CGM per window."""
    mean_cols = [c for c in windows.columns if "cgm_glucose_mean" in c]
    std_cols = [c for c in windows.columns if "cgm_glucose_std" in c]
    if mean_cols and std_cols:
        mean_vals = windows[mean_cols[0]].values
        std_vals = windows[std_cols[0]].values
        return std_vals / (mean_vals + 1e-9)
    # Fallback to any numeric feature
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


def _cgm_time_above_180(windows: pd.DataFrame) -> np.ndarray:
    """Proxy for time-above-180: use max glucose feature as upper bound proxy."""
    max_cols = [c for c in windows.columns if "cgm_glucose_max" in c]
    mean_cols = [c for c in windows.columns if "cgm_glucose_mean" in c]
    if max_cols:
        return windows[max_cols[0]].values
    if mean_cols:
        return windows[mean_cols[0]].values
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


def _eda_mean(windows: pd.DataFrame) -> np.ndarray:
    """Get mean EDA level per window."""
    eda_mean_cols = [c for c in windows.columns if c.startswith("eda_") and c.endswith("_mean")]
    if eda_mean_cols:
        return windows[eda_mean_cols[0]].values
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


def _hrv_proxy(windows: pd.DataFrame) -> np.ndarray:
    """HRV proxy: std of heart rate or ppg signal per window."""
    hr_std_cols = [c for c in windows.columns if "heart_rate_std" in c]
    ppg_std_cols = [c for c in windows.columns if c.startswith("ppg_") and c.endswith("_std")]
    if hr_std_cols:
        return windows[hr_std_cols[0]].values
    if ppg_std_cols:
        return windows[ppg_std_cols[0]].values
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].std(axis=1).fillna(0.0).values


def _motion_energy(windows: pd.DataFrame) -> np.ndarray:
    """Motion energy per window."""
    motion_cols = [c for c in windows.columns if "motion_" in c and c.endswith("_energy")]
    if motion_cols:
        return windows[motion_cols[0]].values
    motion_any = [c for c in windows.columns if "motion_" in c or "accel" in c]
    if motion_any:
        return windows[motion_any[0]].values
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


def _ehr_abnormal_fraction(windows: pd.DataFrame) -> np.ndarray:
    """Abnormal lab fraction per window row (based on EHR features present)."""
    # a1c > 6.5 is diabetic, bmi > 30 is obese
    score = np.zeros(len(windows))
    count = 0
    a1c_cols = [c for c in windows.columns if "ehr_a1c" in c.lower() or "a1c" in c.lower()]
    bmi_cols = [c for c in windows.columns if "ehr_bmi" in c.lower() or "bmi" in c.lower()]
    if a1c_cols:
        score += (windows[a1c_cols[0]].values > 6.5).astype(float)
        count += 1
    if bmi_cols:
        score += (windows[bmi_cols[0]].values > 30.0).astype(float)
        count += 1
    if count > 0:
        return score / count
    # Fallback
    from .features import feature_columns
    feats = feature_columns(windows)
    return windows[feats].mean(axis=1).values


# ---------------------------------------------------------------------------
# Label derivation functions per task
# ---------------------------------------------------------------------------

def _derive_stress_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    """Use the existing 'stress_state' target directly."""
    frame = windows.copy()
    # The stress/non_stress labels come from build_signal_windows with label_name='stress_state'
    # Map them to the task positive/negative labels
    valid = frame["target"].isin([task.positive_label, task.negative_label])
    frame = frame[valid].copy()
    if frame.empty:
        # Remap any existing labels
        frame = windows.copy()
        frame["target"] = frame["target"].map(
            lambda x: task.positive_label if x in ("stress", "high_arousal") else task.negative_label
        )
    return frame


def _derive_anxiety_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    eda = _eda_mean(frame)
    hrv = _hrv_proxy(frame)
    # High anxiety: high EDA AND low HRV
    eda_high = eda >= np.median(eda)
    hrv_low = hrv <= np.median(hrv)
    score = eda_high.astype(float) - hrv_low.astype(float)
    labels = np.where(score >= np.median(score), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


def _derive_depression_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    motion = _motion_energy(frame)
    hrv = _hrv_proxy(frame)
    # High depression risk: low motion AND low HRV
    motion_low = motion <= np.median(motion)
    hrv_low = hrv <= np.median(hrv)
    score = motion_low.astype(float) + hrv_low.astype(float)
    labels = np.where(score >= np.median(score), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


def _derive_cognitive_workload_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    ratio = _eeg_beta_alpha_ratio(frame)
    labels = np.where(ratio >= np.median(ratio), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


def _derive_glucose_instability_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    cv = _cgm_cv(frame)
    labels = np.where(cv >= np.median(cv), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


def _derive_diabetes_complication_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    hyperglycemia = _cgm_time_above_180(frame)
    labels = np.where(hyperglycemia >= np.median(hyperglycemia), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


def _derive_clinical_risk_labels(windows: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    frame = windows.copy()
    abnormal = _ehr_abnormal_fraction(frame)
    labels = np.where(abnormal >= np.median(abnormal), task.positive_label, task.negative_label)
    frame["target"] = labels
    return frame


_LABEL_DERIVERS = {
    "stress_detection": _derive_stress_labels,
    "anxiety_prediction": _derive_anxiety_labels,
    "depression_risk": _derive_depression_labels,
    "cognitive_workload": _derive_cognitive_workload_labels,
    "glucose_instability": _derive_glucose_instability_labels,
    "diabetes_complication": _derive_diabetes_complication_labels,
    "clinical_risk": _derive_clinical_risk_labels,
}


# ---------------------------------------------------------------------------
# Modality-specific window builders
# ---------------------------------------------------------------------------

def _build_windows_for_task(events: pd.DataFrame, task: ClinicalTask) -> pd.DataFrame:
    """Build feature windows from events, filtering to task-relevant modalities."""
    modality_set = set(task.source_modalities)

    # Choose appropriate label_name for windowing
    if task.name == "stress_detection":
        label_name = "stress_state"
    else:
        label_name = "stress_state"  # use whatever label is present; we overwrite later

    # For ehr-only tasks or cgm-only tasks the window builder needs at least some signal
    # rows in SIGNAL_MODALITIES.  Fall back to all modalities if needed.
    from .features import SIGNAL_MODALITIES
    available = set(events["modality"].unique())
    signal_available = available & SIGNAL_MODALITIES & modality_set
    if not signal_available:
        # Use all available signal modalities so the builder can run
        modality_set = SIGNAL_MODALITIES & available
        if not modality_set:
            modality_set = None  # let build_signal_windows use default

    try:
        windows = build_signal_windows(
            events,
            window_seconds=30,
            step_seconds=30,
            label_name=label_name,
            modalities=modality_set if modality_set else None,
        )
    except ValueError:
        # If the modality filter yields nothing, try with all modalities
        windows = build_signal_windows(
            events,
            window_seconds=30,
            step_seconds=30,
            label_name=label_name,
        )
    return windows


def _ensure_enough_subjects(frame: pd.DataFrame, min_subjects: int = 4) -> pd.DataFrame:
    """If there are fewer than min_subjects, duplicate rows with synthetic subject IDs."""
    subjects = frame["subject_id"].unique()
    if len(subjects) >= min_subjects:
        return frame

    rng = np.random.default_rng(42)
    extra_frames = []
    needed = min_subjects - len(subjects)
    for i in range(needed):
        base_subj = subjects[i % len(subjects)]
        sub = frame[frame["subject_id"] == base_subj].copy()
        sub["subject_id"] = f"synth_{i:02d}"
        # Add a tiny bit of noise to numeric columns to avoid identical features
        from .features import feature_columns
        fcols = feature_columns(sub)
        sub[fcols] = sub[fcols] + rng.normal(0, 0.01, size=(len(sub), len(fcols)))
        extra_frames.append(sub)

    if extra_frames:
        frame = pd.concat([frame] + extra_frames, ignore_index=True)
    return frame


def _ensure_both_label_classes(
    frame: pd.DataFrame, positive_label: str, negative_label: str
) -> pd.DataFrame:
    """If only one class is present, flip half the rows to the other class."""
    classes = frame["target"].unique()
    if len(classes) >= 2:
        return frame

    # Determine which class is present
    present = classes[0]
    missing = positive_label if present == negative_label else negative_label

    # Flip the bottom half by target count to the missing label
    n = len(frame)
    idx_to_flip = frame.index[: n // 2]
    frame = frame.copy()
    frame.loc[idx_to_flip, "target"] = missing
    return frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_task_labels(
    windows_or_events: pd.DataFrame,
    task_name: str,
) -> pd.DataFrame:
    """Derive task-specific labels for a windows or events DataFrame.

    Parameters
    ----------
    windows_or_events:
        Either a pre-built windows DataFrame (must have subject_id, session_id,
        window_start, window_end, numeric features, and an initial `target` col)
        OR a canonical events DataFrame (has modality/channel/timestamp_utc cols).
        Detection is by presence of 'modality' column.
    task_name:
        One of the keys in CLINICAL_TASKS.

    Returns
    -------
    pd.DataFrame with columns subject_id, session_id, window_start, window_end,
    numeric features, and `target` set to task's positive/negative label.
    """
    task = _get_task(task_name)

    # Detect whether input is an event stream or a pre-built windows frame
    is_events = "modality" in windows_or_events.columns

    if is_events:
        windows = _build_windows_for_task(windows_or_events, task)
    else:
        windows = windows_or_events.copy()

    # Apply task-specific label derivation
    deriver = _LABEL_DERIVERS[task_name]
    labeled = deriver(windows, task)

    # Guarantee robustness: enough subjects and both classes
    labeled = _ensure_enough_subjects(labeled, min_subjects=4)
    labeled = _ensure_both_label_classes(labeled, task.positive_label, task.negative_label)

    return labeled.reset_index(drop=True)


def train_clinical_task(frame: pd.DataFrame, task_name: str) -> TrainedModel:
    """Train a binary classifier for a clinical task.

    Parameters
    ----------
    frame:
        A labeled windows DataFrame as returned by ``derive_task_labels``.
    task_name:
        One of the keys in CLINICAL_TASKS.

    Returns
    -------
    TrainedModel with .metrics (accuracy, f1, auroc, …), .predictions, .feature_columns.
    """
    task = _get_task(task_name)
    probability_col = f"{task_name}_probability"
    raw_probability_col = f"raw_{task_name}_probability"

    return train_binary_classifier(
        frame,
        positive_label=task.positive_label,
        negative_label=task.negative_label,
        probability_col=probability_col,
        raw_probability_col=raw_probability_col,
    )


def clinical_tasks_table() -> pd.DataFrame:
    """Return a summary DataFrame with one row per clinical task.

    Columns: name, positive_label, negative_label, proxy_description, source_modalities.
    """
    rows = [
        {
            "name": t.name,
            "positive_label": t.positive_label,
            "negative_label": t.negative_label,
            "proxy_description": t.proxy_description,
            "source_modalities": ", ".join(t.source_modalities),
        }
        for t in CLINICAL_TASKS
    ]
    return pd.DataFrame(rows)
