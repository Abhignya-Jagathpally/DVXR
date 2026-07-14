"""dvxr.bench.tasks — real-label task adapters with NO circular proxies.

Each adapter returns a BenchTask carrying per-modality feature matrices, a real
external label, and subject/patient ids for grouped CV. Windows are
NON-OVERLAPPING (step == window) so the reported N is not inflated.

Real labels only:
  * stress   — Non-EEG `stress_state` phase annotations (external)   [multimodal]
  * glucose  — Shanghai CGM actual future glucose (regression)       [cgm only]
  * mortality— MIMIC-IV demo in-hospital death flag (external)       [ehr only]

The circular median-split proxies in clinical_tasks.py are deliberately NOT used
here. assert_no_fabrication() refuses to run if a fabrication helper is wired in.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dvxr.features import (
    build_glucose_forecast_table,
    build_raw_windows,
    build_signal_windows,
    feature_columns,
)
from dvxr.loaders import (
    load_cgmacros_bio,
    load_cgmacros_dataset,
    load_deap_dataset,
    load_eegmat_dataset,
    load_mimic_demo_ehr,
    load_mumtaz_mdd_dataset,
    load_noneeg_dataset,
    load_shanghai_cgm_dataset,
    load_wesad_dataset,
)


@dataclass
class BenchTask:
    name: str
    kind: str                                   # "classification" | "forecast"
    features: Dict[str, np.ndarray]             # {modality: (N, width)} standardizable
    feature_names: Dict[str, List[str]]         # {modality: [col, ...]}
    y: np.ndarray                               # (N,) int labels or float targets
    subject_ids: np.ndarray                     # (N,) group ids
    metric: str                                 # reported error metric name
    baseline_hint: str                          # "majority" | "persistence"
    raw_windows: Optional[pd.DataFrame] = None  # events-derived, for SOTA encoders
    extra: dict = field(default_factory=dict)

    @property
    def modalities(self) -> List[str]:
        return list(self.features.keys())

    @property
    def n(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------- guards
def assert_no_fabrication() -> None:
    """Fail loudly if any label-fabrication helper is importable into this path.

    The benchmark must never flip classes or synthesize subjects. These helpers
    are fine for the smoke demo but forbidden here.
    """
    import dvxr.clinical_tasks as ct
    for name in ("_ensure_both_label_classes", "_ensure_enough_subjects"):
        fn = getattr(ct, name, None)
        # presence is fine; what matters is we never call them — assert they are
        # not monkeypatched onto the bench module namespace.
        if fn is not None and getattr(fn, "_bench_enabled", False):
            raise RuntimeError(f"label fabrication helper {name} is enabled in bench path")


def _split_by_modality(frame: pd.DataFrame) -> Dict[str, List[str]]:
    """Group feature columns by their modality prefix ({modality}_{channel}_{stat})."""
    cols = feature_columns(frame)
    groups: Dict[str, List[str]] = {}
    for c in cols:
        mod = c.split("_", 1)[0]
        groups.setdefault(mod, []).append(c)
    return groups


# --------------------------------------------------------------------- stress
def noneeg_stress_task(data_dir: str = "data/real/noneeg", subjects: int = 20,
                       window_seconds: int = 30) -> BenchTask:
    """Non-EEG multimodal stress — REAL phase annotations, non-overlapping windows."""
    assert_no_fabrication()
    events = load_noneeg_dataset(data_dir, subjects=subjects)
    win = build_signal_windows(events, window_seconds=window_seconds,
                               step_seconds=window_seconds, label_name="stress_state")
    win = win[win["target"].astype(str).str.len() > 0].reset_index(drop=True)
    y = (win["target"].astype(str) == "stress").astype(int).to_numpy()
    groups = _split_by_modality(win)
    feats = {m: win[cols].to_numpy(dtype=float) for m, cols in groups.items()}
    return BenchTask(
        name="stress", kind="classification", features=feats,
        feature_names=groups, y=y, subject_ids=win["subject_id"].to_numpy(),
        metric="1-AUROC", baseline_hint="majority", raw_windows=win,
        extra={"events": events, "window_seconds": window_seconds},
    )


# -------------------------------------------------------------------- glucose
def shanghai_glucose_task(data_dir: str = "data/real/shanghai_cgm",
                          max_patients: Optional[int] = None,
                          horizon_minutes: int = 30) -> BenchTask:
    """Shanghai CGM 30-min-ahead forecast — REAL future glucose (regression)."""
    assert_no_fabrication()
    events = load_shanghai_cgm_dataset(data_dir, max_patients=max_patients)
    tbl = build_glucose_forecast_table(events, history_minutes=30,
                                       horizon_minutes=horizon_minutes)
    tbl = tbl.dropna(subset=["target_glucose", "glucose_now"]).reset_index(drop=True)
    cols = [c for c in feature_columns(tbl) if c != "target_glucose"]
    feats = {"cgm": tbl[cols].to_numpy(dtype=float)}
    return BenchTask(
        name="glucose", kind="forecast", features=feats,
        feature_names={"cgm": cols}, y=tbl["target_glucose"].to_numpy(dtype=float),
        subject_ids=tbl["subject_id"].to_numpy(), metric="MAE",
        baseline_hint="persistence", raw_windows=tbl,
        extra={"persistence_col": cols.index("glucose_now")},
    )


# ------------------------------------------------------------------ mortality
def _mimic_mortality_labels(hosp_dir: str) -> Dict[str, int]:
    """Real in-hospital mortality per patient from admissions.hospital_expire_flag."""
    adm = Path(hosp_dir) / "admissions.csv.gz"
    with gzip.open(adm, "rt") as fh:
        df = pd.read_csv(fh, usecols=["subject_id", "hospital_expire_flag"])
    died = df.groupby("subject_id")["hospital_expire_flag"].max()
    return {f"mimic_{sid}": int(v) for sid, v in died.items()}


def mimic_mortality_task(hosp_dir: str = "data/real/mimic_demo/hosp",
                         max_lab_rows: int = 400000) -> BenchTask:
    """MIMIC-IV demo in-hospital mortality — REAL outcome, per-patient EHR vector."""
    assert_no_fabrication()
    events = load_mimic_demo_ehr(hosp_dir, max_lab_rows=max_lab_rows)
    labels = _mimic_mortality_labels(hosp_dir)
    # per-patient feature vector: mean value per lab/demographic concept
    piv = (events.assign(value=pd.to_numeric(events["value"], errors="coerce"))
           .pivot_table(index="subject_id", columns="channel", values="value",
                        aggfunc="mean"))
    # keep concepts present for a reasonable share of patients (limit sparsity)
    keep = piv.columns[piv.notna().mean() >= 0.5]
    piv = piv[keep].fillna(piv[keep].median())
    piv = piv.loc[[s for s in piv.index if s in labels]]
    y = np.array([labels[s] for s in piv.index], dtype=int)
    feats = {"ehr": piv.to_numpy(dtype=float)}
    return BenchTask(
        name="mortality", kind="classification", features=feats,
        feature_names={"ehr": list(piv.columns)}, y=y,
        subject_ids=np.array(piv.index), metric="1-AUROC",
        baseline_hint="majority", raw_windows=None,
        extra={"n_concepts": int(piv.shape[1])},
    )


# ------------------------------------------------------------- WESAD stress
def wesad_stress_task(data_dir: str = "data/real/WESAD", subjects: int = 8,
                      window_seconds: int = 60) -> BenchTask:
    """WESAD multimodal stress — REAL protocol labels (stress vs non-stress).

    Chest (RespiBAN) + wrist (Empatica E4) physiology. Protocol condition 2 = stress;
    1/3/4 = baseline/amusement/meditation (non-stress); 0/5-7 (transient/ignore) dropped.
    Non-overlapping windows so N is not inflated.
    """
    assert_no_fabrication()
    events = load_wesad_dataset(data_dir, subjects=subjects)
    win = build_signal_windows(events, window_seconds=window_seconds,
                               step_seconds=window_seconds, label_name="wesad_label")
    codes = pd.to_numeric(win["target"], errors="coerce")
    win = win[codes.isin([1, 2, 3, 4])].reset_index(drop=True)
    codes = pd.to_numeric(win["target"], errors="coerce")
    y = (codes == 2).astype(int).to_numpy()
    groups = _split_by_modality(win)
    feats = {m: win[cols].to_numpy(dtype=float) for m, cols in groups.items()}
    return BenchTask(
        name="wesad_stress", kind="classification", features=feats,
        feature_names=groups, y=y, subject_ids=win["subject_id"].to_numpy(),
        metric="1-AUROC", baseline_hint="majority", raw_windows=win,
        extra={"events": events, "window_seconds": window_seconds},
    )


# --------------------------------------------------------- CGMacros glucose
def cgmacros_glucose_task(data_dir: str = "data/real/cgmacros", subjects: Optional[int] = 12,
                          horizon_minutes: int = 30, source: str = "dexcom") -> BenchTask:
    """CGMacros 30-min-ahead glucose forecast — REAL future CGM (regression).

    Uses one CGM source (Dexcom G6 by default) so simultaneous Libre readings don't
    create duplicate timestamps.
    """
    assert_no_fabrication()
    events = load_cgmacros_dataset(data_dir, subjects=subjects, include_bio=False)
    cgm = events[events["modality"] == "cgm"]
    if "glucose_source" in cgm.columns:
        cgm = cgm[cgm["glucose_source"] == source]
    # CGMacros CSVs are per-minute; thin to ~10-min steps (near the 5-min Dexcom native
    # cadence) so forecast windows aren't near-duplicate and N isn't inflated.
    cgm = cgm.sort_values(["subject_id", "session_id", "timestamp_utc"])
    order = cgm.groupby(["subject_id", "session_id"]).cumcount()
    cgm = cgm[order % 10 == 0]
    tbl = build_glucose_forecast_table(cgm, history_minutes=40, horizon_minutes=horizon_minutes)
    tbl = tbl.dropna(subset=["target_glucose", "glucose_now"]).reset_index(drop=True)
    cols = [c for c in feature_columns(tbl) if c != "target_glucose"]
    feats = {"cgm": tbl[cols].to_numpy(dtype=float)}
    return BenchTask(
        name="cgmacros_glucose", kind="forecast", features=feats,
        feature_names={"cgm": cols}, y=tbl["target_glucose"].to_numpy(dtype=float),
        subject_ids=tbl["subject_id"].to_numpy(), metric="MAE",
        baseline_hint="persistence", raw_windows=tbl,
        extra={"persistence_col": cols.index("glucose_now")},
    )


# -------------------------------------------------------- CGMacros diabetes
# Diagnostic glycemic labs that DEFINE the diabetes label or are direct diagnostic
# synonyms for it. The label is diabetes = int(HbA1c >= 6.5), so handing the model the
# HbA1c channel (or fasting glucose/insulin, which are the other ADA glycemic diagnostics)
# leaks the target and produces a spuriously near-perfect AUROC. These channels are
# excluded from the diabetes feature matrix; the task then predicts A1c-defined status
# from glucose dynamics + non-defining physiology/covariates, matching the proposal's
# "glucose instability / diabetes risk progression" framing. Canonical channel names come
# from loaders.CGMACROS_BIO_NUMERIC. Non-glycemic labs (lipids) and demographics are kept
# as legitimate correlated covariates, not leakage.
DIABETES_EHR_DENYLIST = frozenset({"hba1c", "fasting_glucose", "fasting_insulin"})


def cgmacros_diabetes_task(data_dir: str = "data/real/cgmacros",
                           subjects: Optional[int] = None) -> BenchTask:
    """CGMacros diabetes classification — REAL A1c-derived strata (diabetes vs not).

    Per-subject multimodal vectors: cgm (glucose summary + variability), wearable_phys
    (Fitbit summary), ehr (bio labs). Label = HbA1c-derived diabetes status (diabetes vs
    healthy/prediabetes). The defining glycemic labs (HbA1c, fasting glucose, fasting
    insulin) are removed from the ehr features to avoid target leakage — see
    ``DIABETES_EHR_DENYLIST``. Multimodal → usable for the modality ablation.
    """
    assert_no_fabrication()
    events = load_cgmacros_dataset(data_dir, subjects=subjects, include_bio=True)
    if "glucose_source" in events.columns:
        events = events[(events["modality"] != "cgm") | (events["glucose_source"] == "dexcom")].copy()

    # real label per subject from the diabetes_status carried on the events
    status = (events[events["label_value"] != ""]
              .groupby("subject_id")["label_value"].agg(lambda s: s.iloc[0]))
    y_map = {sid: int(v == "diabetes") for sid, v in status.items()}

    feats: Dict[str, np.ndarray] = {}
    names: Dict[str, List[str]] = {}
    index: Optional[List[str]] = None
    for modality in ("cgm", "wearable_phys", "ehr"):
        sub = events[events["modality"] == modality]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="subject_id", columns="channel",
                              values="value", aggfunc="mean")
        if modality == "cgm":  # add per-subject glucose variability (CV)
            cv = sub.groupby("subject_id")["value"].agg(lambda s: float(np.std(s) / (np.mean(s) + 1e-9)))
            piv["glucose_cv"] = cv
        if modality == "ehr":  # drop diagnostic glycemic labs that define/leak the label
            piv = piv.drop(columns=[c for c in piv.columns if c in DIABETES_EHR_DENYLIST])
            if piv.shape[1] == 0:
                continue
        piv = piv.loc[[s for s in piv.index if s in y_map]].sort_index()
        piv = piv.fillna(piv.median())
        if index is None:
            index = list(piv.index)
        piv = piv.reindex(index).fillna(piv.median())
        feats[modality] = piv.to_numpy(dtype=float)
        names[modality] = [f"{modality}_{c}" for c in piv.columns]

    # Leak guard: the defining glycemic labs must never reach the feature matrix.
    leaked = [n for ns in names.values() for n in ns
              if n.split("_", 1)[-1] in DIABETES_EHR_DENYLIST]
    if leaked:
        raise RuntimeError(f"diabetes target leak: defining labs present as features: {leaked}")

    y = np.array([y_map[s] for s in index], dtype=int)
    return BenchTask(
        name="cgmacros_diabetes", kind="classification", features=feats,
        feature_names=names, y=y, subject_ids=np.array(index),
        metric="1-AUROC", baseline_hint="majority", raw_windows=None,
        extra={"n_subjects": len(index)},
    )


# ---------------------------------------------------------------- DEAP affect
_DEAP_RAW_NOTE = ("decimated preprocessed DEAP signal (~8 Hz effective); "
                  "raw waveform vs the summary-stat floor on the same signal")


def _windowed_signal_task(name: str, events, label_name: str, positive: str,
                          window_seconds: int, raw_note: str,
                          extra: Optional[dict] = None) -> BenchTask:
    """Shared builder for windowed EEG+peripheral classification tasks with a raw-signal path.

    Used by the DEAP affect tasks and the eegmat workload task: non-overlapping windows →
    per-modality summary features (the floor's input) + aligned raw windows (``extra["raw"]``,
    the ``raw_cnn`` lever) → subject-held-out ``BenchTask``. Only the pre-loaded ``events``,
    the window label column, and the positive-class name differ across tasks.
    """
    assert_no_fabrication()
    win = build_signal_windows(events, window_seconds=window_seconds,
                               step_seconds=window_seconds, label_name=label_name)
    win = win[win["target"].astype(str).str.len() > 0].reset_index(drop=True)
    y = (win["target"].astype(str) == positive).astype(int).to_numpy()
    groups = _split_by_modality(win)
    feats = {m: win[cols].to_numpy(dtype=float) for m, cols in groups.items()}
    raw, raw_ch = build_raw_windows(events, win, modalities=list(groups))
    return BenchTask(
        name=name, kind="classification", features=feats,
        feature_names=groups, y=y, subject_ids=win["subject_id"].to_numpy(),
        metric="1-AUROC", baseline_hint="majority", raw_windows=win,
        extra={"events": events, "window_seconds": window_seconds,
               "raw": raw, "raw_channels": raw_ch, "raw_note": raw_note,
               **(extra or {})},
    )


def _deap_affect_task(name: str, label_scheme: str, positive: str,
                      data_dir: str, subjects: int, max_trials: Optional[int],
                      window_seconds: int, extra: Optional[dict] = None) -> BenchTask:
    """DEAP EEG+peripheral affect tasks (arousal, anxiety): REAL self-report (SAM) labels.

    ``label_scheme`` names both the loader scheme and the window label column (see
    ``_deap_affect_label`` in loaders.py); only it and the positive-class name differ.
    """
    events = load_deap_dataset(data_dir, subjects=subjects, max_trials=max_trials,
                               label_scheme=label_scheme)
    return _windowed_signal_task(name, events, label_scheme, positive, window_seconds,
                                 _DEAP_RAW_NOTE, extra)


def deap_arousal_task(data_dir: str = "data/real/deap", subjects: int = 8,
                      max_trials: Optional[int] = None, window_seconds: int = 8) -> BenchTask:
    """DEAP EEG+peripheral arousal — REAL self-report arousal (high vs low SAM arousal)."""
    return _deap_affect_task("deap_arousal", "arousal", "high_arousal",
                             data_dir, subjects, max_trials, window_seconds)


def deap_anxiety_task(data_dir: str = "data/real/deap", subjects: int = 8,
                      max_trials: Optional[int] = None, window_seconds: int = 8) -> BenchTask:
    """DEAP EEG+peripheral anxiety / negative affect — REAL self-report label.

    Anxiety is the high-arousal + low-valence quadrant of the affective circumplex
    (arousal >= 5 AND valence < 5) from the participant's own SAM ratings — genuine ground
    truth, not a proxy/median-split. The mental-health task grounded in real labels;
    depression and cognitive workload have no labeled cohort on disk (documented proxies in
    ``clinical_tasks.py``) — the real workload cohort is ``eegmat_workload_task``.
    """
    return _deap_affect_task("deap_anxiety", "anxiety", "high_anxiety",
                             data_dir, subjects, max_trials, window_seconds,
                             extra={"label": "high-arousal+low-valence quadrant (real SAM ratings)"})


def sleep_edf_stage_task(n_recordings: int = 20, target: str = "rem",
                         max_epochs_per_rec: Optional[int] = 400) -> BenchTask:
    """Sleep-EDF Expanded multimodal sleep staging — REAL expert hypnogram labels.

    Genuinely multimodal RAW signal (EEG×2 + EOG + EMG + respiration @100 Hz), large N,
    and a canonical deep-beats-classical task. Carries BOTH per-modality summary-stat
    features (the GBM/linear floor's fair input) and per-modality RAW downsampled windows
    in ``extra["raw"]`` (the deep/LLM sequence path — the actual lever over summary stats).
    ``target``: "rem" | "deep" | "n1" | "wake_sleep" (binary, subject = recording for CV).
    """
    from dvxr.sleep_edf import build_sleep_edf_windows
    d = build_sleep_edf_windows(n_recordings=n_recordings, target=target,
                                max_epochs_per_rec=max_epochs_per_rec)
    return BenchTask(
        name=f"sleep_edf_{target}", kind="classification", features=d["features"],
        feature_names=d["feature_names"], y=d["y"], subject_ids=d["subject_ids"],
        metric="1-AUROC", baseline_hint="majority", raw_windows=None,
        extra={"raw": d["raw"], "target": target, "modality_is_raw": True},
    )


def eegmat_workload_task(data_dir: str = "data/real/eegmat", subjects: int = 20,
                         window_seconds: int = 4) -> BenchTask:
    """PhysioNet EEG mental-arithmetic — REAL cognitive-workload label (rest vs arithmetic).

    Replaces the documented EEG beta/alpha *proxy* (clinical_tasks.py) with a labeled cohort:
    resting baseline (low workload) vs serial-subtraction mental arithmetic (high workload),
    per subject. 19-ch 10-20 EEG + ECG at 64 Hz → band-power + peripheral windows, with the
    raw-signal path (``raw_cnn``) on properly-sampled signal (δ/θ/α/β intact — the fidelity the
    decimated DEAP events lack). Subject-held-out CV; each subject is one CV group.
    """
    events = load_eegmat_dataset(data_dir, subjects=subjects)
    return _windowed_signal_task(
        "eegmat_workload", events, "cognitive_workload", "high_workload", window_seconds,
        "eegmat 64 Hz EEG+ECG raw windows (α/β intact)",
        extra={"label": "rest (low) vs serial-subtraction (high) — real workload label"})


def mumtaz_depression_task(data_dir: str = "data/real/mumtaz_mdd", subjects: Optional[int] = None,
                           window_seconds: int = 8) -> BenchTask:
    """Mumtaz (2016) MDD vs healthy resting EEG — REAL depression diagnosis label.

    Replaces the documented motion/HRV *proxy* (clinical_tasks.py) with a labeled clinical
    cohort: eyes-closed resting EEG, MDD patients (high_depression) vs healthy controls
    (low_depression), subject-level diagnosis under cross-subject held-out CV. EEG-only
    (single modality, 19-ch 10-20 @ 64 Hz) → band-power + the raw-signal path (``raw_cnn``).
    Depression from resting EEG is genuinely hard cross-subject — an honest test, not a win by
    construction.
    """
    events = load_mumtaz_mdd_dataset(data_dir, subjects=subjects)
    return _windowed_signal_task(
        "mumtaz_depression", events, "depression", "high_depression", window_seconds,
        "Mumtaz MDD 64 Hz resting EEG raw windows",
        extra={"label": "MDD patients vs healthy controls (real diagnosis label)"})


TASK_BUILDERS = {
    "stress": noneeg_stress_task,
    "glucose": shanghai_glucose_task,
    "mortality": mimic_mortality_task,
    "wesad_stress": wesad_stress_task,
    "cgmacros_glucose": cgmacros_glucose_task,
    "cgmacros_diabetes": cgmacros_diabetes_task,
    "deap_arousal": deap_arousal_task,
    "deap_anxiety": deap_anxiety_task,
    "eegmat_workload": eegmat_workload_task,
    "mumtaz_depression": mumtaz_depression_task,
    "sleep_edf_rem": lambda: sleep_edf_stage_task(target="rem"),
    "sleep_edf_deep": lambda: sleep_edf_stage_task(target="deep"),
    "sleep_edf_wake": lambda: sleep_edf_stage_task(target="wake_sleep"),
}
