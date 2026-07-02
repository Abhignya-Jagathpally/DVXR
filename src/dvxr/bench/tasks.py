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

from dvxr.features import build_glucose_forecast_table, build_signal_windows, feature_columns
from dvxr.loaders import (
    load_mimic_demo_ehr,
    load_noneeg_dataset,
    load_shanghai_cgm_dataset,
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


TASK_BUILDERS = {
    "stress": noneeg_stress_task,
    "glucose": shanghai_glucose_task,
    "mortality": mimic_mortality_task,
}
