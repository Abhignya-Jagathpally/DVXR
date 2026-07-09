from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import validate_events


# --- PhysioNet Non-EEG stress dataset (real wearable physiology + stress labels) ---

# Map the dataset's seven labeled phases to a binary stress target for the classifier.
NONEEG_STRESS_PHASES = {"PhysicalStress", "EmotionalStress", "CognitiveStress"}

# Canonical modality / unit for each WFDB signal across the two records.
NONEEG_SIGNAL_MAP = {
    "ax": ("motion", "acc_x", "g"),
    "ay": ("motion", "acc_y", "g"),
    "az": ("motion", "acc_z", "g"),
    "temp": ("temp", "temperature", "C"),
    "EDA": ("eda", "eda", "uS"),
    "hr": ("ppg", "heart_rate", "bpm"),
    "SpO2": ("ppg", "spo2", "%"),
}


def load_noneeg_subject(
    data_dir: str | Path,
    subject: int,
    max_samples_per_signal: int | None = 6000,
) -> pd.DataFrame:
    """Load one PhysioNet Non-EEG subject (two WFDB records) into the canonical schema.

    The AccTempEDA record (8 Hz: 3-axis accel, temperature, EDA) carries .atr phase
    annotations (Relax / PhysicalStress / EmotionalStress / CognitiveStress); the SpO2HR
    record (1 Hz: SpO2, heart rate) shares the same experiment clock. Phases are mapped to
    a binary stress_state label aligned to each sample's elapsed time.
    """
    import wfdb

    data_dir = Path(data_dir)
    subject_id = f"noneeg_S{subject:02d}"
    start = pd.Timestamp("2026-01-01T00:00:00Z")

    annotation = wfdb.rdann(str(data_dir / f"Subject{subject}_AccTempEDA"), "atr")
    phase_starts_sec = np.asarray(annotation.sample, dtype=float) / 8.0  # .atr indexed at 8 Hz
    phase_labels = list(annotation.aux_note)

    rows: list[dict] = []
    for record_name in (f"Subject{subject}_AccTempEDA", f"Subject{subject}_SpO2HR"):
        record = wfdb.rdrecord(str(data_dir / record_name))
        rate = float(record.fs)
        signal = np.asarray(record.p_signal, dtype=float)
        n = signal.shape[0]
        step = max(1, n // max_samples_per_signal) if max_samples_per_signal else 1

        for col_idx, name in enumerate(record.sig_name):
            mapped = NONEEG_SIGNAL_MAP.get(name)
            if mapped is None:
                continue
            modality, channel, unit = mapped
            for sample_idx in range(0, n, step):
                elapsed = sample_idx / rate
                rows.append(
                    {
                        "subject_id": subject_id,
                        "session_id": "noneeg",
                        "timestamp_utc": start + pd.Timedelta(seconds=elapsed),
                        "source_system": "physionet_noneeg",
                        "device": "wrist_noneeg",
                        "modality": modality,
                        "channel": channel,
                        "value": float(signal[sample_idx, col_idx]),
                        "unit": unit,
                        "sampling_rate_hz": rate,
                        "quality_flag": "ok",
                        "label_name": "stress_state",
                        "label_value": _noneeg_label_at(elapsed, phase_starts_sec, phase_labels),
                    }
                )

    return validate_events(pd.DataFrame(rows))


def load_noneeg_dataset(data_dir: str | Path, subjects: int = 4) -> pd.DataFrame:
    """Load and concatenate the first N Non-EEG subjects into one canonical table."""
    frames = [load_noneeg_subject(data_dir, subject) for subject in range(1, subjects + 1)]
    return validate_events(pd.concat(frames, ignore_index=True))


def _noneeg_label_at(elapsed_sec: float, phase_starts_sec: np.ndarray, phase_labels: list[str]) -> str:
    if len(phase_starts_sec) == 0:
        return "non_stress"
    idx = int(np.searchsorted(phase_starts_sec, elapsed_sec, side="right") - 1)
    idx = min(max(idx, 0), len(phase_labels) - 1)
    return "stress" if phase_labels[idx] in NONEEG_STRESS_PHASES else "non_stress"


# --- MIMIC-IV clinical demo (real structured EHR event ingestion) ---


def load_mimic_demo_ehr(hosp_dir: str | Path, max_lab_rows: int = 40000) -> pd.DataFrame:
    """Load the open MIMIC-IV demo into canonical EHR events (labs + demographics).

    Numeric lab results become per-concept EHR events on their charttime; patient age and
    gender are added as static EHR concepts. This exercises the structured-EHR ingestion
    path on real published clinical data (no credentials required for the demo subset).
    """
    hosp_dir = Path(hosp_dir)
    labitems = pd.read_csv(hosp_dir / "d_labitems.csv.gz")
    label_by_item = labitems.set_index("itemid")["label"].to_dict()

    labs = pd.read_csv(
        hosp_dir / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    labs = labs.dropna(subset=["valuenum", "charttime"]).head(max_lab_rows)

    rows: list[dict] = []
    for record in labs.itertuples(index=False):
        concept = str(label_by_item.get(record.itemid, f"item_{record.itemid}"))
        rows.append(
            {
                "subject_id": f"mimic_{int(record.subject_id)}",
                "session_id": f"hadm_{record.hadm_id}" if pd.notna(record.hadm_id) else "outpatient",
                "timestamp_utc": pd.to_datetime(record.charttime, utc=True),
                "source_system": "mimic_iv_demo",
                "device": "ehr_lab",
                "modality": "ehr",
                "channel": _slug(concept),
                "value": float(record.valuenum),
                "unit": "lab",
                "sampling_rate_hz": 0.0,
                "quality_flag": "ok",
                "label_name": "",
                "label_value": "",
            }
        )

    patients = pd.read_csv(hosp_dir / "patients.csv.gz")
    anchor = pd.Timestamp("2150-01-01T00:00:00Z")  # MIMIC dates are deidentified/shifted
    for record in patients.itertuples(index=False):
        for concept, value in (("age", float(record.anchor_age)), ("sex_is_female", 1.0 if record.gender == "F" else 0.0)):
            rows.append(
                {
                    "subject_id": f"mimic_{int(record.subject_id)}",
                    "session_id": "demographics",
                    "timestamp_utc": anchor,
                    "source_system": "mimic_iv_demo",
                    "device": "ehr_static",
                    "modality": "ehr",
                    "channel": concept,
                    "value": value,
                    "unit": "value",
                    "sampling_rate_hz": 0.0,
                    "quality_flag": "ok",
                    "label_name": "",
                    "label_value": "",
                }
            )

    return validate_events(pd.DataFrame(rows))


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower()).strip("_")[:48] or "concept"


# --- Shanghai T1DM/T2DM diabetes dataset (real continuous glucose monitoring) ---

SHANGHAI_CGM_COLUMN = "CGM (mg / dl)"


def load_shanghai_cgm_file(path: str | Path) -> pd.DataFrame:
    """Load one Shanghai patient workbook (Date + CGM) into canonical CGM events.

    File names are ``<cohort>__<patient>_<period>_<date>.xlsx``; each patient/period
    becomes a subject/session so repeated recordings never leak across the split.
    """
    path = Path(path)
    cohort, _, remainder = path.stem.partition("__")
    parts = remainder.split("_")
    patient, period = parts[0], (parts[1] if len(parts) > 1 else "0")
    subject_id = f"shanghai_{cohort}_{patient}"

    frame = pd.read_excel(path, engine="openpyxl", usecols=["Date", SHANGHAI_CGM_COLUMN])
    frame = frame.rename(columns={"Date": "timestamp_utc", SHANGHAI_CGM_COLUMN: "value"})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value", "timestamp_utc"]).sort_values("timestamp_utc")
    if frame.empty:
        raise ValueError(f"No CGM readings in {path.name}")

    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["subject_id"] = subject_id
    frame["session_id"] = f"period_{period}"
    frame["source_system"] = "shanghai_diabetes"
    frame["device"] = f"cgm_{cohort.lower()}"
    frame["modality"] = "cgm"
    frame["channel"] = "glucose"
    frame["unit"] = "mg/dL"
    frame["sampling_rate_hz"] = 1.0 / 900.0  # 15-minute CGM cadence
    frame["quality_flag"] = "ok"
    frame["label_name"] = ""
    frame["label_value"] = ""
    return validate_events(frame)


def load_shanghai_cgm_dataset(data_dir: str | Path, max_patients: int | None = None) -> pd.DataFrame:
    """Load and concatenate Shanghai CGM workbooks into one canonical CGM table."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if max_patients is not None:
        files = files[:max_patients]
    if not files:
        raise ValueError(f"No Shanghai CGM workbooks found in {data_dir}")
    frames = [load_shanghai_cgm_file(path) for path in files]
    return validate_events(pd.concat(frames, ignore_index=True))


WESAD_RATES = {
    ("chest", "ACC"): 700.0,
    ("chest", "ECG"): 700.0,
    ("chest", "EDA"): 700.0,
    ("chest", "EMG"): 700.0,
    ("chest", "Resp"): 700.0,
    ("chest", "Temp"): 700.0,
    ("wrist", "ACC"): 32.0,
    ("wrist", "BVP"): 64.0,
    ("wrist", "EDA"): 4.0,
    ("wrist", "TEMP"): 4.0,
}

WESAD_MODALITIES = {
    "ACC": "motion",
    "ECG": "ecg",
    "EDA": "eda",
    "EMG": "emg",
    "Resp": "resp",
    "Temp": "temp",
    "TEMP": "temp",
    "BVP": "ppg",
}


def load_canonical_csv(path: str | Path) -> pd.DataFrame:
    return validate_events(pd.read_csv(path))


def load_wesad_subject_pickle(path: str | Path, max_samples_per_channel: int | None = 5000) -> pd.DataFrame:
    """Load one official WESAD subject pickle into the canonical schema."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    subject_id = str(payload.get("subject", path.stem))
    labels = np.asarray(payload.get("label", []))
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows: list[dict] = []

    for device_name, signals in payload["signal"].items():
        for signal_name, values in signals.items():
            rate = WESAD_RATES.get((device_name, signal_name))
            if rate is None:
                continue
            values = np.asarray(values)
            if values.ndim == 1:
                values = values.reshape(-1, 1)

            channel_names = _channel_names(signal_name, values.shape[1])
            step = max(1, values.shape[0] // max_samples_per_channel) if max_samples_per_channel else 1

            for sample_idx in range(0, values.shape[0], step):
                timestamp = start + pd.Timedelta(seconds=sample_idx / rate)
                label = _aligned_wesad_label(labels, sample_idx, values.shape[0])
                for dim_idx, channel in enumerate(channel_names):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "session_id": "wesad",
                            "timestamp_utc": timestamp,
                            "source_system": "wesad",
                            "device": f"wesad_{device_name}",
                            "modality": WESAD_MODALITIES.get(signal_name, signal_name.lower()),
                            "channel": channel,
                            "value": float(values[sample_idx, dim_idx]),
                            "unit": _unit_for_signal(signal_name),
                            "sampling_rate_hz": rate,
                            "quality_flag": "ok",
                            "label_name": "wesad_label",
                            "label_value": str(label),
                        }
                    )

    return validate_events(pd.DataFrame(rows))


# WESAD protocol condition codes (Schmidt et al. 2018); 5-7 are ignore/transient states.
WESAD_CONDITION_LABELS = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_wesad_dataset(
    data_dir: str | Path,
    subjects: int | None = None,
    max_samples_per_channel: int | None = 5000,
) -> pd.DataFrame:
    """Load and concatenate official WESAD subject pickles into one canonical table.

    Expects the official archive layout ``<data_dir>/S<n>/S<n>.pkl`` (as shipped in the
    Siegen ``WESAD.zip``). ``subjects`` caps how many subject pickles are read.
    """
    data_dir = Path(data_dir)
    pickles = sorted(
        data_dir.glob("S*/S*.pkl"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    if not pickles:
        raise ValueError(f"No WESAD subject pickles (S*/S*.pkl) found in {data_dir}")
    if subjects is not None:
        pickles = pickles[:subjects]
    frames = [load_wesad_subject_pickle(p, max_samples_per_channel) for p in pickles]
    return validate_events(pd.concat(frames, ignore_index=True))


def load_deap_preprocessed_pickle(path: str | Path, max_trials: int | None = 3) -> pd.DataFrame:
    """Load one DEAP preprocessed subject file into the canonical schema."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    data = np.asarray(payload["data"])
    labels = np.asarray(payload["labels"])
    subject_id = path.stem
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rate = 128.0
    rows: list[dict] = []
    channel_names = _deap_channel_names(data.shape[1])
    trial_count = min(data.shape[0], max_trials or data.shape[0])

    for trial_idx in range(trial_count):
        valence, arousal = labels[trial_idx, 0], labels[trial_idx, 1]
        label = "high_arousal" if arousal >= 5 else "low_arousal"
        trial_start = start + pd.Timedelta(minutes=trial_idx * 2)

        for channel_idx, channel in enumerate(channel_names):
            series = data[trial_idx, channel_idx]
            step = max(1, len(series) // 512)
            for sample_idx in range(0, len(series), step):
                modality = "eeg" if channel_idx < 32 else "physiology"
                rows.append(
                    {
                        "subject_id": subject_id,
                        "session_id": f"deap_trial_{trial_idx:02d}",
                        "timestamp_utc": trial_start + pd.Timedelta(seconds=sample_idx / rate),
                        "source_system": "deap",
                        "device": "deap_lab",
                        "modality": modality,
                        "channel": channel,
                        "value": float(series[sample_idx]),
                        "unit": "uV" if modality == "eeg" else "a.u.",
                        "sampling_rate_hz": rate,
                        "quality_flag": "ok",
                        "label_name": "arousal",
                        "label_value": label,
                    }
                )

    return validate_events(pd.DataFrame(rows))


def _aligned_wesad_label(labels: np.ndarray, sample_idx: int, signal_len: int) -> int:
    if len(labels) == 0:
        return -1
    label_idx = min(len(labels) - 1, int(sample_idx * len(labels) / max(signal_len, 1)))
    return int(labels[label_idx])


def _channel_names(signal_name: str, width: int) -> list[str]:
    if signal_name == "ACC" and width == 3:
        return ["acc_x", "acc_y", "acc_z"]
    if width == 1:
        return [signal_name.lower()]
    return [f"{signal_name.lower()}_{i}" for i in range(width)]


def _unit_for_signal(signal_name: str) -> str:
    return {
        "ECG": "mV",
        "EDA": "uS",
        "EMG": "mV",
        "Resp": "a.u.",
        "Temp": "C",
        "TEMP": "C",
        "BVP": "a.u.",
        "ACC": "g",
    }.get(signal_name, "value")


def _deap_channel_names(width: int) -> list[str]:
    eeg = [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
        "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
    ]
    peripheral = [f"peripheral_{i}" for i in range(max(0, width - len(eeg)))]
    return (eeg + peripheral)[:width]
