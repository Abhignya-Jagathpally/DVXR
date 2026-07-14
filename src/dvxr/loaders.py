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


# --- CGMacros multimodal dataset: CGM (Libre+Dexcom) + Fitbit + diet + labs (PhysioNet, real) ---

# Per-subject CSV continuous channels: (column, modality, channel, unit, glucose_source).
CGMACROS_CONTINUOUS = [
    ("Libre GL", "cgm", "glucose", "mg/dL", "libre"),
    ("Dexcom GL", "cgm", "glucose", "mg/dL", "dexcom"),
    ("HR", "wearable_phys", "heart_rate", "bpm", ""),
    ("Calories (Activity)", "wearable_phys", "calories_activity", "kcal", ""),
    ("Mets", "wearable_phys", "mets_x10", "met_x10", ""),
]

# Per-subject CSV meal-event channels (emitted only on rows that start a meal).
CGMACROS_MEAL = [
    ("Calories", "meal_calories", "kcal"),
    ("Carbs", "meal_carbs", "g"),
    ("Protein", "meal_protein", "g"),
    ("Fat", "meal_fat", "g"),
    ("Fiber", "meal_fiber", "g"),
    ("Amount Consumed", "meal_amount_consumed", "pct"),
]

# bio.csv numeric static concepts -> EHR events (source column -> canonical channel).
CGMACROS_BIO_NUMERIC = {
    "Age": "age",
    "BMI": "bmi",
    "Body weight": "body_weight_lb",
    "Height": "height_in",
    "A1c PDL (Lab)": "hba1c",
    "Fasting GLU - PDL (Lab)": "fasting_glucose",
    "Insulin": "fasting_insulin",
    "Triglycerides": "triglycerides",
    "Cholesterol": "cholesterol",
    "HDL": "hdl",
    "Non HDL": "non_hdl",
    "LDL (Cal)": "ldl",
    "VLDL (Cal)": "vldl",
    "Cho/HDL Ratio": "chol_hdl_ratio",
}


def _diabetes_status_from_a1c(a1c: float | None) -> str:
    """ADA HbA1c(%) strata: >=6.5 diabetes, 5.7-6.4 prediabetes, else healthy."""
    if a1c is None or pd.isna(a1c):
        return ""
    if a1c >= 6.5:
        return "diabetes"
    if a1c >= 5.7:
        return "prediabetes"
    return "healthy"


def _cgmacros_subject_num(path: Path) -> str:
    import re

    match = re.search(r"(\d{2,3})", path.stem)
    return match.group(1) if match else path.stem


def _find_bio_csv(data_dir: Path) -> Path | None:
    for pattern in ("**/bio.csv", "**/Bio.csv", "**/CGMacros_bio.csv"):
        hits = [p for p in data_dir.glob(pattern) if "DataDictionary" not in p.name]
        if hits:
            return hits[0]
    return None


def _cgmacros_bio_status_map(data_dir: Path) -> dict[str, str]:
    """Map cgmacros subject_id -> diabetes status derived from bio.csv HbA1c."""
    bio_path = _find_bio_csv(data_dir)
    if bio_path is None:
        return {}
    bio = pd.read_csv(bio_path)
    bio.columns = [c.strip() for c in bio.columns]
    id_col = next((c for c in bio.columns if c.lower() in {"subject", "subjectid", "subject_id", "id", "pid", "participant"}), None)
    a1c_col = next((c for c in bio.columns if c.strip().startswith("A1c")), None)
    status: dict[str, str] = {}
    for i, record in bio.reset_index(drop=True).iterrows():
        num = str(int(record[id_col])) if id_col and pd.notna(record[id_col]) else str(i + 1)
        num = num.zfill(3)
        a1c = pd.to_numeric(record.get(a1c_col), errors="coerce") if a1c_col else None
        status[f"cgmacros_{num}"] = _diabetes_status_from_a1c(a1c)
    return status


def load_cgmacros_subject(csv_path: str | Path, diabetes_status: str = "") -> pd.DataFrame:
    """Load one CGMacros per-subject CSV into canonical events across three modalities.

    Emits `cgm` (Libre + Dexcom glucose, disambiguated by the extra `glucose_source`
    column), `wearable_phys` (Fitbit HR / activity calories / METs), and `behavior`
    (per-meal macronutrients, with `meal_type` and `meal_photo_path` extras). Uses the
    relaxed schema so those dataset-specific extra columns are preserved.

    Note: CGMacros records two simultaneous CGMs (Libre + Dexcom), so each glucose
    timestamp yields two `(cgm, glucose)` rows that share the canonical key and differ
    only by `glucose_source`. This is intentional; filter to one source (as the glucose
    benchmark task does) to avoid double-counting.
    """
    csv_path = Path(csv_path)
    subject_id = f"cgmacros_{_cgmacros_subject_num(csv_path)}"
    frame = pd.read_csv(csv_path)
    frame.columns = [c.strip() for c in frame.columns]
    # Column names vary in case across the real files (e.g. "METs", "Image path").
    lower_map = {c.lower(): c for c in frame.columns}

    def _col(name: str):
        return lower_map.get(name.lower())

    ts = pd.to_datetime(frame[_col("Timestamp")], errors="coerce", utc=True)

    parts: list[pd.DataFrame] = []

    def _emit(mask, values, modality, channel, unit, rate, glucose_source="", meal_type="", photo=""):
        mask = mask & ts.notna()
        if not mask.any():
            return
        parts.append(
            pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "session_id": "cgmacros",
                    "timestamp_utc": ts[mask].values,
                    "source_system": "cgmacros_physionet",
                    "device": "cgmacros",
                    "modality": modality,
                    "channel": channel,
                    "value": pd.to_numeric(values[mask], errors="coerce").values,
                    "unit": unit,
                    "sampling_rate_hz": rate,
                    "quality_flag": "ok",
                    "label_name": "diabetes_status" if diabetes_status else "",
                    "label_value": diabetes_status,
                    "glucose_source": glucose_source,
                    "meal_type": meal_type if isinstance(meal_type, str) else "",
                    "meal_photo_path": photo if isinstance(photo, str) else "",
                }
            )
        )

    for col, modality, channel, unit, source in CGMACROS_CONTINUOUS:
        actual = _col(col)
        if actual is None:
            continue
        values = pd.to_numeric(frame[actual], errors="coerce")
        _emit(values.notna(), values, modality, channel, unit, 1.0 / 60.0, glucose_source=source)

    meal_type_col = _col("Meal Type")
    if meal_type_col is not None:
        meal_mask = frame[meal_type_col].notna() & (frame[meal_type_col].astype(str).str.strip() != "")
        photo_col = _col("Image Path") or _col("Image path")
        for col, channel, unit in CGMACROS_MEAL:
            actual = _col(col)
            if actual is None:
                continue
            values = pd.to_numeric(frame[actual], errors="coerce")
            m = meal_mask & values.notna()
            if not (m & ts.notna()).any():
                continue
            sub = pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "session_id": "cgmacros",
                    "timestamp_utc": ts[m].values,
                    "source_system": "cgmacros_physionet",
                    "device": "diet_log",
                    "modality": "behavior",
                    "channel": channel,
                    "value": values[m].values,
                    "unit": unit,
                    "sampling_rate_hz": 0.0,
                    "quality_flag": "ok",
                    "label_name": "diabetes_status" if diabetes_status else "",
                    "label_value": diabetes_status,
                    "glucose_source": "",
                    "meal_type": frame.loc[m, meal_type_col].astype(str).values,
                    "meal_photo_path": (frame.loc[m, photo_col].astype(str).values if photo_col else ""),
                }
            )
            parts.append(sub)

    if not parts:
        raise ValueError(f"No usable CGMacros signals in {csv_path.name}")
    return validate_events(pd.concat(parts, ignore_index=True))


def load_cgmacros_bio(data_dir: str | Path) -> pd.DataFrame:
    """Load CGMacros bio.csv into static per-subject EHR events (demographics + labs)."""
    data_dir = Path(data_dir)
    bio_path = _find_bio_csv(data_dir)
    if bio_path is None:
        raise ValueError(f"No bio.csv found under {data_dir}")
    bio = pd.read_csv(bio_path)
    bio.columns = [c.strip() for c in bio.columns]
    id_col = next((c for c in bio.columns if c.lower() in {"subject", "subjectid", "subject_id", "id", "pid", "participant"}), None)
    anchor = pd.Timestamp("2021-01-01T00:00:00Z")

    rows: list[dict] = []
    for i, record in bio.reset_index(drop=True).iterrows():
        num = (str(int(record[id_col])) if id_col and pd.notna(record[id_col]) else str(i + 1)).zfill(3)
        subject_id = f"cgmacros_{num}"
        status = _diabetes_status_from_a1c(pd.to_numeric(record.get(next((c for c in bio.columns if c.startswith("A1c")), "")), errors="coerce"))
        concepts = {canonical: pd.to_numeric(record.get(src), errors="coerce") for src, canonical in CGMACROS_BIO_NUMERIC.items()}
        if "Gender" in bio.columns:
            concepts["sex_is_female"] = 1.0 if str(record.get("Gender")).strip().upper().startswith("F") else 0.0
        for channel, value in concepts.items():
            if pd.isna(value):
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": "bio",
                    "timestamp_utc": anchor,
                    "source_system": "cgmacros_physionet",
                    "device": "ehr_static",
                    "modality": "ehr",
                    "channel": channel,
                    "value": float(value),
                    "unit": "value",
                    "sampling_rate_hz": 0.0,
                    "quality_flag": "ok",
                    "label_name": "diabetes_status" if status else "",
                    "label_value": status,
                }
            )
    if not rows:
        raise ValueError(f"No numeric bio concepts parsed from {bio_path}")
    return validate_events(pd.DataFrame(rows))


def load_cgmacros_dataset(data_dir: str | Path, subjects: int | None = None, include_bio: bool = True) -> pd.DataFrame:
    """Load CGMacros per-subject CSVs (+ bio EHR) into one canonical multimodal table."""
    data_dir = Path(data_dir)
    csvs = sorted(
        (p for p in data_dir.glob("**/CGMacros-*.csv") if "DataDictionary" not in p.name),
        key=lambda p: _cgmacros_subject_num(p),
    )
    if not csvs:
        raise ValueError(f"No CGMacros-*.csv subject files found under {data_dir}")
    if subjects is not None:
        csvs = csvs[:subjects]
    status_map = _cgmacros_bio_status_map(data_dir)
    loaded_ids = {f"cgmacros_{_cgmacros_subject_num(p)}" for p in csvs}
    frames = [load_cgmacros_subject(p, diabetes_status=status_map.get(f"cgmacros_{_cgmacros_subject_num(p)}", "")) for p in csvs]
    if include_bio:
        try:
            bio = load_cgmacros_bio(data_dir)
            if subjects is not None:
                bio = bio[bio["subject_id"].isin(loaded_ids)]
            frames.append(bio)
        except ValueError:
            pass
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
                            # prefix with the sensor location so chest and wrist streams of
                            # the same modality (both have EDA/TEMP/ACC) never collide.
                            "channel": f"{device_name}_{channel}",
                            "value": float(values[sample_idx, dim_idx]),
                            "unit": _unit_for_signal(signal_name, device_name),
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


def _deap_affect_label(valence: float, arousal: float, scheme: str = "arousal") -> tuple[str, str]:
    """Derive a REAL self-report affect label from DEAP SAM ratings (1-9 scale).

    ``scheme="arousal"`` — the original binary arousal split (high vs low, threshold 5).
    ``scheme="anxiety"`` — negative-affect / anxiety operationalized as the high-arousal +
    low-valence quadrant of Russell's affective circumplex (arousal >= 5 AND valence < 5),
    which is where anxiety/fear/tension sit. Both labels come from the participant's own SAM
    ratings, so neither is a proxy/median-split — they are genuine ground truth.
    """
    if scheme == "anxiety":
        positive = arousal >= 5 and valence < 5
        return "anxiety", "high_anxiety" if positive else "low_anxiety"
    return "arousal", "high_arousal" if arousal >= 5 else "low_arousal"


def load_deap_preprocessed_pickle(path: str | Path, max_trials: int | None = 3,
                                  label_scheme: str = "arousal") -> pd.DataFrame:
    """Load one DEAP preprocessed subject file into the canonical schema.

    ``label_scheme`` selects which real self-report label to attach: ``"arousal"`` (high vs
    low arousal) or ``"anxiety"`` (high-arousal + low-valence quadrant). Both draw on the
    participant's SAM valence/arousal ratings — see ``_deap_affect_label``.
    """
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
        label_name, label = _deap_affect_label(valence, arousal, label_scheme)
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
                        "label_name": label_name,
                        "label_value": label,
                    }
                )

    return validate_events(pd.DataFrame(rows))


def _deap_ratings_map(data_dir: Path) -> dict[tuple[int, int], float]:
    """Map (participant, trial) -> arousal rating from the DEAP participant_ratings sheet.

    Raw DEAP ships ratings separately from the .bdf recordings. Returns {} if no ratings
    file is found (loader then emits without a label).
    """
    for pat in ("**/participant_ratings.csv", "**/*ratings*.csv", "**/*ratings*.xls*"):
        for path in data_dir.glob(pat):
            try:
                table = pd.read_excel(path) if path.suffix.lower().startswith(".xls") else pd.read_csv(path)
            except Exception:
                continue
            cols = {c.lower().strip(): c for c in table.columns}
            pcol = cols.get("participant_id") or cols.get("participant")
            tcol = cols.get("trial")
            acol = cols.get("arousal")
            if not (pcol and tcol and acol):
                continue
            out: dict[tuple[int, int], float] = {}
            for r in table.itertuples(index=False):
                try:
                    out[(int(getattr(r, pcol)), int(getattr(r, tcol)))] = float(getattr(r, acol))
                except Exception:
                    continue
            if out:
                return out
    return {}


def load_deap_raw_bdf(
    bdf_path: str | Path,
    ratings: dict[tuple[int, int], float] | None = None,
    max_seconds: float | None = 60.0,
    target_rate_hz: float = 128.0,
) -> pd.DataFrame:
    """Load one raw DEAP BioSemi ``.bdf`` recording into the canonical schema.

    Raw BioSemi ActiveTwo is DC-coupled and reference-free, so the signals carry large
    per-channel DC offsets and are NOT analysis-ready as read. This applies the standard
    minimal preprocessing before emitting events:

    * drop the ``Status`` trigger/stim channel (not a biosignal),
    * band-pass filter the EEG (``0.5-45 Hz``) to remove the DC offset and drift,
    * average re-reference the EEG,
    * downsample to ``target_rate_hz``.

    EEG channels keep the montage names stored in the file (BioSemi order); the remaining
    external channels (EXG/GSR/Resp/Plet/Temp) are emitted as ``physiology`` under their
    own names. Arousal label is attached from ``ratings`` when available. Requires ``mne``.

    Note: the raw Kaggle set (``sayuksh/deap-datasetraw-data``) ships ``.bdf`` signals
    ONLY — no ratings file — so it loads unlabeled and cannot drive the supervised
    ``deap_arousal`` task on its own; supply the official ``participant_ratings.csv`` or
    use the preprocessed set for labels.
    """
    import re

    import mne

    bdf_path = Path(bdf_path)
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")

    # 1. drop Status/stim trigger channels (not biosignals)
    stim = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "stim"]
    if stim:
        raw.drop_channels(stim)

    # BDF stores no channel types, so mne types every channel 'eeg'. In DEAP the first 32
    # are the real EEG montage; the rest are external sensors (EXG/GSR/Resp/Plet/Temp).
    # Re-type the peripherals to 'misc' so the EEG filter + average reference touch ONLY
    # the 32 EEG channels (mixing GSR's huge values into the reference would corrupt it).
    eeg_names = raw.ch_names[:32]
    periph = raw.ch_names[32:]
    if periph:
        raw.set_channel_types({ch: "misc" for ch in periph}, verbose="ERROR")
    eeg_set = set(eeg_names)

    # 2. band-pass the EEG to strip the DC offset/drift, then 3. average re-reference
    raw.filter(l_freq=0.5, h_freq=45.0, picks="eeg", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # 4. downsample
    if target_rate_hz and raw.info["sfreq"] > target_rate_hz:
        raw.resample(target_rate_hz, verbose="ERROR")
    rate = float(raw.info["sfreq"])
    n_keep = int(max_seconds * rate) if max_seconds else None

    pmatch = re.search(r"s(\d+)", bdf_path.stem, re.I)
    tmatch = re.search(r"trial[_-]?(\d+)", bdf_path.stem, re.I)
    participant = int(pmatch.group(1)) if pmatch else 0
    trial = int(tmatch.group(1)) if tmatch else 0
    subject_id = f"deap_s{participant:02d}"

    arousal = (ratings or {}).get((participant, trial))
    label_name, label_value = ("", "")
    if arousal is not None:
        label_name, label_value = "arousal", ("high_arousal" if arousal >= 5 else "low_arousal")

    start = pd.Timestamp("2026-01-01T00:00:00Z")
    data = raw.get_data()  # Volts; EEG now filtered + average-referenced
    parts: list[pd.DataFrame] = []
    for ch_idx, channel in enumerate(raw.ch_names):
        series = data[ch_idx]
        if n_keep:
            series = series[:n_keep]
        is_eeg = channel in eeg_set
        parts.append(
            pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "session_id": f"deap_trial_{trial:02d}",
                    "timestamp_utc": start + pd.to_timedelta(np.arange(len(series)) / rate, unit="s"),
                    "source_system": "deap_raw",
                    "device": "biosemi_activetwo",
                    "modality": "eeg" if is_eeg else "physiology",
                    "channel": channel,
                    "value": series * (1e6 if is_eeg else 1.0),  # EEG Volts -> uV
                    "unit": "uV" if is_eeg else "a.u.",
                    "sampling_rate_hz": rate,
                    "quality_flag": "ok",
                    "label_name": label_name,
                    "label_value": label_value,
                }
            )
        )
    return validate_events(pd.concat(parts, ignore_index=True))


def load_deap_dataset(
    data_dir: str | Path,
    subjects: int | None = None,
    max_trials: int | None = 3,
    label_scheme: str = "arousal",
) -> pd.DataFrame:
    """Load DEAP into the canonical schema, auto-detecting preprocessed vs raw layout.

    Prefers the preprocessed ``data_preprocessed_python/sXX.dat`` pickles (one file per
    participant, includes valence/arousal labels). Falls back to raw ``.bdf`` recordings
    (joined to the participant_ratings sheet) when no ``.dat`` files are present.

    ``label_scheme`` (``"arousal"`` | ``"anxiety"``) selects which real self-report label to
    attach; only the preprocessed path carries valence, so ``"anxiety"`` requires the ``.dat``
    set (the raw ``.bdf`` ratings sheet has arousal only).

    Note: the raw Kaggle set (``sayuksh/deap-datasetraw-data``) ships ``.bdf`` signals
    ONLY — no ratings file — so it loads unlabeled and cannot drive the supervised
    ``deap_arousal`` task on its own; supply the official ``participant_ratings.csv`` or
    use the preprocessed set for labels.
    """
    data_dir = Path(data_dir)
    dats = sorted(
        data_dir.glob("**/data_preprocessed_python/s*.dat"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    ) or sorted(data_dir.glob("**/s*.dat"))
    if dats:
        if subjects is not None:
            dats = dats[:subjects]
        frames = [load_deap_preprocessed_pickle(p, max_trials=max_trials,
                                                label_scheme=label_scheme) for p in dats]
        return validate_events(pd.concat(frames, ignore_index=True))

    if label_scheme == "anxiety":
        raise ValueError(
            "label_scheme='anxiety' needs DEAP valence, which only the preprocessed "
            "(.dat) set carries; the raw .bdf ratings sheet has arousal only."
        )
    bdfs = sorted(data_dir.glob("**/*.bdf"))
    if not bdfs:
        raise ValueError(f"No DEAP .dat or .bdf files found under {data_dir}")
    if subjects is not None:
        bdfs = bdfs[:subjects]
    ratings = _deap_ratings_map(data_dir)
    frames = [load_deap_raw_bdf(p, ratings=ratings) for p in bdfs]
    return validate_events(pd.concat(frames, ignore_index=True))


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


def _unit_for_signal(signal_name: str, location: str = "") -> str:
    if signal_name == "ACC":
        # Empatica E4 (wrist) reports ACC in 1/64 g counts; RespiBAN (chest) in raw counts.
        return "counts_1_64g" if location == "wrist" else "counts"
    return {
        "ECG": "mV",
        "EDA": "uS",
        "EMG": "mV",
        "Resp": "a.u.",
        "Temp": "C",
        "TEMP": "C",
        "BVP": "a.u.",
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


def load_eegmat_dataset(
    data_dir: str | Path = "data/real/eegmat",
    subjects: int | None = None,
    target_rate_hz: float = 64.0,
    max_seconds: float = 60.0,
) -> pd.DataFrame:
    """Load the PhysioNet EEG-during-Mental-Arithmetic cohort (``eegmat``) into canonical events.

    REAL cognitive-workload label (not a proxy): every subject has a resting-baseline
    recording (``_1`` → ``low_workload``) and a serial-subtraction mental-arithmetic recording
    (``_2`` → ``high_workload``). 19-channel 10-20 EEG + ECG @ 500 Hz; the EEG is band-passed
    (0.5-45 Hz) and average-referenced, and every channel is downsampled to ``target_rate_hz``
    (64 Hz keeps δ/θ/α/β intact for band-power). The ~3-min rest is truncated to its **last**
    ``max_seconds`` to match the ~60 s task, so neither class dominates by sheer duration.
    Each subject contributes two sessions and stays a single CV group.
    Source: https://physionet.org/content/eegmat/1.0.0/ (Zyma et al., 2019).
    """
    import mne

    data_dir = Path(data_dir)
    subs = sorted({p.stem.rsplit("_", 1)[0] for p in data_dir.glob("Subject*_*.edf")})
    if subjects:
        subs = subs[:subjects]
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    parts: list[pd.DataFrame] = []
    for sid in subs:
        for cond, label in (("1", "low_workload"), ("2", "high_workload")):
            edf = data_dir / f"{sid}_{cond}.edf"
            if not edf.exists():
                continue
            raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
            # canonical names: strip the 'EEG '/'ECG ' prefixes; drop the linked-ear reference.
            raw.rename_channels({ch: ch.split(" ", 1)[1] if " " in ch else ch
                                 for ch in raw.ch_names})
            if "A2-A1" in raw.ch_names:
                raw.drop_channels(["A2-A1"])
            eeg_names = [c for c in raw.ch_names if c.upper() != "ECG"]
            ecg_names = [c for c in raw.ch_names if c.upper() == "ECG"]
            raw.set_channel_types({**{c: "eeg" for c in eeg_names},
                                   **{c: "ecg" for c in ecg_names}}, verbose="ERROR")
            raw.filter(l_freq=0.5, h_freq=45.0, picks="eeg", verbose="ERROR")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            if target_rate_hz and raw.info["sfreq"] > target_rate_hz:
                raw.resample(target_rate_hz, verbose="ERROR")
            rate = float(raw.info["sfreq"])
            data = raw.get_data()  # Volts (EEG filtered + avg-referenced)
            n = data.shape[1]
            keep = int(max_seconds * rate) if max_seconds else n
            # rest: last `keep`; task: first `keep` (both ≈ max_seconds long).
            sl = slice(max(0, n - keep), n) if cond == "1" else slice(0, min(n, keep))
            eeg_set = set(eeg_names)
            for ci, ch in enumerate(raw.ch_names):
                series = data[ci, sl]
                is_eeg = ch in eeg_set
                parts.append(pd.DataFrame({
                    "subject_id": sid,
                    "session_id": f"eegmat_{cond}",
                    "timestamp_utc": start + pd.to_timedelta(np.arange(len(series)) / rate, unit="s"),
                    "source_system": "eegmat",
                    "device": "neurocom_eeg",
                    "modality": "eeg" if is_eeg else "physiology",
                    "channel": ch if is_eeg else ch.lower(),
                    "value": series * (1e6 if is_eeg else 1.0),  # EEG Volts -> uV
                    "unit": "uV" if is_eeg else "a.u.",
                    "sampling_rate_hz": rate,
                    "quality_flag": "ok",
                    "label_name": "cognitive_workload",
                    "label_value": label,
                }))
    if not parts:
        raise FileNotFoundError(
            f"No eegmat EDFs under {data_dir}. Fetch with: python3 scripts/fetch_data.py eegmat")
    return validate_events(pd.concat(parts, ignore_index=True))
