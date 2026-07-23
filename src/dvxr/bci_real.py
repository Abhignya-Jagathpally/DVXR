"""Real-device BCI ingestion, epoching, features, and manifold embedding.

This module operates directly on the *collected* recordings shipped in
``data/*.zip`` (no external/stale paths):

* **EMOTIV EPOC X** export (``EmotivBCI-*.zip``) — 14-ch EEG @128 Hz with the
  built-in Mental Command stream (Neutral / Left / Right / Push / Pull), Emotiv's
  own band-power (POW.*) and Performance-Metrics (PM.*) streams, and motion.
  This is the supervised *avatar-control* analog: decode intended cube movement
  from EEG, mirroring the real-time neural-manifold decoding of Busch et al.
  (avatarRT / MRAE / TPHATE).
* **Galea / OpenBCI** BrainFlow export (``OpenBCISession_*.zip``) — 16-ch EEG
  @~250 Hz resting recording, used for multi-device ingestion + signal-quality
  reporting + an unsupervised manifold.

The manifold embedding (:func:`temporal_diffusion_map`) is a self-contained,
dependency-light diffusion-map that is *PHATE/TPHATE-inspired*: it blends a
feature-space affinity (adaptive-bandwidth kernel over k-NN) with an optional
temporal-adjacency affinity, then embeds the diffusion operator. It needs only
numpy/scipy, so it runs anywhere without the PHATE package.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
_np_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))  # numpy 1.x/2.x compat
import pandas as pd
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCX_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Emotiv Mental Command action code -> human label (from export JSON metadata)
MC_ACTION_MAP = {1: "Neutral", 2: "Push", 4: "Pull", 8: "Lift", 16: "Drop", 32: "Left", 64: "Right"}

# The five commands the subject actually trained in this session.
COMMAND_CLASSES = ["Neutral", "Left", "Right", "Push", "Pull"]

# Band definitions mirroring Emotiv's POW.* output (theta/alpha/betaL/betaH/gamma).
EEG_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "betaL": (12.0, 16.0),
    "betaH": (16.0, 25.0),
    "gamma": (25.0, 45.0),
}


# ---------------------------------------------------------------------------
# Recording containers
# ---------------------------------------------------------------------------


@dataclass
class EmotivRecording:
    eeg: pd.DataFrame          # columns: t (float s), <14 channel names> in uV
    fs: float                  # EEG sampling rate (Hz)
    ch_names: list[str]
    mc: pd.DataFrame           # columns: t, action_code, action, power
    pow: pd.DataFrame          # columns: t, POW.<ch>.<band> (Emotiv FFT band power)
    pm: pd.DataFrame           # columns: t, stress, engagement, excitement, ...
    motion: pd.DataFrame = field(default_factory=pd.DataFrame)  # t, acc/quaternion/mag
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.eeg["t"].iloc[-1] - self.eeg["t"].iloc[0])


@dataclass
class GaleaRecording:
    eeg: pd.DataFrame          # columns: t, eeg_1..eeg_N
    fs: float
    ch_names: list[str]
    quality: pd.DataFrame      # per-channel signal-quality summary
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.eeg["t"].iloc[-1] - self.eeg["t"].iloc[0])


# ---------------------------------------------------------------------------
# EMOTIV ingestion
# ---------------------------------------------------------------------------


def _emotiv_csv_member(zf: zipfile.ZipFile) -> str:
    members = [n for n in zf.namelist() if n.endswith(".csv") and "intervalMarker" not in n]
    if not members:
        raise FileNotFoundError("No EMOTIV data CSV found in zip")
    return members[0]


def _parse_emotiv_header(line: str) -> dict:
    """Parse the EMOTIV title row (key:value comma-separated) into a dict."""
    meta: dict = {}
    for part in line.strip().split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def _emotiv_text(source: str | Path) -> tuple[str, str]:
    """Return (csv_text, source_name) for a .zip, a directory, or a direct CSV."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _emotiv_csv_member(zf)
            with zf.open(member) as f:
                return f.read().decode("utf-8", errors="replace"), source.name
    if source.is_dir():
        cands = [p for p in sorted(source.rglob("*.csv"))
                 if "intervalMarker" not in p.name]
        if not cands:
            raise FileNotFoundError(f"No EMOTIV data CSV under {source}")
        return cands[0].read_text(errors="replace"), cands[0].name
    if source.suffix.lower() == ".csv":
        return source.read_text(errors="replace"), source.name
    raise FileNotFoundError(f"Unsupported EMOTIV source (need .zip, dir, or .csv): {source}")


def ingest_emotiv(source: str | Path) -> EmotivRecording:
    """Load an EMOTIV EPOC X export into an :class:`EmotivRecording`.

    ``source`` may be the ``.zip`` export, a directory containing the exported CSV
    (e.g. ``data/sample/emotiv``), or a direct path to the ``.md.mc.pm.fe.bp.csv``.
    """
    text, source_name = _emotiv_text(source)
    header_line, _, _ = text.partition("\n")
    meta = _parse_emotiv_header(header_line)

    # The real column header is the 2nd line; read everything after the title row.
    df = pd.read_csv(io.StringIO(text), skiprows=1, low_memory=False)

    fs = 128.0
    if "sampling rate" in meta:
        for tok in meta["sampling rate"].split(";"):
            if tok.startswith("eeg_"):
                fs = float(tok.split("_")[1])

    t = df["Timestamp"].astype(float).to_numpy()
    t0 = t[0]
    t = t - t0  # seconds from recording start

    eeg_cols = [f"EEG.{c}" for c in EPOCX_CHANNELS]
    eeg = pd.DataFrame({"t": t})
    for ch, col in zip(EPOCX_CHANNELS, eeg_cols):
        eeg[ch] = df[col].astype(float).to_numpy()

    # Mental-command stream (sampled ~8 Hz; rows without a sample are NaN).
    mc_mask = df["MC.Action"].notna()
    mc = pd.DataFrame({
        "t": t[mc_mask.to_numpy()],
        "action_code": df.loc[mc_mask, "MC.Action"].astype(float).astype(int).to_numpy(),
        "power": df.loc[mc_mask, "MC.ActionPower"].astype(float).to_numpy(),
    })
    mc["action"] = mc["action_code"].map(MC_ACTION_MAP).fillna("Unknown")

    # Emotiv FFT band-power stream (POW.*), sampled ~8 Hz.
    pow_cols = [c for c in df.columns if c.startswith("POW.")]
    pmask = df[pow_cols[0]].notna() if pow_cols else pd.Series(False, index=df.index)
    pow_df = pd.DataFrame({"t": t[pmask.to_numpy()]}) if pow_cols else pd.DataFrame({"t": []})
    for c in pow_cols:
        pow_df[c] = df.loc[pmask, c].astype(float).to_numpy()

    # Performance-metrics stream (PM.*), sampled ~2 Hz. Only metrics whose
    # IsActive flag is set carry data; for each, prefer Scaled, fall back to Raw.
    # (In EmotivBCI exports often only Excitement is active.)
    metrics = ["stress", "engagement", "excitement", "relaxation",
               "attention", "interest", "focus"]
    have: dict[str, str] = {}
    for m in metrics:
        for suffix in ("Scaled", "Raw"):
            col = f"PM.{m.capitalize()}.{suffix}"
            if col in df.columns and df[col].notna().any():
                have[m] = col
                break
    if "PM.LongTermExcitement" in df.columns and df["PM.LongTermExcitement"].notna().any():
        have["long_term_excitement"] = "PM.LongTermExcitement"
    if have:
        first = next(iter(have.values()))
        pmm = df[first].notna()
        pm = pd.DataFrame({"t": t[pmm.to_numpy()]})
        for k, v in have.items():
            pm[k] = df.loc[pmm, v].astype(float).to_numpy()
    else:
        pm = pd.DataFrame({"t": []})

    # Motion stream (MOT.*), sampled ~64 Hz: accelerometer, quaternion, magnetometer.
    mot_cols = [c for c in df.columns if c.startswith("MOT.")
                and any(k in c for k in ("Acc", "Q", "Mag"))]
    if mot_cols:
        mmask = df[mot_cols[0]].notna()
        motion = pd.DataFrame({"t": t[mmask.to_numpy()]})
        for c in mot_cols:
            motion[c.replace("MOT.", "mot_")] = df.loc[mmask, c].astype(float).to_numpy()
    else:
        motion = pd.DataFrame({"t": []})

    meta_out = {
        "device": "EMOTIV EPOC X",
        "serial": meta.get("headset serial"),
        "firmware": meta.get("headset firmware"),
        "start_timestamp": meta.get("start timestamp"),
        "samples": meta.get("samples"),
        "source": source_name,
    }
    return EmotivRecording(eeg=eeg, fs=fs, ch_names=list(EPOCX_CHANNELS),
                           mc=mc, pow=pow_df, pm=pm, motion=motion, meta=meta_out)


# ---------------------------------------------------------------------------
# GALEA / OpenBCI ingestion
# ---------------------------------------------------------------------------


def _galea_raw(source: str | Path) -> tuple[pd.DataFrame, str]:
    """Return (raw BrainFlow DataFrame, session_name) for a .zip, directory, or CSV."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = sorted(n for n in zf.namelist()
                             if "BrainFlow-RAW" in n and n.endswith(".csv"))
            if not members:
                raise FileNotFoundError("No BrainFlow-RAW CSV in Galea zip")
            data = zf.read(members[0]).decode("utf-8", errors="replace")
            session = members[0].split("/")[0]
    elif source.is_dir():
        cands = sorted(source.rglob("BrainFlow-RAW*.csv"))
        if not cands:
            raise FileNotFoundError(f"No BrainFlow-RAW CSV under {source}")
        data = cands[0].read_text(errors="replace")
        session = cands[0].parent.name
    elif source.suffix.lower() == ".csv":
        data = source.read_text(errors="replace")
        session = source.parent.name
    else:
        raise FileNotFoundError(f"Unsupported Galea source (need .zip, dir, or .csv): {source}")
    first = data.split("\n", 1)[0]
    sep = "\t" if first.count("\t") >= first.count(",") else ","
    return pd.read_csv(io.StringIO(data), sep=sep, header=None), session


def ingest_galea(source: str | Path, max_seconds: float | None = None) -> GaleaRecording:
    """Load the first BrainFlow session from a Galea/OpenBCI ``.zip``, directory, or CSV.

    BrainFlow columns: col0 = sample index, col1..N = channel volts, a unix-epoch
    timestamp column (~1.7e9), plus aux/marker columns. We auto-detect the
    timestamp column and treat the leading high-amplitude columns as EEG.
    """
    raw, session = _galea_raw(source)

    # Locate the unix-epoch timestamp column.
    tcol = None
    for c in raw.columns:
        med = raw[c].median()
        if 1.6e9 < med < 2.0e9:
            tcol = c
            break
    if tcol is None:
        raise ValueError("Could not locate a unix-epoch timestamp column in Galea CSV")

    t = raw[tcol].astype(float).to_numpy()
    t = t - t[0]
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    fs = float(round((len(t) - 1) / span)) if span > 0 else 250.0

    # BrainFlow EXG channels are columns 1..16 (col0 is the sample counter).
    n_ch = min(16, tcol - 1) if tcol > 1 else 16
    ch_idx = list(range(1, 1 + n_ch))
    ch_names = [f"eeg_{i}" for i in ch_idx]

    eeg = pd.DataFrame({"t": t})
    for i, name in zip(ch_idx, ch_names):
        eeg[name] = raw[i].astype(float).to_numpy()

    if max_seconds is not None:
        keep = eeg["t"] <= max_seconds
        eeg = eeg[keep].reset_index(drop=True)

    # Signal-quality summary: a channel railed at the ADC limit (|x| ~ saturation,
    # near-zero variance after clipping) is flagged as poor contact.
    rows = []
    for name in ch_names:
        x = eeg[name].to_numpy()
        rail = float(np.mean(np.abs(np.abs(x) - np.median(np.abs(x))) < 1e-6))
        rows.append({
            "channel": name,
            "std_uv": float(np.std(x)),
            "ptp_uv": float(np.ptp(x)),
            "frac_railed": rail,
            "usable": bool(np.std(x) > 1.0 and rail < 0.5),
        })
    quality = pd.DataFrame(rows)

    meta = {"device": "Galea (OpenBCI BrainFlow)", "source": str(Path(source).name),
            "session": session}
    return GaleaRecording(eeg=eeg, fs=fs, ch_names=ch_names, quality=quality, meta=meta)


# ---------------------------------------------------------------------------
# Epoching + features
# ---------------------------------------------------------------------------


def _bandpower_welch(x: np.ndarray, fs: float) -> dict[str, float]:
    """Welch PSD band power for a single 1-D EEG window (relative power)."""
    nper = min(len(x), int(fs * 1.0))
    if nper < 16:
        return {b: 0.0 for b in EEG_BANDS}
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nper)
    total = _np_trapz(psd, freqs) + 1e-12
    out = {}
    for band, (lo, hi) in EEG_BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[band] = float(_np_trapz(psd[m], freqs[m]) / total) if m.any() else 0.0
    return out


def _label_window(mc: pd.DataFrame, t0: float, t1: float, power_thresh: float) -> tuple[str, str, float]:
    """Assign a 5-class label to the window [t0, t1).

    Returns (label, trial_key, mean_power). A window is a *command* if an active
    non-Neutral command (power > thresh) dominates its mental-command samples;
    otherwise it is Neutral. The trial_key groups temporally-contiguous windows of
    the same command so leakage-controlled CV can keep a trial intact.
    """
    seg = mc[(mc["t"] >= t0) & (mc["t"] < t1)]
    if seg.empty:
        return "Neutral", "neutral", 0.0
    active = seg[(seg["action"] != "Neutral") & (seg["power"] > power_thresh)]
    if len(active) >= max(1, int(0.34 * len(seg))):
        top = active["action"].mode().iloc[0]
        return top, top, float(active["power"].mean())
    return "Neutral", "neutral", 0.0


def epoch_emotiv(
    rec: EmotivRecording,
    win_s: float = 2.0,
    step_s: float = 0.5,
    power_thresh: float = 0.05,
) -> pd.DataFrame:
    """Slice the EMOTIV EEG into labeled overlapping windows with features.

    Each row = one window with:
      * ``eeg_<ch>_<band>`` — relative Welch band power (recomputed from raw EEG)
      * ``pow_<ch>_<band>`` — Emotiv's own FFT band power averaged over the window
      * ``label`` (5-class), ``trial_id`` (leakage-control group), timing, power.
    """
    fs = rec.fs
    t = rec.eeg["t"].to_numpy()
    X = rec.eeg[rec.ch_names].to_numpy()  # (n_samples, n_ch)

    pow_cols = [c for c in rec.pow.columns if c.startswith("POW.")]
    pow_t = rec.pow["t"].to_numpy() if len(rec.pow) else np.array([])

    mot_cols = [c for c in rec.motion.columns if c.startswith("mot_")] if len(rec.motion) else []
    mot_t = rec.motion["t"].to_numpy() if len(rec.motion) else np.array([])
    pm_cols = [c for c in rec.pm.columns if c != "t"] if len(rec.pm) else []
    pm_t = rec.pm["t"].to_numpy() if len(rec.pm) else np.array([])

    win_n = int(round(win_s * fs))
    step_n = int(round(step_s * fs))
    rows: list[dict] = []
    last_label, trial_counter = None, 0

    start = 0
    while start + win_n <= len(t):
        t0, t1 = t[start], t[start + win_n - 1]
        label, trial_key, mpow = _label_window(rec.mc, t0, t1, power_thresh)

        # Leakage-control trial id: increment whenever the label changes.
        if label != last_label:
            trial_counter += 1
            last_label = label
        trial_id = f"{trial_key}_{trial_counter}"

        row = {
            "t_center": float((t0 + t1) / 2.0),
            "t_start": float(t0),
            "label": label,
            "trial_id": trial_id,
            "cmd_power": mpow,
        }
        # Welch band power per channel from raw EEG.
        seg = X[start:start + win_n]
        for ci, ch in enumerate(rec.ch_names):
            bp = _bandpower_welch(seg[:, ci], fs)
            for band, val in bp.items():
                row[f"eeg_{ch}_{band}"] = val
        # Emotiv POW band power averaged over the window.
        if len(pow_t):
            pmask = (pow_t >= t0) & (pow_t < t1)
            if pmask.any():
                pseg = rec.pow.loc[pmask, pow_cols].mean()
                for c in pow_cols:
                    row[f"pow_{c[4:].replace('.', '_')}"] = float(pseg[c])
        # Motion features (accelerometer/quaternion/magnetometer): mean + std.
        if mot_cols:
            mmask = (mot_t >= t0) & (mot_t < t1)
            if mmask.any():
                mseg = rec.motion.loc[mmask, mot_cols]
                for c in mot_cols:
                    row[f"{c}_mean"] = float(mseg[c].mean())
                    row[f"{c}_std"] = float(mseg[c].std())
        # Affective performance-metrics (PM.*): window mean of each scaled score.
        if pm_cols:
            qmask = (pm_t >= t0) & (pm_t < t1)
            if qmask.any():
                qseg = rec.pm.loc[qmask, pm_cols].mean()
                for c in pm_cols:
                    row[f"pm_{c}"] = float(qseg[c])
        rows.append(row)
        start += step_n

    win = pd.DataFrame(rows).fillna(0.0)
    return win


def feature_cols(win: pd.DataFrame, kind: str = "welch") -> list[str]:
    """Return feature columns by modality.

    'welch' raw-EEG bandpower · 'pow' Emotiv FFT · 'eeg' both EEG sets ·
    'motion' MOT.* accelerometer/quaternion/mag · 'pm' PM.* affective metrics ·
    'all' everything.
    """
    if kind == "welch":
        return [c for c in win.columns if c.startswith("eeg_")]
    if kind == "pow":
        return [c for c in win.columns if c.startswith("pow_")]
    if kind == "eeg":
        return [c for c in win.columns if c.startswith("eeg_") or c.startswith("pow_")]
    if kind == "motion":
        return [c for c in win.columns if c.startswith("mot_")]
    if kind == "pm":
        return [c for c in win.columns if c.startswith("pm_")]
    if kind == "all":
        return [c for c in win.columns if c.split("_")[0] in ("eeg", "pow", "mot", "pm")]
    return [c for c in win.columns if c.startswith("eeg_") or c.startswith("pow_")]


# ---------------------------------------------------------------------------
# Manifold embedding (PHATE / TPHATE-inspired diffusion map)
# ---------------------------------------------------------------------------


def temporal_diffusion_map(
    X: np.ndarray,
    n_components: int = 3,
    k: int = 15,
    t_diffusion: int = 8,
    temporal_weight: float = 0.0,
    times: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Embed feature windows with a PHATE/TPHATE-inspired diffusion map.

    1. z-score features, build an adaptive-bandwidth Gaussian affinity over each
       point's k-NN distances (the PHATE alpha-decay kernel, simplified).
    2. optionally blend a temporal affinity (Gaussian on |t_i - t_j|) — the TPHATE
       idea that neural states evolve smoothly in time.
    3. row-normalise to a Markov diffusion operator P, raise to ``t_diffusion``
       steps, and take the leading non-trivial eigenvectors scaled by their
       eigenvalues as coordinates.

    Returns an ``(n_samples, n_components)`` embedding. Pure numpy/scipy — no PHATE
    package required.
    """
    rng = np.random.default_rng(seed)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = Z.shape[0]
    if n < n_components + 2:
        return np.zeros((n, n_components))

    # Pairwise Euclidean distances in feature space.
    sq = np.sum(Z**2, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    D = np.sqrt(D2)

    # Adaptive bandwidth = distance to the k-th nearest neighbour.
    kk = min(k, n - 1)
    sigma = np.sort(D, axis=1)[:, kk][:, None] + 1e-9
    A = np.exp(-(D2) / (sigma * sigma.T))

    if temporal_weight > 0 and times is not None:
        dt = np.abs(times[:, None] - times[None, :])
        tau = np.median(np.diff(np.sort(times))) * 5.0 + 1e-9
        At = np.exp(-(dt**2) / (2.0 * tau**2))
        A = (1.0 - temporal_weight) * A + temporal_weight * At

    A = 0.5 * (A + A.T)
    deg = A.sum(1, keepdims=True) + 1e-12
    P = A / deg

    # Diffuse t steps via eigdecomposition of the (symmetrically-normalisable) P.
    d = deg.ravel()
    Dm = 1.0 / np.sqrt(d)
    S = (Dm[:, None] * A) * Dm[None, :]      # symmetric, same spectrum as P
    S = 0.5 * (S + S.T)
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # Map back to P's right eigenvectors; skip the trivial leading component.
    psi = Dm[:, None] * evecs
    coords = psi[:, 1:1 + n_components] * (evals[1:1 + n_components] ** t_diffusion)[None, :]
    # Stabilise sign/scale.
    coords = coords - coords.mean(0)
    scale = np.std(coords, axis=0) + 1e-12
    return coords / scale * 1.0
