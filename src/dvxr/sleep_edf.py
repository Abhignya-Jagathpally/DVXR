"""dvxr.sleep_edf — Sleep-EDF Expanded (PhysioNet) loader for the multimodal win benchmark.

Why this dataset (per the model's requirements): it is genuinely MULTIMODAL raw signal
(2× EEG + EOG + EMG + respiration at 100 Hz) with expert sleep-stage labels per 30 s epoch,
it is LARGE (tens of thousands of labelled windows across recordings), and sleep staging is a
canonical *deep-beats-classical* task — so a sequence/LLM model has an honest shot at beating
the tuned-GBM floor here, unlike the tiny summary-stat tasks. It also streams over the night
(30 s epochs → a live hypnogram) for the real-time dashboard.

This bypasses the per-sample canonical long format (a full night × multi-channel × 100 Hz is
billions of rows) and windows directly with MNE's standard sleep epoching, exposing BOTH:
  * per-modality **summary-stat features** (means/std/ptp + relative EEG band powers) — the
    fair input for the classical GBM/linear floor and the current bench pipeline;
  * per-modality **raw window arrays** (downsampled fixed-length signal) in ``extra["raw"]`` —
    the input for the deep/LLM sequence path (the actual lever over summary-stat GBMs).

Requires ``mne`` (already a dependency). Fetch uses ``mne.datasets.sleep_physionet``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# canonical Sleep-EDF channels -> our modality names
_CH_MODALITY = {
    "EEG Fpz-Cz": "eeg", "EEG Pz-Oz": "eeg",
    "EOG horizontal": "eog",
    "EMG submental": "emg",
    "Resp oro-nasal": "resp",
}
# hypnogram annotation description -> coarse stage
_STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3",  # merge 3+4 (AASM N3)
    "Sleep stage R": "REM",
}
_EEG_BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 12),
              ("sigma", 12, 16), ("beta", 16, 30)]
EPOCH_SECONDS = 30.0


def local_sleep_edf_pairs() -> List[Tuple[str, str]]:
    """(PSG, hypnogram) pairs already present in the MNE data dir — no download. Lets the
    benchmark run on whatever has been fetched so far (PhysioNet throttles hard)."""
    from pathlib import Path
    root = Path.home() / "mne_data" / "physionet-sleep-data"
    if not root.exists():
        return []
    pairs = []
    for psg in sorted(root.glob("*-PSG.edf")):
        stem = psg.name[:6]  # e.g. SC4001
        hyp = next(iter(sorted(root.glob(f"{stem}*-Hypnogram.edf"))), None)
        if hyp is not None:
            pairs.append((str(psg), str(hyp)))
    return pairs


def fetch_sleep_edf(n_recordings: int = 20, download: bool = False) -> List[Tuple[str, str]]:
    """Return up to ``n_recordings`` (PSG, hypnogram) pairs. By default uses ONLY locally
    present recordings (no blocking on the slow PhysioNet fetch); ``download=True`` fetches
    the first ``n_recordings`` subjects via MNE."""
    if not download:
        local = local_sleep_edf_pairs()
        if local:
            return local[:n_recordings]
    from mne.datasets.sleep_physionet.age import fetch_data
    return [tuple(p) for p in fetch_data(subjects=list(range(n_recordings)),
                                         recording=[1], on_missing="warn", verbose=False)]


def _epoch_features(epoch: np.ndarray, sfreq: float, is_eeg: bool) -> np.ndarray:
    """Summary-stat features for one (n_ch, n_times) modality epoch: per-channel
    mean/std/ptp (+ relative band powers for EEG). This is the FLOOR's fair input."""
    feats: List[float] = []
    for ch in epoch:
        feats += [float(ch.mean()), float(ch.std()), float(np.ptp(ch))]
        if is_eeg:
            from mne.time_frequency import psd_array_welch
            psd, freqs = psd_array_welch(ch[None, :], sfreq=sfreq, fmin=0.5, fmax=30,
                                         n_fft=min(256, len(ch)), verbose=False)
            psd = psd[0]
            total = psd.sum() or 1.0
            for _, lo, hi in _EEG_BANDS:
                m = (freqs >= lo) & (freqs < hi)
                feats.append(float(psd[m].sum() / total))
    return np.asarray(feats, dtype=np.float32)


def build_sleep_edf_windows(n_recordings: int = 20, target: str = "wake_sleep",
                            max_epochs_per_rec: Optional[int] = 400,
                            raw_rate_hz: float = 32.0, seed: int = 7) -> dict:
    """Window every recording into 30 s epochs; return per-modality summary-stat features,
    raw downsampled windows, binary labels, and subject ids.

    ``target``: "wake_sleep" (W vs sleep), "rem" (REM vs rest), "deep" (N3 vs rest),
    "n1" (N1 vs rest). Returns dict(features, raw, feature_names, y, subject_ids, modalities).
    """
    import warnings

    import mne

    pairs = fetch_sleep_edf(n_recordings)
    modalities = ["eeg", "eog", "emg", "resp"]
    feats: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    raws: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    ys: List[int] = []
    sids: List[str] = []
    fnames: Dict[str, List[str]] = {}

    pos_map = {"wake_sleep": {"W"}, "rem": {"REM"}, "deep": {"N3"}, "n1": {"N1"}}
    positive = pos_map[target]
    # for wake_sleep the positive class is "sleep" (not W), so invert below.

    rng = np.random.default_rng(seed)
    for ri, (psg, hyp) in enumerate(pairs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
            ann = mne.read_annotations(hyp)
            raw.set_annotations(ann, emit_warning=False, verbose="ERROR")
        keep = [c for c in raw.ch_names if c in _CH_MODALITY]
        raw.pick(keep)
        sfreq = float(raw.info["sfreq"])
        chan_mod = {c: _CH_MODALITY[c] for c in raw.ch_names}

        # events from annotations, one per 30 s stage epoch
        event_id = {d: i for i, d in enumerate(sorted(_STAGE))}
        try:
            events, _ = mne.events_from_annotations(
                raw, event_id={d: event_id[d] for d in _STAGE}, chunk_duration=EPOCH_SECONDS,
                verbose="ERROR")
        except ValueError:
            continue
        inv = {v: k for k, v in event_id.items()}
        stage_of = {ev_code: _STAGE[inv[ev_code]] for ev_code in np.unique(events[:, 2])
                    if inv.get(ev_code) in _STAGE}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epochs = mne.Epochs(raw, events, tmin=0.0, tmax=EPOCH_SECONDS - 1.0 / sfreq,
                                baseline=None, preload=True, verbose="ERROR",
                                on_missing="ignore")
        codes = epochs.events[:, 2]
        idx = np.arange(len(codes))
        if max_epochs_per_rec and len(idx) > max_epochs_per_rec:
            idx = np.sort(rng.choice(idx, size=max_epochs_per_rec, replace=False))

        n_ds = int(round(EPOCH_SECONDS * raw_rate_hz))
        data = epochs.get_data(copy=False)  # (n_epochs, n_ch, n_times)
        ch_names = epochs.ch_names
        mod_ch = {m: [i for i, c in enumerate(ch_names) if chan_mod[c] == m] for m in modalities}

        for e in idx:
            stg = stage_of.get(codes[e])
            if stg is None:
                continue
            label = (stg != "W") if target == "wake_sleep" else (stg in positive)
            for m in modalities:
                chs = mod_ch[m]
                if not chs:
                    # modality missing in this recording -> zeros (rare); keep width stable
                    ep = np.zeros((1, data.shape[2]), dtype=np.float32)
                else:
                    ep = data[e, chs, :].astype(np.float32) * 1e6  # V -> uV
                feats[m].append(_epoch_features(ep, sfreq, is_eeg=(m == "eeg")))
                # raw window: downsample each channel to n_ds, flatten channels
                ds = np.stack([np.interp(np.linspace(0, ep.shape[1] - 1, n_ds),
                                         np.arange(ep.shape[1]), ch) for ch in ep])
                raws[m].append(ds.reshape(-1).astype(np.float32))
            ys.append(int(label))
            sids.append(f"sleep_sc{ri:02d}")

    # stack; pad modality feature widths to a common size per modality (already uniform)
    out_feats = {m: np.vstack(feats[m]) for m in modalities}
    out_raw = {m: np.vstack(raws[m]) for m in modalities}
    for m in modalities:
        w = out_feats[m].shape[1]
        fnames[m] = [f"{m}_f{i}" for i in range(w)]
    return {
        "features": out_feats, "raw": out_raw, "feature_names": fnames,
        "y": np.asarray(ys, dtype=int), "subject_ids": np.asarray(sids),
        "modalities": modalities, "target": target,
    }
