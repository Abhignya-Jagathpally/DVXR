"""dvxr.bench.labram_bench — real LaBraM EEG foundation model as a frozen bench encoder.

`pred_labram` extracts the raw EEG windows for a task, resamples each window to
``window_seconds`` one-second patches at LaBraM's nominal 200 Hz, runs the frozen vendored
LaBraM (dvxr.encoders.labram_real) to a 200-d CLS embedding per window (computed once over all
rows — unsupervised, leak-free, like the MOMENT SOTA embedding), and fits the SAME shared head
per fold. It competes on the exact folds against `single:eeg` (band-power) and `raw_cnn`.

HONEST FIDELITY CAVEAT (must accompany any result): the EEG cohorts here are 64 Hz
(eegmat, mumtaz), so real spectral content is ≤32 Hz. LaBraM was pretrained on 200 Hz EEG
(≤100 Hz content). Resampling 64 Hz → nominal 200 Hz gives LaBraM its expected patch structure
but NOT its expected bandwidth — this is a fidelity-limited test (the same sub-Nyquist story as
the DEAP decimation finding). A tie/loss here is about the data rate, not the FM or the vendored
forward (which is validated by strict load + non-degeneracy + this above-chance decoding).
"""
from __future__ import annotations

import numpy as np

from dvxr.bench.representations import _fit_head
from dvxr.bench.tasks import BenchTask

_ENC = None  # process-level cache of the loaded LaBraM encoder


def _encoder():
    global _ENC
    if _ENC is None:
        from dvxr.encoders.labram_real import LaBraMEncoder
        _ENC = LaBraMEncoder.from_pretrained()
    return _ENC


def _eeg_channel_names(task: BenchTask):
    """Channel names for the EEG modality, in build_raw_windows' sorted-unique order."""
    events = task.extra.get("events")
    if events is None:
        return None
    ev = events[events["modality"] == "eeg"]
    return sorted(ev["channel"].unique().tolist())


def labram_embeddings(task: BenchTask) -> np.ndarray:
    """Frozen LaBraM CLS embedding (N, 200) for every window; cached on the task.
    Raises cleanly (run.py logs + marks the config unstable) if EEG/raw is unavailable."""
    if "_labram_emb" in task.extra:
        return task.extra["_labram_emb"]
    from dvxr.features import build_raw_windows

    if "eeg" not in task.modalities:
        raise RuntimeError("labram: task has no EEG modality")
    win = task.raw_windows
    events = task.extra.get("events")
    if win is None or events is None:
        raise RuntimeError("labram: task lacks raw_windows/events for EEG extraction")
    ch_names = _eeg_channel_names(task)
    wsec = int(task.extra.get("window_seconds", 4))
    samples = max(200, wsec * 200)                       # >=1 one-second patch at 200 Hz
    raw, _ch = build_raw_windows(events, win, modalities=["eeg"], samples=samples)
    arr = raw["eeg"]                                     # (N, C*samples), channel-major
    n = arr.shape[0]
    c = len(ch_names)
    eeg = arr.reshape(n, c, samples).astype(np.float32)
    # per-window per-channel z-score (scale-invariant, unsupervised — matches EEG-FM practice)
    mu = eeg.mean(axis=2, keepdims=True)
    sd = eeg.std(axis=2, keepdims=True)
    eeg = (eeg - mu) / np.clip(sd, 1e-6, None)
    emb = _encoder().embed(eeg, ch_names, patch_size=200)
    task.extra["_labram_emb"] = emb
    task.extra["_labram_note"] = (f"LaBraM frozen CLS, {c}ch, {wsec}s→{samples//200} patches "
                                  f"@200Hz nominal (source ≤32Hz, fidelity-limited)")
    return emb


def pred_labram(task: BenchTask, tr, te, seed: int = 7) -> np.ndarray:
    """Frozen LaBraM embedding → shared head (train-only fit). Classification tasks only."""
    if task.kind != "classification":
        raise RuntimeError("labram: classification tasks only")
    emb = labram_embeddings(task)
    return _fit_head(task.kind, emb[tr], task.y[tr], emb[te], seed=seed)


def _weights_reachable() -> bool:
    """True when the LaBraM weights are already cached or downloads are allowed — avoids
    registering a config that would then fail every fold in a truly offline environment."""
    import os
    from pathlib import Path

    from dvxr.encoders.labram_real import LABRAM_REPO
    cache = (Path.home() / ".cache" / "huggingface" / "hub"
             / f"models--{LABRAM_REPO.replace('/', '--')}")
    return cache.exists() or bool(os.environ.get("DVXR_LABRAM_ALLOW_DOWNLOAD"))


def labram_bench_available(task: BenchTask) -> bool:
    """True when the encoder deps + weights + this task's EEG/raw are all present."""
    from dvxr.encoders.labram_real import labram_available
    return (labram_available() and _weights_reachable() and "eeg" in task.modalities
            and task.raw_windows is not None and task.extra.get("events") is not None)
