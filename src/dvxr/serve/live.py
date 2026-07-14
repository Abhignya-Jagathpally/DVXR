"""dvxr.serve.live — run the screening pipeline LIVE on one subject, stage by stage.

The static demo (`build_screen_demo.py`) and `dvxr report` only *display* precomputed numbers. This
module actually *runs the pipeline*: it takes one subject's raw signal and executes
raw → embed (real LaBraM) → calibrate → score → explain in real time, emitting a stage callback for
each step so a UI can show it happening, and returning the per-window probability trace as visible
proof it ran window-by-window.

Embedding a single subject's ~15 EEG windows through LaBraM is seconds on CPU (unlike fitting the
whole cohort), so this is genuine live compute. Two entry points feed it:
  * a held-out cohort subject (validated — carries the benchmark AUROC), or
  * an uploaded recording (illustrative / out-of-distribution — NOT the validated number).

Framework-agnostic (no Streamlit import) so the Streamlit app, the `dvxr screen` CLI, and the tests
all share the exact same computation. Offline / CPU / deterministic.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

StageCB = Optional[Callable[[str, str], None]]   # (stage_key, human_message) -> None


def _stage(cb: StageCB, key: str, msg: str) -> None:
    if cb is not None:
        cb(key, msg)


# --------------------------------------------------------------------- encoder
_ENC = None


def get_encoder():
    """Process-singleton real LaBraM encoder (weights load once; slow first call only)."""
    global _ENC
    if _ENC is None:
        from dvxr.encoders.labram_real import LaBraMEncoder
        _ENC = LaBraMEncoder.from_pretrained()
    return _ENC


# ------------------------------------------------------------ single-subject embed
def embed_subject_live(task, sid, encoder=None, on_stage: StageCB = None
                       ) -> Tuple[np.ndarray, Dict]:
    """Embed ONE subject's windows through the task's validated representation, live.

    For LaBraM tasks this re-extracts that subject's raw EEG at 200 Hz and runs only their windows
    through the frozen model (seconds), mirroring `labram_bench.labram_embeddings` exactly but on the
    subject subset. For band-power tasks it slices the subject's precomputed feature rows. Returns
    ``(emb, meta)`` where meta carries channel/window/timing detail for the UI.
    """
    from dvxr.serve.screener import REPRESENTATION_BY_TASK
    representation = task.extra.get("_representation") or REPRESENTATION_BY_TASK.get(
        getattr(task, "name", ""), "bandpower_concat")

    subjects = np.asarray(task.subject_ids)
    mask = subjects == sid
    if not mask.any():
        raise ValueError(f"subject {sid!r} not present in task {getattr(task, 'name', '?')}")

    if representation == "labram_eeg":
        from dvxr.features import build_raw_windows
        _stage(on_stage, "raw", "reading subject raw EEG windows…")
        win_sub = task.raw_windows[mask].reset_index(drop=True)
        events = task.extra["events"]
        ev_sub = events[events["subject_id"] == sid]
        ch_names = sorted(ev_sub[ev_sub["modality"] == "eeg"]["channel"].unique().tolist())
        wsec = int(task.extra.get("window_seconds", 4))
        samples = max(200, wsec * 200)
        t0 = time.perf_counter()
        raw, ch = build_raw_windows(ev_sub, win_sub, modalities=["eeg"], samples=samples)
        c = int(ch["eeg"])
        k = win_sub.shape[0]
        eeg = raw["eeg"].reshape(k, c, samples).astype(np.float32)
        mu = eeg.mean(axis=2, keepdims=True)
        sd = eeg.std(axis=2, keepdims=True)
        eeg = (eeg - mu) / np.clip(sd, 1e-6, None)
        t_raw = time.perf_counter() - t0
        _stage(on_stage, "embed",
               f"embedding {k} windows ({c}ch × {samples//200} patches) through LaBraM…")
        enc = encoder or get_encoder()
        t1 = time.perf_counter()
        emb = np.asarray(enc.embed(eeg, ch_names, patch_size=200), dtype=float)
        t_embed = time.perf_counter() - t1
        meta = {"representation": representation, "n_windows": int(k), "n_channels": c,
                "patches": samples // 200, "channels": ch_names,
                "encoder": "real LaBraM EEG foundation model (frozen)",
                "t_raw": round(t_raw, 3), "t_embed": round(t_embed, 3)}
        return emb, meta

    # band-power path (e.g. wearable stress) — the "embedding" is the band-power feature vector
    from dvxr.bench.representations import _concat
    _stage(on_stage, "raw", "assembling band-power feature windows…")
    t0 = time.perf_counter()
    full = np.asarray(_concat(task), dtype=float)
    emb = full[mask]
    _stage(on_stage, "embed", f"{emb.shape[0]} band-power windows ({emb.shape[1]} features)")
    meta = {"representation": representation, "n_windows": int(emb.shape[0]),
            "n_features": int(emb.shape[1]), "encoder": "band-power physiology features",
            "t_raw": round(time.perf_counter() - t0, 3), "t_embed": 0.0}
    return emb, meta


# ------------------------------------------------------------ orchestration
def run_screening_live(screener, task, sid, encoder=None, on_stage: StageCB = None,
                       validated: bool = True, source: str = "cohort") -> Dict:
    """Run the full live pipeline for one subject and return a structured, UI-ready result.

    Stages (each timed, each announced via ``on_stage``): embed → calibrate/predict per window →
    aggregate score → attribution → grounded explanation. ``window_probs`` is the per-window
    calibrated-probability trace (the visible proof the pipeline ran window-by-window).
    """
    from dvxr.serve.explain import explain, top_feature_attribution

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    emb, emb_meta = embed_subject_live(task, sid, encoder=encoder, on_stage=on_stage)
    timings["embed"] = round(emb_meta.get("t_raw", 0) + emb_meta.get("t_embed", 0), 3)

    _stage(on_stage, "calibrate", "calibrating per-window probabilities…")
    t = time.perf_counter()
    window_probs = np.asarray(screener.predict_windows(emb), dtype=float)
    timings["calibrate"] = round(time.perf_counter() - t, 3)

    _stage(on_stage, "score", "aggregating subject-level risk + conformal interval…")
    t = time.perf_counter()
    result = screener.score_subject(emb)
    timings["score"] = round(time.perf_counter() - t, 3)

    _stage(on_stage, "explain", "attribution + grounded explanation…")
    t = time.perf_counter()
    drivers = top_feature_attribution(screener, emb, k=5)
    try:
        narrative = explain([result])
    except Exception:
        narrative = {"clinician": result.get("caveat", ""), "personal": "", "facts": ""}
    timings["explain"] = round(time.perf_counter() - t, 3)
    timings["total"] = round(time.perf_counter() - t_all, 3)

    _stage(on_stage, "done", f"done in {timings['total']}s")
    return {
        "subject": str(sid),
        "result": result,
        "window_probs": window_probs.tolist(),
        "drivers": drivers,
        "narrative": narrative,
        "embed_meta": emb_meta,
        "stage_timings": timings,
        "validated": bool(validated),
        "source": source,
    }


# ------------------------------------------------------------ upload path
def build_task_from_events(events, task_name: str = "mumtaz_depression",
                           window_seconds: Optional[int] = None):
    """Assemble a minimal single-subject BenchTask from uploaded canonical events.

    Mirrors `_windowed_signal_task` but stamps a placeholder label so unlabeled uploads still yield
    windows (the label is never used at inference). Returns ``(task, sid)``. The screener reads only
    the embeddings, so the (constant) y here is inert.
    """
    from dvxr.bench.tasks import BenchTask, _split_by_modality
    from dvxr.features import build_raw_windows, build_signal_windows
    from dvxr.serve.screener import REPRESENTATION_BY_TASK

    ev = events.copy()
    lname = "upload"
    if (ev["label_name"].astype(str).str.len() == 0).all():
        ev["label_name"] = lname
        ev["label_value"] = "unknown"
    else:
        lname = str(ev["label_name"].iloc[0]) or lname

    wsec = int(window_seconds if window_seconds is not None else 8)
    win = build_signal_windows(ev, window_seconds=wsec, step_seconds=wsec, label_name=lname)
    win = win[win["target"].astype(str).str.len() > 0].reset_index(drop=True)
    if win.empty:
        raise ValueError("no windows could be built from the uploaded recording "
                         "(too short, or no signal channels).")
    groups = _split_by_modality(win)
    feats = {m: win[cols].to_numpy(dtype=float) for m, cols in groups.items()}
    raw, raw_ch = build_raw_windows(ev, win, modalities=list(groups))
    sids = np.asarray(win["subject_id"].to_numpy())
    task = BenchTask(
        name=task_name, kind="classification", features=feats, feature_names=groups,
        y=np.zeros(len(win), dtype=int), subject_ids=sids, metric="1-AUROC",
        baseline_hint="majority", raw_windows=win,
        extra={"events": ev, "window_seconds": wsec, "raw": raw, "raw_channels": raw_ch,
               "_representation": REPRESENTATION_BY_TASK.get(task_name, "bandpower_concat")})
    return task, sids[0]


def ingest_upload(path: str):
    """Turn an uploaded recording into canonical events, dispatching by file extension.

    Reuses existing single-file loaders: ``.edf`` → :func:`load_single_eeg_edf`;
    ``.bdf`` → :func:`load_deap_raw_bdf`; ``.csv`` → canonical CSV if it already validates, else the
    EMOTIV device-export converter. Returns the validated events frame.
    """
    from pathlib import Path

    from dvxr import loaders

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".edf":
        return loaders.load_single_eeg_edf(p, subject_id="upload_sub")
    if ext == ".bdf":
        return loaders.load_deap_raw_bdf(p, max_seconds=120.0)
    if ext == ".csv":
        try:
            return loaders.load_canonical_csv(p)
        except Exception:
            from scripts.convert_emotiv_subject import convert  # device export → canonical
            return convert(str(p), None, subject_id="upload_sub")
    raise ValueError(f"unsupported upload type {ext!r}; expected .edf, .bdf, or .csv")


def screen_file(path: str, task_name: str = "mumtaz_depression", screener_dir: Optional[str] = None,
                on_stage: StageCB = None) -> Dict:
    """Live-screen an uploaded file end to end: ingest → task → run. Result is flagged
    ``validated=False`` (out-of-distribution) — a pipeline demonstration, not the cohort AUROC."""
    from pathlib import Path

    from dvxr.serve.screener import Screener, fit_screener

    _stage(on_stage, "ingest", f"ingesting {Path(path).name}…")
    events = ingest_upload(path)
    task, sid = build_task_from_events(events, task_name=task_name)
    if screener_dir:
        screener = Screener.load(screener_dir)
    else:
        screener = fit_screener(task_name)
    return run_screening_live(screener, task, sid, on_stage=on_stage,
                              validated=False, source="upload")
