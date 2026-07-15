"""dvxr.serve.glassbox — trace BOTH pipelines on one subject, stage by stage.

The live runner (`serve/live.py`) executes the *validated single-modality* screener and returns its
per-window trace. This module adds the honest side-by-side: it also runs the **proposed multimodal
fLLM** path on the same subject and surfaces its internals — per-modality VQ tokens + perplexity, the
cross-modal fusion attention, the frozen-LLM soft-prompt embedding + modality attribution, and a
subject-held-out proposed probability — so a UI can show *how the proposed model actually works* next to
the model that actually wins.

Nothing is reimplemented: the internals come from the modules' own exports — `CACMFModel.fuse()` stores
VQ code indices (`_last_codes`) and attention (`attention_weights()`); `VQBiosignalEncoder.quantize` /
`.perplexity` give tokens + usage; `llm.predictor` gives the frozen-LLM embedding + `modality_attribution`;
`bench.gated_fusion`/`serve.screener` give the probabilities. We only read what they already compute.

Honesty (enforced by `tests/test_honesty_audit.py`):
  * The proposed path is shown **as-is**. On full observation it LOSES to the single-modality winner
    (see the scoreboard panel); that is displayed, never hidden or reframed as a win.
  * A user-provided sample is `validated=False` / `source="upload"` — out-of-distribution, never the
    validated cohort AUROC.
  * Every trace carries the not-a-diagnosis disclaimer.

Offline / CPU / deterministic. torch/transformers/weights are optional — without them the trace falls
back to a clearly-flagged synthetic fixture so the unit test and a no-torch UI still run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

DISCLAIMER = ("Research-grade screening, not a diagnosis. The proposed multimodal path is shown exactly "
              "as it performs — on full observation it underperforms the single-modality screener; its "
              "honest edge is graceful degradation under missing sensors (see the scoreboard).")

# WESAD is the default subject because it is genuinely co-registered (one device, many signals).
DEFAULT_TASK = "wesad_stress"
_MAX_WINDOWS = 64          # cap per-subject windows so the live demo stays responsive


@dataclass
class PipelineTrace:
    """Everything a glass-box UI needs to render both pipelines for one subject."""
    task: str
    subject: str
    source: str                       # "cohort" (validated) | "upload" (OOD sample entry)
    validated: bool
    winner: Dict = field(default_factory=dict)     # the model that actually wins (single-modality)
    proposed: Dict = field(default_factory=dict)   # the proposed multimodal fLLM, shown as-is
    scoreboard: Dict = field(default_factory=dict) # committed full-obs verdict + dropout crossover
    disclaimer: str = DISCLAIMER
    note: str = ""                    # non-empty when the synthetic fixture was used

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------- public entry
def trace_pipeline(task_name: str = DEFAULT_TASK, sid=None, sample_events=None,
                   seed: int = 7, max_windows: int = _MAX_WINDOWS,
                   include_llm: bool = True) -> PipelineTrace:
    """Trace both pipelines on one subject. Real path when torch + data are present; a flagged
    synthetic fixture otherwise (so a no-torch UI and the unit test still work).

    ``sample_events`` (canonical events for one uploaded recording) routes through the OOD upload path
    (`validated=False`, `source="upload"`). Otherwise a held-out cohort subject is used (`validated`).
    """
    try:
        return _real_trace(task_name, sid, sample_events, seed, max_windows, include_llm)
    except Exception as exc:   # noqa: BLE001 — any failure degrades to the honest synthetic fixture
        return _synthetic_trace(task_name, note=f"synthetic fixture ({type(exc).__name__}: {exc})")


# ----------------------------------------------------------------- real path
def _real_trace(task_name, sid, sample_events, seed, max_windows, include_llm) -> PipelineTrace:
    from dvxr.serve.live import build_task_from_events, run_screening_live
    from dvxr.serve.screener import Screener, fit_screener

    if sample_events is not None:
        task, sid = build_task_from_events(sample_events, task_name=task_name)
        source, validated = "upload", False
    else:
        from dvxr.bench.tasks import TASK_BUILDERS
        task = TASK_BUILDERS[task_name]()
        if sid is None:
            sid = np.asarray(task.subject_ids)[0]
        source, validated = "cohort", True

    screener = _load_or_fit_screener(task_name, Screener, fit_screener)
    live = run_screening_live(screener, task, sid, validated=validated, source=source)
    heldout = getattr(screener, "heldout", {}) or {}
    winner = {
        "label": live["result"].get("label", task_name),
        "representation": live["embed_meta"].get("representation"),
        "encoder": live["embed_meta"].get("encoder"),
        "probability": live["result"].get("probability"),
        "risk_band": live["result"].get("risk_band"),
        "interval": live["result"].get("interval"),
        "window_probs": live["window_probs"],
        "drivers": live["drivers"],
        "narrative": live["narrative"],
        "heldout_auroc": live["result"].get("heldout_auroc"),
        "heldout_auroc_subject": live["result"].get("heldout_auroc_subject"),
        "ece": heldout.get("ece"),
        "decision_curve": heldout.get("decision_curve"),
        "caveat": live["result"].get("caveat", ""),
        "n_windows": live["embed_meta"].get("n_windows"),
        "stage_timings": live["stage_timings"],
    }

    proposed = _proposed_trace(task, sid, seed, max_windows, include_llm)
    scoreboard = scoreboard_panel(task_name)
    return PipelineTrace(task=task_name, subject=str(sid), source=source, validated=validated,
                         winner=winner, proposed=proposed, scoreboard=scoreboard)


def _load_or_fit_screener(task_name, Screener, fit_screener):
    """Prefer a committed served screener; otherwise fit one (fast for band-power tasks)."""
    from pathlib import Path
    for rel in (f"outputs/product/screeners/{task_name}", f"screeners/{task_name}"):
        if (Path(rel) / "manifest.json").exists():
            try:
                return Screener.load(rel)
            except Exception:  # noqa: BLE001
                pass
    return fit_screener(task_name)


def _proposed_trace(task, sid, seed, max_windows, include_llm) -> Dict:
    """The proposed multimodal fLLM internals on this subject, read from the modules' own exports."""
    subjects = np.asarray(task.subject_ids)
    subj_rows = np.where(subjects == sid)[0]
    if len(subj_rows) > max_windows:
        subj_rows = subj_rows[:max_windows]

    vq, latents = _vq_tokens(task, subj_rows, seed)
    attention, fused_dim = _cross_modal_attention(task, latents, seed)
    llm = _llm_mechanism(task, sid, seed) if include_llm else {"included": False}
    prob, prob_note = _proposed_probability(task, sid, seed, include_llm)

    return {
        "modalities": task.modalities,
        "vq": vq,                         # {modality: {codes, perplexity, n_codes}}
        "attention": attention,           # {modality: alpha} — cross-modal fusion attention (sums ~1)
        "fused_dim": fused_dim,
        "llm": llm,                       # frozen-LLM backend + pooled dim + modality attribution
        "probability": prob,              # subject-held-out proposed probability (may be None)
        "probability_note": prob_note,
        "note": ("Shown as-is. This is the proposed multimodal fLLM; on full observation it "
                 "underperforms the single-modality winner (see scoreboard). Its real advantage is "
                 "graceful degradation under missing sensors."),
    }


def _vq_tokens(task, subj_rows, seed) -> tuple[Dict, Dict]:
    """Per-modality VQ code indices + perplexity for this subject, plus continuous latents to fuse.

    Reuses `VQBiosignalEncoder` exactly as the LLM predictor does (trained per modality, K=64)."""
    import pandas as pd

    from dvxr.encoders.codebook import VQBiosignalEncoder
    vq: Dict[str, Dict] = {}
    latents: Dict[str, np.ndarray] = {}
    for m in task.modalities:
        X = np.asarray(task.features[m], dtype=float)
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        enc = VQBiosignalEncoder(embedding_dim=24, hidden_dim=32, n_layers=1, n_heads=2,
                                 epochs=8, codebook_size=64, seed=seed)
        emb = enc.fit_transform(df, cols)                       # continuous latent per row
        sub_df = df.iloc[subj_rows].reset_index(drop=True)
        idx, _quant = enc.quantize(sub_df)
        codes = idx["code_index"].to_numpy().astype(int).tolist()
        vq[m] = {"codes": codes, "perplexity": round(float(enc.perplexity(sub_df)), 3),
                 "n_codes": int(enc.codebook_size)}
        latents[m] = emb.to_numpy(dtype=np.float32)[subj_rows]
    return vq, latents


def _cross_modal_attention(task, latents, seed) -> tuple[Dict, int]:
    """Cross-modal fusion attention α per modality, from CACMFModel's own export.

    We build the CACMF cross-modal transformer over these subject latents and read the attention it
    exports (`attention_weights()`), averaged over the subject's windows. This is the fusion layer's
    attention *mechanism*; the trained cohort verdict is the scoreboard, not this forward."""
    import torch

    from dvxr.config import CACMFConfig
    from dvxr.fusion.model import build_cacmf_model
    d = int(next(iter(latents.values())).shape[1])
    cfg = CACMFConfig(d=d, d_f=2 * d, codebook_size=64, n_heads=4, n_fusion_layers=2,
                      fusion_strategy="cross_modal", seed=seed)
    model = build_cacmf_model(cfg, list(task.modalities))
    model.eval()
    with torch.no_grad():
        lat = {m: torch.tensor(v, dtype=torch.float32) for m, v in latents.items()}
        fo = model.fuse(lat, use_codebook=True)
        att = model.attention_weights() or {}
        alpha = {m: round(float(att[m].mean().item()), 4) for m in att}
    # normalize for display (attention over present modalities sums to 1)
    tot = sum(alpha.values()) or 1.0
    alpha = {m: round(v / tot, 4) for m, v in alpha.items()}
    return alpha, int(fo.h.shape[1])


def _llm_mechanism(task, sid, seed) -> Dict:
    """Frozen-LLM soft-prompt embedding + real modality attribution (L2 shift when a modality drops)."""
    try:
        from dvxr.bench.tasks import BenchTask
        from dvxr.llm.predictor import llm_window_embeddings, modality_attribution
        subjects = np.asarray(task.subject_ids)
        mask = subjects == sid
        sub = BenchTask(name=task.name, kind=task.kind,
                        features={m: np.asarray(task.features[m])[mask] for m in task.modalities},
                        feature_names=task.feature_names, y=np.asarray(task.y)[mask],
                        subject_ids=subjects[mask], metric=task.metric,
                        baseline_hint=task.baseline_hint, extra={})
        emb = llm_window_embeddings(sub, seed=seed)
        attrib = modality_attribution(sub, seed=seed)
        return {"included": True, "backend": sub.extra.get("_llm_backend", "frozen causal LM"),
                "pooled_dim": int(emb.shape[1]),
                "attribution": {m: round(float(v), 4) for m, v in attrib.items()},
                "role": "weakest predictor; validated role is explanation-only + missing-modality reader"}
    except Exception as exc:  # noqa: BLE001 — LLM is optional; degrade with a note
        return {"included": False, "note": f"LLM path unavailable ({type(exc).__name__})"}


def _proposed_probability(task, sid, seed, include_llm) -> tuple[Optional[float], str]:
    """Subject-held-out proposed probability: the frozen-LLM representation with a leave-this-subject-out
    logistic head (the exact benchmark protocol, one subject). Honest single-subject read; the cohort
    verdict is the scoreboard. Falls back to the do-no-harm gated fusion, then None."""
    subjects = np.asarray(task.subject_ids)
    y = np.asarray(task.y, dtype=int)
    te = subjects == sid
    tr = ~te
    if len(np.unique(y[tr])) < 2:
        return None, "training fold has one class — no proposed probability for this subject"
    if include_llm:
        try:
            from dvxr.llm.predictor import llm_window_embeddings
            from dvxr.serve.screener import _fit_head, _head_proba
            emb = llm_window_embeddings(task, seed=seed)
            sc, clf = _fit_head(emb[tr], y[tr], seed)
            p = float(np.mean(_head_proba(sc, clf, emb[te])))
            return round(p, 4), "frozen-LLM soft-prompt rep, leave-this-subject-out logistic head"
        except Exception:  # noqa: BLE001
            pass
    try:
        from dvxr.bench.gated_fusion import pred_dnh_gated
        p = float(np.mean(pred_dnh_gated(task, np.where(tr)[0], np.where(te)[0], seed=seed)))
        return round(p, 4), "do-no-harm reliability-gated late fusion, leave-this-subject-out"
    except Exception as exc:  # noqa: BLE001
        return None, f"proposed probability unavailable ({type(exc).__name__})"


# ----------------------------------------------------------------- scoreboard panel
def scoreboard_panel(task_name: str) -> Dict:
    """The committed full-observation verdict for this task (proposed vs best baseline) + the
    sensor-dropout crossover if a streaming showdown was recorded. Torch-free (reads committed files)."""
    from dvxr.serve import evidence
    panel: Dict = {"task": task_name, "full_observation": None, "dropout_crossover": None}
    for rel in ("outputs/benchmark_scoreboard.csv", "outputs/_dnh_labram/benchmark_scoreboard.csv"):
        board = evidence._read_scoreboard(rel)
        if board and task_name in board:
            row = board[task_name]
            def _f(k):
                try:
                    return round(float(row[k]), 4)
                except Exception:  # noqa: BLE001
                    return None
            panel["full_observation"] = {
                "best_baseline": row.get("best_baseline"),
                "base_err": _f("base_err"), "proposed_err": _f("prop_err"),
                "rer_pct": _f("RER_pct"), "source_file": rel,
                "verdict": "proposed loses at full observation"
                           if (_f("prop_err") or 0) > (_f("base_err") or 0)
                           else "see scoreboard",
            }
            break
    panel["dropout_crossover"] = _dropout_crossover(task_name)
    return panel


def _dropout_crossover(task_name: str) -> Optional[Dict]:
    """Read a committed streaming-showdown result if present — the honest sensor-dropout record: does the
    gap narrow (graceful degradation), and is there a CI-backed crossover (a win) or not?"""
    import json
    from pathlib import Path
    p = Path(f"outputs/streaming_showdown_{task_name}.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    # partial_observation_showdown writes crossover_k (int|None), crossover_model, and a `curve`
    # (one row per (k, model): rer_pct, win, ...). When no crossover survives we still report the
    # HONEST graceful-degradation summary: does the proposed model's gap to the floor narrow as
    # sensors drop? (a real, measurable property even without an outright win).
    k = data.get("crossover_k", data.get("crossover"))
    degradation = None
    curve = data.get("curve") or []
    fused = [r for r in curve if r.get("model") == "fused" and r.get("rer_pct") is not None]
    if fused:
        at_full = next((r["rer_pct"] for r in fused if r.get("k") == 0), fused[0]["rer_pct"])
        best = max(fused, key=lambda r: r["rer_pct"])   # least-negative RER = smallest gap
        degradation = {"model": "fused", "rer_at_0_dropped": round(float(at_full), 1),
                       "best_rer": round(float(best["rer_pct"]), 1), "best_k": int(best["k"]),
                       "narrows": best["rer_pct"] > at_full}
    return {"source_file": str(p),
            "crossover": k,
            "model": data.get("crossover_model"),
            "degradation": degradation,
            "note": "smallest #dropped-modalities where the proposed model beats the floor with a "
                    "bootstrap CI excluding a tie; null = no CI-backed crossover survived"}


# ----------------------------------------------------------------- synthetic fixture
def _synthetic_trace(task_name: str, note: str) -> PipelineTrace:
    """Deterministic, clearly-flagged fixture so a no-torch UI and the unit test still run.

    Numbers are illustrative (NOT a validated cohort result); the scoreboard panel still reads the real
    committed board if present."""
    rng = np.random.default_rng(7)
    mods = ["ecg", "eda", "emg", "resp", "temp"]
    alpha_raw = {m: float(rng.uniform(0.5, 1.5)) for m in mods}
    tot = sum(alpha_raw.values())
    attention = {m: round(v / tot, 4) for m, v in alpha_raw.items()}
    vq = {m: {"codes": rng.integers(0, 64, size=8).tolist(),
              "perplexity": round(float(rng.uniform(8, 40)), 3), "n_codes": 64} for m in mods}
    # a real (torch-free) decision curve from synthetic labels so the renderer's DCA panel is exercised
    dca = None
    try:
        from dvxr.serve.utility import decision_curve
        yy = (rng.uniform(0, 1, size=60) < 0.4).astype(int)
        pp = np.clip(yy * 0.6 + rng.uniform(0, 0.4, size=60), 0, 1)
        dca = decision_curve(yy.tolist(), pp.tolist())
    except Exception:  # noqa: BLE001
        dca = None
    winner = {"label": "Acute-stress screen (illustrative)", "representation": "bandpower_concat",
              "encoder": "band-power physiology features", "probability": 0.82,
              "risk_band": "elevated", "interval": [0.71, 0.93],
              "window_probs": [round(float(x), 3) for x in rng.uniform(0.6, 0.95, size=10)],
              "drivers": [], "narrative": {"clinician": DISCLAIMER, "personal": "", "facts": ""},
              "heldout_auroc": 0.955, "heldout_auroc_subject": None, "ece": 0.06,
              "decision_curve": dca, "caveat": DISCLAIMER, "n_windows": 10, "stage_timings": {}}
    proposed = {"modalities": mods, "vq": vq, "attention": attention, "fused_dim": 48,
                "llm": {"included": True, "backend": "frozen causal LM (illustrative)", "pooled_dim": 896,
                        "attribution": attention,
                        "role": "weakest predictor; validated role is explanation-only"},
                "probability": 0.61, "probability_note": "illustrative (synthetic fixture)",
                "note": ("Shown as-is. Proposed multimodal fLLM underperforms the single-modality "
                         "winner at full observation (see scoreboard).")}
    return PipelineTrace(task=task_name, subject="synthetic", source="synthetic", validated=False,
                         winner=winner, proposed=proposed, scoreboard=scoreboard_panel(task_name),
                         note=note)
