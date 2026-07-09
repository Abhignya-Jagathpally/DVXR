#!/usr/bin/env python
"""scripts/build_dashboard.py — build the real-time streaming replay dashboard.

Replays REAL held-out-subject dataset windows in time order through the TRAINED
CACMF model (LLM-inspired multimodal fusion: fuse -> predict -> explain -> intervene)
to simulate real-time clinical monitoring, per the DVXR proposal's "real-time stress
and glucose monitoring + adaptive intervention recommendations".

For each task (wesad_stress, cgmacros_glucose) it:
  1. Trains CACMF on a held-out-subject TRAIN split (with modality dropout so the model
     degrades gracefully when a sensor drops out).
  2. Orders the held-out test subject's windows chronologically and runs the TRAINED
     model forward per window -> the live stress/glucose trace (REAL model numbers).
  3. Produces TWO runs: (a) full modalities, (b) a mid-stream SENSOR DROPOUT (one
     modality dropped partway) to show graceful degradation.
  4. Derives per-modality attribution by occlusion on the trained model (or the LLM
     attribution when a local LLM is available), fires transparent interventions via
     the reused rule engine, and grounds narration via the reused insight module.
  5. Writes outputs/dashboard/replay_<task>.json (deterministic) + a self-contained
     index.html that animates the replay offline.

Reused (not reinvented): dvxr.bench.representations._train_fused (trained CACMF +
train-fit standardisation), dvxr.calibration.risk_band, dvxr.realtime.intervention.
evaluate_interventions, dvxr.llm.insight.{build_grounded_facts,personal_insight},
dvxr.realtime.monitor.stream_fused_predictions (replay-engine sanity check).

Run:  venv/bin/python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# make `src/` importable when run directly from the repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvxr.calibration import risk_band
from dvxr.llm.insight import build_grounded_facts, personal_insight
from dvxr.realtime.intervention import evaluate_interventions

OUT_DIR = ROOT / "outputs" / "dashboard"
SEED = 7
STRESS_EPOCHS = 20
GLUCOSE_EPOCHS = 12
MODALITY_DROPOUT = 0.3          # train-time robustness so sensor dropout degrades gracefully
MAX_STEPS = 120                 # cap on replay steps per run (keeps the JSON/animation light)

# fine-grained modality -> a short human label for the presence lights
_MOD_LABEL = {
    "ecg": "ECG", "eda": "EDA", "emg": "EMG", "motion": "Motion",
    "ppg": "PPG", "resp": "Resp", "temp": "Temp", "cgm": "CGM",
}


# --------------------------------------------------------------------- helpers
def _pick_test_subject(task) -> Tuple[str, np.ndarray]:
    """Choose a held-out test subject with a rich, informative trace."""
    subjects = np.unique(task.subject_ids)
    best = None
    for s in subjects:
        idx = np.where(task.subject_ids == s)[0]
        if task.kind == "classification":
            y = task.y[idx].astype(int)
            if len(np.unique(y)) < 2:            # need both classes for an interesting trace
                continue
            counts = np.bincount(y, minlength=2)
            score = (int(counts.min()), int(len(idx)))
        else:
            score = (int(len(idx)),)             # regression: pick the longest trace
        if best is None or score > best[0]:
            best = (score, s, idx)
    if best is None:                             # fallback: largest group regardless of balance
        s = max(subjects, key=lambda z: (task.subject_ids == z).sum())
        return s, np.where(task.subject_ids == s)[0]
    return best[1], best[2]


def _order_chronologically(task, te: np.ndarray) -> List[int]:
    """Return test-window global indices sorted by window time."""
    rw = task.raw_windows
    if rw is not None and "window_start" in rw.columns:
        key = rw["window_start"].to_numpy()
    elif rw is not None and "timestamp_utc" in rw.columns:
        key = rw["timestamp_utc"].to_numpy()
    else:
        key = np.arange(task.n)
    return sorted(te.tolist(), key=lambda i: key[i])


def _cap_steps(order: List[int]) -> Tuple[List[int], bool]:
    """Evenly subsample (preserving order) if the trace exceeds MAX_STEPS."""
    if len(order) <= MAX_STEPS:
        return order, False
    sel = np.linspace(0, len(order) - 1, MAX_STEPS).round().astype(int)
    sel = sorted(set(sel.tolist()))
    return [order[i] for i in sel], True


def _timestamp(task, i: int) -> str:
    rw = task.raw_windows
    for col in ("window_start", "timestamp_utc"):
        if rw is not None and col in rw.columns:
            return str(rw[col].iloc[i])
    return str(i)


def _tensors(f_all, order, mods, torch):
    return {m: f_all[m][order].clone() for m in mods}


def _class_probs(model, feats, task_name) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model.probabilities(feats)[task_name][:, 1].numpy()


def _forecast(model, feats, y_mu, y_sd) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(feats)["forecast"].numpy() * y_sd + y_mu


# ------------------------------------------------------------- attribution
def _occlusion_attribution_cls(model, base_feats, mods, task_name) -> np.ndarray:
    """Per-step, per-modality attribution: |Δ probability| when a modality is zeroed
    (its standardised buffer flattened to the training mean). Rows=mods, cols=steps."""
    base = _class_probs(model, base_feats, task_name)
    rows = []
    for m in mods:
        occ = {k: v.clone() for k, v in base_feats.items()}
        occ[m][:] = 0.0
        rows.append(np.abs(base - _class_probs(model, occ, task_name)))
    return np.vstack(rows)


def _normalise_columns(mat: np.ndarray) -> np.ndarray:
    tot = mat.sum(axis=0, keepdims=True)
    tot[tot == 0] = 1.0
    return mat / tot


def _llm_attribution_or_none(task) -> Optional[Dict[str, float]]:
    """Use the local-LLM modality attribution when available, else None (fallback)."""
    try:
        from dvxr.bench.representations import llm_available
        if not llm_available():
            return None
        from dvxr.llm.predictor import modality_attribution
        return modality_attribution(task)
    except Exception:
        return None


# --------------------------------------------------------------- narration
def _step_bundle(kind, prob, band, glu, interventions, top_mod) -> Dict:
    bundle: Dict = {"interventions": interventions}
    if top_mod:
        bundle["top_modality"] = top_mod
    if kind == "classification":
        bundle["tasks"] = {"stress": {"probability": prob, "band": band}}
    if glu is not None:
        bundle["glucose"] = glu
    return bundle


def _narration(kind, prob, band, glu, interventions, top_mod) -> str:
    """A short, fully grounded one-liner (every number comes from the model/replay)."""
    if kind == "classification":
        head = f"Stress {prob*100:.0f}% ({band})"
        if top_mod:
            head += f"; top driver {_MOD_LABEL.get(top_mod, top_mod)}"
    else:
        head = (f"Glucose {glu['now']:.0f} mg/dL" if glu and glu.get("now") is not None
                else "CGM sensor gap")
        if glu and glu.get("forecast") is not None:
            head += (f", 30-min forecast {glu['forecast']:.0f}"
                     f" [{glu['lower']:.0f}-{glu['upper']:.0f}]")
    tail = interventions[0] if interventions else "No action needed."
    return head + ". " + tail


# ------------------------------------------------------------- step builder
def _stress_steps(task, model, f_all, order, mods, dropped=None, drop_from=None,
                  attr_override=None):
    import torch
    feats = _tensors(f_all, order, mods, torch)
    if dropped is not None:
        feats[dropped][drop_from:] = 0.0
    probs = _class_probs(model, feats, task.name)
    attr_mat = _normalise_columns(_occlusion_attribution_cls(model, feats, mods, task.name))
    y_true = task.y[order].astype(int)
    steps = []
    for j, gi in enumerate(order):
        prob = float(probs[j])
        band = risk_band(prob)
        present = [m for m in mods if not (dropped == m and j >= drop_from)]
        if attr_override is not None:                 # LLM attribution (global) if available
            attribution = {m: float(attr_override.get(m, 0.0)) for m in present}
        else:
            attribution = {m: float(attr_mat[k, j]) for k, m in enumerate(mods) if m in present}
        state = {"stress_probability": prob, "stress_band": band}
        interventions = [r.message for r in evaluate_interventions(state)]
        top_mod = max(attribution, key=attribution.get) if attribution else None
        steps.append({
            "idx": j,
            "t": _timestamp(task, gi),
            "stress_prob": round(prob, 4),
            "stress_band": band,
            "stress_label": "stress" if prob >= 0.5 else "non_stress",
            "y_true": int(y_true[j]),
            "glucose_now": None, "glucose_forecast": None,
            "glucose_lower": None, "glucose_upper": None,
            "present_modalities": [_MOD_LABEL.get(m, m) for m in present],
            "attribution": {_MOD_LABEL.get(m, m): round(v, 4) for m, v in attribution.items()},
            "interventions": interventions,
            "narration": _narration("classification", prob, band, None, interventions, top_mod),
        })
    return steps


def _glucose_steps(task, model, f_all, order, mods, y_mu, y_sd, interval,
                   dropped=None, drop_from=None):
    import torch
    feats = _tensors(f_all, order, mods, torch)
    fc = _forecast(model, feats, y_mu, y_sd)
    rw = task.raw_windows
    now_col = rw["glucose_now"].to_numpy(dtype=float)
    slope_col = (rw["glucose_slope"].to_numpy(dtype=float)
                 if "glucose_slope" in rw.columns else np.zeros(task.n))
    steps = []
    last_fc = None
    for j, gi in enumerate(order):
        gap = dropped is not None and j >= drop_from       # CGM sensor gap
        if gap:
            present: List[str] = []
            now = None
            forecast = last_fc                              # hold last known forecast
            trend = 0.0
        else:
            present = list(mods)
            now = float(now_col[gi])
            forecast = float(fc[j])
            last_fc = forecast
            trend = float(slope_col[gi])
        glu = None
        if forecast is not None:
            glu = {"now": now, "forecast": forecast,
                   "lower": forecast - interval, "upper": forecast + interval}
        state = {"glucose_now": now, "glucose_forecast": forecast, "glucose_trend": trend}
        interventions = [r.message for r in evaluate_interventions(state)]
        attribution = {} if gap else {_MOD_LABEL.get("cgm", "CGM"): 1.0}
        steps.append({
            "idx": j,
            "t": _timestamp(task, gi),
            "stress_prob": None, "stress_band": None,
            "stress_label": None, "y_true": None,
            "glucose_now": None if now is None else round(now, 2),
            "glucose_forecast": None if forecast is None else round(forecast, 2),
            "glucose_lower": None if glu is None else round(glu["lower"], 2),
            "glucose_upper": None if glu is None else round(glu["upper"], 2),
            "glucose_target": round(float(task.y[gi]), 2) if not gap else None,
            "present_modalities": [_MOD_LABEL.get(m, m) for m in present],
            "attribution": attribution,
            "interventions": interventions,
            "narration": _narration("forecast", None, None, glu, interventions, None),
        })
    return steps


# --------------------------------------------------------------- replay build
def _monitor_sanity_check(events) -> int:
    """Run the reused replay engine end-to-end (default monitor) as a sanity check."""
    try:
        from dvxr.realtime.monitor import stream_fused_predictions
        df = stream_fused_predictions(events, step_seconds=30, window_seconds=30)
        return int(len(df))
    except Exception as exc:                               # never hard-fail the build
        print(f"  [monitor] stream_fused_predictions skipped: {exc}")
        return -1


def build_stress_replay() -> Dict:
    from dvxr.bench.representations import _train_fused
    from dvxr.bench.tasks import wesad_stress_task
    print("[stress] loading wesad_stress task ...")
    task = wesad_stress_task()
    test_subject, te = _pick_test_subject(task)
    tr = np.array([i for i in range(task.n) if i not in set(te.tolist())])
    print(f"[stress] test subject {test_subject} ({len(te)} windows), "
          f"train {len(tr)} windows, modalities {task.modalities}")

    model, f_all, _ = _train_fused(task, tr, seed=SEED, epochs=STRESS_EPOCHS,
                                   modality_dropout=MODALITY_DROPOUT)
    mods = task.modalities
    order, capped = _cap_steps(_order_chronologically(task, te))
    if capped:
        print(f"[stress] trace capped to {MAX_STEPS} steps")

    # Per-step attribution comes from OCCLUSION on the trained CACMF (real, live, animated).
    # When a local LLM is present we ALSO record its global modality attribution as provenance.
    llm_attr = _llm_attribution_or_none(task)
    print(f"[stress] per-step attribution: occlusion (trained CACMF); "
          f"LLM global attribution: {'recorded' if llm_attr else 'unavailable (fallback)'}")

    full = _stress_steps(task, model, f_all, order, mods, attr_override=None)

    # choose the most influential modality (mean occlusion attribution) to drop mid-stream
    mean_attr = {m: float(np.mean([s["attribution"].get(_MOD_LABEL.get(m, m), 0.0)
                                   for s in full])) for m in mods}
    dropped = max(mean_attr, key=mean_attr.get)
    drop_from = len(order) // 2
    dropout = _stress_steps(task, model, f_all, order, mods,
                            dropped=dropped, drop_from=drop_from, attr_override=None)
    print(f"[stress] dropout run drops {dropped} from step {drop_from}/{len(order)}")

    peak = max(full, key=lambda s: s["stress_prob"])
    bundle = _step_bundle("classification", peak["stress_prob"], peak["stress_band"],
                          None, peak["interventions"],
                          max(peak["attribution"], key=peak["attribution"].get)
                          if peak["attribution"] else None)
    insight = personal_insight(bundle)

    n_mon = _monitor_sanity_check(task.extra["events"])
    return {
        "task": task.name,
        "title": "Stress monitoring (WESAD)",
        "kind": task.kind,
        "test_subject": str(test_subject),
        "modalities": [_MOD_LABEL.get(m, m) for m in mods],
        "attribution_source": "occlusion (trained CACMF)",
        "llm_attribution": ({_MOD_LABEL.get(m, m): round(float(v), 4) for m, v in llm_attr.items()}
                            if llm_attr else None),
        "runs": {
            "full": {"label": "Full modalities", "dropped": None,
                     "drop_from_step": None, "steps": full},
            "dropout": {"label": f"Sensor dropout ({_MOD_LABEL.get(dropped, dropped)} lost mid-stream)",
                        "dropped": _MOD_LABEL.get(dropped, dropped),
                        "drop_from_step": drop_from, "steps": dropout},
        },
        "insight": insight,
        "grounded_facts": build_grounded_facts(bundle),
        "monitor_replay_steps": n_mon,
        "n_steps": len(full),
        "capped": capped,
    }


def build_glucose_replay() -> Dict:
    from dvxr.bench.representations import _train_fused
    from dvxr.bench.tasks import cgmacros_glucose_task
    print("[glucose] loading cgmacros_glucose task ...")
    task = cgmacros_glucose_task()
    test_subject, te = _pick_test_subject(task)
    tr = np.array([i for i in range(task.n) if i not in set(te.tolist())])
    print(f"[glucose] test subject {test_subject} ({len(te)} windows), train {len(tr)} windows")

    model, f_all, (y_mu, y_sd) = _train_fused(task, tr, seed=SEED, epochs=GLUCOSE_EPOCHS)
    mods = task.modalities

    # prediction interval from TRAIN residuals (grounded, not invented)
    import torch
    tr_feats = {m: f_all[m][tr] for m in mods}
    tr_pred = _forecast(model, tr_feats, y_mu, y_sd)
    interval = float(np.std(task.y[tr] - tr_pred)) or 15.0
    print(f"[glucose] forecast interval (train residual std) = {interval:.1f} mg/dL")

    order, capped = _cap_steps(_order_chronologically(task, te))
    if capped:
        print(f"[glucose] trace capped to {MAX_STEPS} steps")

    full = _glucose_steps(task, model, f_all, order, mods, y_mu, y_sd, interval)
    drop_from = len(order) // 2
    dropout = _glucose_steps(task, model, f_all, order, mods, y_mu, y_sd, interval,
                             dropped="cgm", drop_from=drop_from)
    print(f"[glucose] dropout run: CGM sensor gap from step {drop_from}/{len(order)}")

    # peak-risk step (furthest from 120 mg/dL) for the grounded insight
    peak = max(full, key=lambda s: abs((s["glucose_now"] or 120) - 120))
    glu = {"now": peak["glucose_now"], "forecast": peak["glucose_forecast"],
           "lower": peak["glucose_lower"], "upper": peak["glucose_upper"]}
    bundle = _step_bundle("forecast", None, None, glu, peak["interventions"], "CGM")
    insight = personal_insight(bundle)

    return {
        "task": task.name,
        "title": "Glucose monitoring (CGMacros)",
        "kind": task.kind,
        "test_subject": str(test_subject),
        "modalities": [_MOD_LABEL.get(m, m) for m in mods],
        "attribution_source": "single-modality (CGM)",
        "forecast_interval": round(interval, 2),
        "runs": {
            "full": {"label": "Full modalities", "dropped": None,
                     "drop_from_step": None, "steps": full},
            "dropout": {"label": "Sensor dropout (CGM gap mid-stream)",
                        "dropped": "CGM", "drop_from_step": drop_from, "steps": dropout},
        },
        "insight": insight,
        "grounded_facts": build_grounded_facts(bundle),
        "n_steps": len(full),
        "capped": capped,
    }


# --------------------------------------------------------------- HTML render
def render_html(replays: Dict[str, Dict]) -> str:
    """Return a fully self-contained (offline) HTML page with the replays embedded."""
    payload = {
        "tasks": [replays[k]["task"] for k in replays],
        "replays": replays,
    }
    data_json = json.dumps(payload, ensure_ascii=True)
    return _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


_HTML_TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVXR — Real-Time Multimodal Health Monitor</title>
<style>
:root{
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ec;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series1:#2a78d6; --series2:#1baf7a; --series1-soft:rgba(42,120,214,.14);
  --series2-soft:rgba(27,175,122,.16);
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --shadow:0 1px 2px rgba(11,11,11,.05),0 4px 16px rgba(11,11,11,.05);
}
@media (prefers-color-scheme: dark){
  :root{
    --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220;
    --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --series1:#3987e5; --series2:#199e70; --series1-soft:rgba(57,135,229,.18);
    --series2-soft:rgba(25,158,112,.20);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 20px rgba(0,0,0,.35);
  }
}
:root[data-theme="light"]{
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ec;
  --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,.10);
  --series1:#2a78d6; --series2:#1baf7a; --series1-soft:rgba(42,120,214,.14);
  --series2-soft:rgba(27,175,122,.16);
}
:root[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220;
  --ink:#ffffff; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,.10);
  --series1:#3987e5; --series2:#199e70; --series1-soft:rgba(57,135,229,.18);
  --series2-soft:rgba(25,158,112,.20);
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
a{color:var(--series1)}
h1{font-size:23px;font-weight:650;letter-spacing:-.01em;margin:0}
.sub{color:var(--ink2);font-size:14px;margin:6px 0 0}
.tag{display:inline-block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--series1);font-weight:600;margin-bottom:6px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
  flex-wrap:wrap;margin-bottom:20px}
.themebtn{border:1px solid var(--border);background:var(--surface);color:var(--ink2);
  border-radius:9px;padding:7px 12px;font-size:13px;cursor:pointer}
.themebtn:hover{color:var(--ink)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;
  background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:12px 14px;box-shadow:var(--shadow);margin-bottom:18px}
.seg{display:inline-flex;background:var(--surface-2);border-radius:10px;padding:3px;
  border:1px solid var(--border)}
.seg button{border:0;background:transparent;color:var(--ink2);font-size:13px;
  font-weight:550;padding:6px 12px;border-radius:8px;cursor:pointer}
.seg button.on{background:var(--surface);color:var(--ink);box-shadow:var(--shadow)}
.ctl-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
  margin-right:2px}
.play{width:40px;height:40px;border-radius:50%;border:1px solid var(--border);
  background:var(--series1);color:#fff;font-size:15px;cursor:pointer;display:grid;
  place-items:center;flex:none}
.play:hover{filter:brightness(1.06)}
input[type=range]{accent-color:var(--series1);cursor:pointer}
.scrub{flex:1 1 220px;min-width:180px}
.speed{width:110px}
.clock{font-variant-numeric:tabular-nums;font-size:12px;color:var(--ink2);
  white-space:nowrap}
.grid{display:grid;grid-template-columns:320px 1fr;gap:16px;align-items:start}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:16px 18px;box-shadow:var(--shadow)}
.card h2{font-size:12px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--muted);margin:0 0 12px}
.hero{font-size:46px;font-weight:680;letter-spacing:-.02em;line-height:1}
.hero small{font-size:16px;color:var(--ink2);font-weight:500}
.band{display:inline-flex;align-items:center;gap:7px;font-size:13px;font-weight:600;
  padding:4px 11px;border-radius:999px;margin-top:10px}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
.meter{height:12px;border-radius:999px;margin-top:16px;position:relative;
  background:linear-gradient(90deg,var(--good) 0%,var(--good) 25%,var(--warn) 25%,
    var(--warn) 50%,var(--serious) 50%,var(--serious) 75%,var(--critical) 75%);
  opacity:.9}
.meter .knob{position:absolute;top:50%;width:16px;height:16px;border-radius:50%;
  background:var(--surface);border:2px solid var(--ink);transform:translate(-50%,-50%);
  transition:left .18s ease}
.kpis{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.kpi{flex:1 1 90px;background:var(--surface-2);border-radius:10px;padding:9px 11px}
.kpi .v{font-size:18px;font-weight:640;font-variant-numeric:tabular-nums}
.kpi .k{font-size:11px;color:var(--muted);margin-top:1px}
svg{display:block;width:100%;height:auto}
.chartbox{overflow-x:auto}
.lights{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.light{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:550;
  padding:6px 11px;border-radius:999px;border:1px solid var(--border);
  background:var(--surface-2);color:var(--ink2);transition:all .2s}
.light .ld{width:8px;height:8px;border-radius:50%;background:var(--muted);
  box-shadow:none;transition:all .2s}
.light.on{color:var(--ink);border-color:color-mix(in srgb,var(--good) 45%,var(--border))}
.light.on .ld{background:var(--good);box-shadow:0 0 0 3px color-mix(in srgb,var(--good) 22%,transparent)}
.light.off{opacity:.5;text-decoration:line-through}
.abars{display:flex;flex-direction:column;gap:9px;margin-top:2px}
.abar{display:grid;grid-template-columns:64px 1fr 44px;gap:10px;align-items:center;
  font-size:12px}
.abar .lab{color:var(--ink2);font-weight:550}
.abar .track{height:9px;background:var(--surface-2);border-radius:999px;overflow:hidden}
.abar .fill{height:100%;background:var(--series1);border-radius:999px;
  transition:width .25s ease}
.abar .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.ticker{display:flex;flex-direction:column;gap:8px;margin-top:2px;min-height:44px}
.rec{display:flex;gap:9px;align-items:flex-start;font-size:13.5px;padding:9px 12px;
  border-radius:10px;background:var(--surface-2);border-left:3px solid var(--serious)}
.rec.calm{border-left-color:var(--good);color:var(--ink2)}
.rec .ic{flex:none;font-size:14px;line-height:1.35}
.narr{font-size:14px;color:var(--ink2);margin-top:12px;padding-top:12px;
  border-top:1px solid var(--border)}
.narr b{color:var(--ink);font-weight:600}
.flow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11px;
  color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:2px 0 16px}
.flow span{background:var(--surface-2);border:1px solid var(--border);border-radius:999px;
  padding:3px 10px}
.flow .arr{background:none;border:0;padding:0 1px}
.foot{margin-top:22px;font-size:12.5px;color:var(--ink2)}
.foot .facts{white-space:pre-wrap;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;margin-top:8px;font-variant-numeric:tabular-nums}
.caveat{color:var(--muted);font-size:11.5px;margin-top:12px;font-style:italic}
.prov{font-size:11.5px;color:var(--muted);margin-top:6px}
.stack{display:flex;flex-direction:column;gap:16px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-top:8px}
.legend i{width:11px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle;margin-right:5px}
</style>

<div class="wrap">
  <header>
    <div>
      <div class="tag">DVXR · CACMF multimodal fusion</div>
      <h1>Real-Time Multimodal Health Monitor</h1>
      <p class="sub">Replaying real held-out-subject sensor windows through the trained fusion model —
        <b>fuse → predict → explain → intervene</b>.</p>
    </div>
    <button class="themebtn" id="theme">◐ Theme</button>
  </header>

  <div class="flow" aria-hidden="true">
    <span>Sensors</span><span class="arr">→</span><span>Fuse</span><span class="arr">→</span>
    <span>Predict</span><span class="arr">→</span><span>Explain</span><span class="arr">→</span>
    <span>Intervene</span>
  </div>

  <div class="controls">
    <button class="play" id="play" aria-label="Play/pause">▶</button>
    <span class="ctl-label">Task</span>
    <span class="seg" id="taskseg"></span>
    <span class="ctl-label">Scenario</span>
    <span class="seg" id="runseg">
      <button data-run="full" class="on">Full modalities</button>
      <button data-run="dropout">Sensor dropout</button>
    </span>
    <input class="scrub" id="scrub" type="range" min="0" max="1" value="0" step="1">
    <span class="ctl-label">Speed</span>
    <input class="speed" id="speed" type="range" min="1" max="20" value="6" step="1">
    <span class="clock" id="clock">step 0</span>
  </div>

  <div class="grid">
    <div class="stack">
      <div class="card" id="gaugecard">
        <h2 id="gaugetitle">Live risk</h2>
        <div class="hero" id="herov">—</div>
        <div class="band" id="bandpill"><span class="dot"></span><span id="bandtext">—</span></div>
        <div class="meter" id="meter"><div class="knob" id="knob" style="left:0%"></div></div>
        <div class="kpis" id="kpis"></div>
      </div>
      <div class="card">
        <h2>Sensor presence</h2>
        <div class="lights" id="lights"></div>
      </div>
      <div class="card">
        <h2 id="attrtitle">Modality attribution</h2>
        <div class="abars" id="abars"></div>
      </div>
    </div>

    <div class="stack">
      <div class="card">
        <h2 id="tracetitle">Streaming trace</h2>
        <div class="chartbox"><svg id="chart" viewBox="0 0 860 300" preserveAspectRatio="xMidYMid meet"></svg></div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="card">
        <h2>Recommendations &amp; narration</h2>
        <div class="ticker" id="ticker"></div>
        <div class="narr" id="narr">—</div>
      </div>
    </div>
  </div>

  <div class="foot">
    <div id="insighthead" style="font-weight:600;color:var(--ink)">Grounded insight</div>
    <div class="facts" id="facts">—</div>
    <div class="prov" id="prov"></div>
    <div class="caveat" id="caveat"></div>
  </div>
</div>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const BAND_COLOR = {low:'--good', watch:'--warn', elevated:'--serious', high:'--critical'};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const st = { task: DATA.tasks[0], run: 'full', step: 0, playing:false, speed:6, acc:0, last:0 };

/* ---- build task selector ---- */
const taskseg = document.getElementById('taskseg');
DATA.tasks.forEach((t,i)=>{
  const b=document.createElement('button');
  b.textContent = DATA.replays[t].title; b.dataset.task=t;
  if(i===0) b.classList.add('on');
  b.onclick=()=>{ setTask(t); };
  taskseg.appendChild(b);
});
document.querySelectorAll('#runseg button').forEach(b=>{
  b.onclick=()=>{ st.run=b.dataset.run;
    document.querySelectorAll('#runseg button').forEach(x=>x.classList.toggle('on',x===b));
    st.step=0; syncScrub(); render(); };
});

function rep(){ return DATA.replays[st.task]; }
function steps(){ return rep().runs[st.run].steps; }

function setTask(t){
  st.task=t; st.step=0;
  document.querySelectorAll('#taskseg button').forEach(x=>x.classList.toggle('on',x.dataset.task===t));
  syncScrub(); render();
}
function syncScrub(){
  const s=document.getElementById('scrub'); s.max=steps().length-1; s.value=st.step;
}

/* ---------- rendering ---------- */
function num(x,d=0){ return x==null?'—':Number(x).toFixed(d); }

function render(){
  const R=rep(), S=steps(), i=Math.min(st.step,S.length-1), cur=S[i];
  const cls = R.kind==='classification';
  document.getElementById('clock').textContent =
    `step ${i+1} / ${S.length}` + (cur.t? ` · ${String(cur.t).slice(11,19)||String(cur.t).slice(0,16)}`:'');

  /* hero + gauge / big number */
  const gTitle=document.getElementById('gaugetitle');
  const herov=document.getElementById('herov');
  const meter=document.getElementById('meter');
  const knob=document.getElementById('knob');
  const bandpill=document.getElementById('bandpill');
  const bandtext=document.getElementById('bandtext');
  if(cls){
    gTitle.textContent='Live stress risk';
    herov.innerHTML = `${(cur.stress_prob*100).toFixed(0)}<small>% stress</small>`;
    const bc=cssv(BAND_COLOR[cur.stress_band]||'--muted');
    bandpill.style.background=`color-mix(in srgb, ${bc} 16%, transparent)`;
    bandpill.querySelector('.dot').style.background=bc;
    bandpill.style.color=bc;
    bandtext.textContent = cur.stress_band.toUpperCase()+
      (cur.y_true!=null? ` · truth: ${cur.y_true? 'stress':'calm'}`:'');
    meter.style.display=''; knob.style.left=(cur.stress_prob*100)+'%';
  } else {
    gTitle.textContent='Live glucose';
    const g=cur.glucose_now;
    herov.innerHTML = g==null? `—<small> sensor gap</small>`
      : `${g.toFixed(0)}<small> mg/dL</small>`;
    let lab='in range', bc=cssv('--good');
    if(g==null){lab='no signal';bc=cssv('--muted');}
    else if(g>180){lab='hyperglycemia';bc=cssv('--critical');}
    else if(g<70){lab='hypoglycemia';bc=cssv('--critical');}
    else if(g>140){lab='elevated';bc=cssv('--serious');}
    bandpill.style.background=`color-mix(in srgb, ${bc} 16%, transparent)`;
    bandpill.querySelector('.dot').style.background=bc; bandpill.style.color=bc;
    bandtext.textContent=lab.toUpperCase();
    meter.style.display='none';
  }

  /* KPIs */
  const kp=document.getElementById('kpis'); kp.innerHTML='';
  const addKpi=(v,k)=>{const d=document.createElement('div');d.className='kpi';
    d.innerHTML=`<div class="v">${v}</div><div class="k">${k}</div>`;kp.appendChild(d);};
  if(cls){
    addKpi(cur.present_modalities.length+'/'+R.modalities.length,'sensors live');
    addKpi(cur.stress_label==='stress'?'STRESS':'calm','model call');
    addKpi(cur.interventions.length,'actions');
  } else {
    addKpi(cur.glucose_forecast==null?'—':cur.glucose_forecast.toFixed(0),'30-min forecast');
    addKpi(cur.glucose_target==null?'—':cur.glucose_target.toFixed(0),'actual +30 min');
    addKpi(cur.interventions.length,'actions');
  }

  /* sensor lights */
  const lights=document.getElementById('lights'); lights.innerHTML='';
  const present=new Set(cur.present_modalities);
  R.modalities.forEach(m=>{
    const on=present.has(m);
    const el=document.createElement('span');
    el.className='light '+(on?'on':'off');
    el.innerHTML=`<span class="ld"></span>${m}`;
    lights.appendChild(el);
  });

  /* attribution bars */
  const ab=document.getElementById('abars'); ab.innerHTML='';
  const entries=Object.entries(cur.attribution).sort((a,b)=>b[1]-a[1]);
  const amax=Math.max(0.0001,...entries.map(e=>e[1]));
  if(entries.length===0){ ab.innerHTML='<div class="abar"><span class="lab">—</span><div class="track"></div><span class="val">n/a</span></div>'; }
  entries.forEach(([m,v])=>{
    const row=document.createElement('div'); row.className='abar';
    row.innerHTML=`<span class="lab">${m}</span><div class="track"><div class="fill" style="width:${(v/amax*100).toFixed(1)}%"></div></div><span class="val">${(v*100).toFixed(0)}%</span>`;
    ab.appendChild(row);
  });
  document.getElementById('attrtitle').textContent =
    'Modality attribution · '+(R.attribution_source||'');

  /* ticker */
  const tk=document.getElementById('ticker'); tk.innerHTML='';
  if(cur.interventions.length===0){
    const d=document.createElement('div'); d.className='rec calm';
    d.innerHTML='<span class="ic">✓</span><span>All signals within range — no intervention.</span>';
    tk.appendChild(d);
  } else cur.interventions.forEach(m=>{
    const d=document.createElement('div'); d.className='rec';
    d.innerHTML=`<span class="ic">▲</span><span>${m}</span>`; tk.appendChild(d);
  });
  document.getElementById('narr').innerHTML='<b>Now:</b> '+cur.narration;

  drawChart(R,S,i,cls);
  document.getElementById('scrub').value=i;
}

/* ---------- SVG time-series ---------- */
function drawChart(R,S,i,cls){
  const svg=document.getElementById('chart');
  const W=860,H=300,padL=46,padR=16,padT=14,padB=26;
  const iw=W-padL-padR, ih=H-padT-padB, n=S.length;
  const X=k=> padL + (n<=1?0:k/(n-1)*iw);
  let vmin,vmax,fmt, ticks;
  if(cls){ vmin=0; vmax=1; fmt=v=>(v*100).toFixed(0); ticks=[0,.25,.5,.75,1]; }
  else {
    const vals=[];
    S.forEach(s=>{[s.glucose_now,s.glucose_forecast,s.glucose_lower,s.glucose_upper,s.glucose_target]
      .forEach(v=>{if(v!=null)vals.push(v);});});
    vmin=Math.min(60,...vals)-5; vmax=Math.max(200,...vals)+5;
    fmt=v=>v.toFixed(0);
    ticks=[70,120,180].filter(t=>t>=vmin&&t<=vmax);
    if(ticks.length<2) ticks=[Math.round(vmin),Math.round((vmin+vmax)/2),Math.round(vmax)];
  }
  const Y=v=> padT + (1-(v-vmin)/(vmax-vmin))*ih;
  const c1=cssv('--series1'), c2=cssv('--series2'), grid=cssv('--grid'),
        axis=cssv('--axis'), muted=cssv('--muted'), ink=cssv('--ink');
  let out='';
  // grid + y ticks
  ticks.forEach(t=>{ const y=Y(t);
    out+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="${grid}" stroke-width="1"/>`;
    out+=`<text x="${padL-8}" y="${y+3}" text-anchor="end" font-size="10" fill="${muted}">${fmt(t)}</text>`;
  });
  // reference thresholds
  if(cls){ const y=Y(.5);
    out+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="${axis}" stroke-width="1" stroke-dasharray="4 4"/>`;
    out+=`<text x="${W-padR}" y="${y-4}" text-anchor="end" font-size="9.5" fill="${muted}">decision 0.50</text>`;
  }
  const line=(sel,upto,col,w,dash)=>{
    let d='';
    for(let k=0;k<=upto;k++){ const v=sel(S[k]); if(v==null){ d=''; continue; }
      d+=(d===''?'M':'L')+X(k).toFixed(1)+' '+Y(v).toFixed(1)+' '; }
    return d?`<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" ${dash||''} stroke-linejoin="round" stroke-linecap="round"/>`:'';
  };
  if(!cls){
    // forecast band (revealed up to cursor)
    let up='',lo='';
    for(let k=0;k<=i;k++){ const s=S[k]; if(s.glucose_upper==null) continue;
      up+=(up===''?'M':'L')+X(k).toFixed(1)+' '+Y(s.glucose_upper).toFixed(1)+' '; }
    for(let k=i;k>=0;k--){ const s=S[k]; if(s.glucose_lower==null) continue;
      lo+='L'+X(k).toFixed(1)+' '+Y(s.glucose_lower).toFixed(1)+' '; }
    if(up) out+=`<path d="${up+lo}Z" fill="var(--series2-soft)" stroke="none"/>`;
  }
  // faint full trace + bold revealed
  if(cls){
    out+=line(s=>s.stress_prob,n-1,c1,1.2,'stroke-opacity="0.28"');
    out+=line(s=>s.stress_prob,i,c1,2.2);
    // true-label ticks along the top
    for(let k=0;k<=i;k++){ if(S[k].y_true===1){ const x=X(k);
      out+=`<circle cx="${x.toFixed(1)}" cy="${padT+4}" r="2.1" fill="${cssv('--critical')}"/>`; } }
  } else {
    out+=line(s=>s.glucose_now,n-1,c1,1.2,'stroke-opacity="0.28"');
    out+=line(s=>s.glucose_now,i,c1,2.2);
    out+=line(s=>s.glucose_forecast,i,c2,1.8,'stroke-dasharray="5 4"');
  }
  // cursor
  const cx=X(i);
  out+=`<line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${H-padB}" stroke="${axis}" stroke-width="1"/>`;
  const cv = cls? S[i].stress_prob : S[i].glucose_now;
  if(cv!=null) out+=`<circle cx="${cx.toFixed(1)}" cy="${Y(cv).toFixed(1)}" r="4.5" fill="${cls?c1:c1}" stroke="${cssv('--surface')}" stroke-width="2"/>`;
  // baseline
  out+=`<line x1="${padL}" y1="${H-padB}" x2="${W-padR}" y2="${H-padB}" stroke="${axis}" stroke-width="1"/>`;
  svg.innerHTML=out;

  // legend
  const lg=document.getElementById('legend');
  lg.innerHTML = cls
    ? `<span><i style="background:${c1}"></i>stress probability</span><span><i style="background:${cssv('--critical')}"></i>ground-truth stress window</span>`
    : `<span><i style="background:${c1}"></i>glucose now</span><span><i style="background:${c2}"></i>30-min forecast</span><span><i style="background:var(--series2-soft)"></i>forecast interval</span>`;
}

/* footer (per task) */
function renderFooter(){
  const R=rep();
  document.getElementById('facts').textContent=R.grounded_facts||'—';
  document.getElementById('insighthead').textContent='Grounded insight — '+R.title;
  const cap=(R.insight||'').split('Caveat:');
  document.getElementById('caveat').textContent = cap.length>1? 'Caveat:'+cap[1] : '';
  let prov=`Held-out test subject ${R.test_subject} · trained CACMF (held-out-subject split) · `
    +`${R.runs.full.steps.length} steps`;
  if(R.monitor_replay_steps!=null && R.monitor_replay_steps>=0)
    prov+=` · replay engine sanity: ${R.monitor_replay_steps} monitor steps`;
  document.getElementById('prov').textContent=prov;
}

/* ---------- playback loop ---------- */
function tick(ts){
  if(st.playing){
    if(!st.last) st.last=ts;
    st.acc += (ts-st.last); st.last=ts;
    const interval = 1100/st.speed;   // ms per step
    while(st.acc>=interval){
      st.acc-=interval; st.step++;
      if(st.step>=steps().length){ st.step=steps().length-1; setPlay(false); break; }
      render();
    }
  } else st.last=ts;
  requestAnimationFrame(tick);
}
function setPlay(p){ st.playing=p; document.getElementById('play').textContent=p?'⏸':'▶';
  if(p && st.step>=steps().length-1) st.step=0; st.last=0; }

document.getElementById('play').onclick=()=>setPlay(!st.playing);
document.getElementById('scrub').oninput=e=>{ st.step=+e.target.value; setPlay(false); render(); };
document.getElementById('speed').oninput=e=>{ st.speed=+e.target.value; };

/* task switch also refreshes the footer */
const _setTask=setTask; setTask=function(t){ _setTask(t); renderFooter(); };

/* theme toggle */
document.getElementById('theme').onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const mql=matchMedia('(prefers-color-scheme: dark)').matches;
  const next = cur? (cur==='dark'?'light':'dark') : (mql?'light':'dark');
  document.documentElement.setAttribute('data-theme',next); render();
};

syncScrub(); render(); renderFooter(); requestAnimationFrame(tick);
</script>
"""


def build_replays(out_dir: Path = OUT_DIR) -> Dict[str, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    replays: Dict[str, Dict] = {}
    for builder in (build_stress_replay, build_glucose_replay):
        rep = builder()
        path = out_dir / f"replay_{rep['task']}.json"
        path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[write] {path}  ({rep['n_steps']} steps)")
        replays[rep["task"]] = rep

    # self-contained HTML with both replays embedded inline
    html = render_html(replays)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[write] {out_dir / 'index.html'}")
    return replays


def main() -> None:
    replays = build_replays()
    print("\n=== summary ===")
    for name, rep in replays.items():
        run = rep["runs"]["full"]["steps"]
        drop = rep["runs"]["dropout"]
        print(f"{name}: {len(run)} steps, test subject {rep['test_subject']}, "
              f"dropout={drop['dropped']} from step {drop['drop_from_step']}")


if __name__ == "__main__":
    main()
