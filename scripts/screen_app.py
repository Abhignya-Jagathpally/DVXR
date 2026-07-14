#!/usr/bin/env python3
"""DVXR Screen — LIVE interactive screening app.

Pick a real held-out subject (or drop an EEG/wearable file), hit Run, and watch the pipeline
execute in real time: raw signal → LaBraM embedding → calibration → risk → explanation. Unlike the
static evidence page, every score here is COMPUTED LIVE by the validated Screener.

    venv/bin/pip install -e ".[app]"          # installs streamlit
    venv/bin/streamlit run scripts/screen_app.py

Streamlit is imported inside main() so this file stays importable (and testable) without it. The
compute lives in dvxr.serve.live; this file is only the UI. Offline / CPU / deterministic.

Honesty: held-out cohort subjects carry the validated benchmark AUROC; uploads are flagged
out-of-distribution (illustrative — the validated number applies to the research cohort, not an
arbitrary recording). Research-grade screening, never a diagnosis.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCREENER_DIR = ROOT / "outputs" / "product" / "screeners"
TASKS = [
    ("mumtaz_depression", "🧠  Depression — resting EEG (headline)"),
    ("eegmat_workload", "🧩  Cognitive workload — EEG"),
    ("wesad_stress", "⌚  Acute stress — wearable physiology"),
]
BAND_ICON = {"low": "🟢", "watch": "🟡", "elevated": "🟠", "high": "🔴"}
BAND_HEX = {"low": "#0ca30c", "watch": "#fab219", "elevated": "#ec835a", "high": "#d03b3b"}


def _meter_html(prob: float, band: str) -> str:
    """The 25/50/75 gradient risk meter from build_dashboard.py, positioned by prob."""
    left = max(0.0, min(1.0, prob)) * 100
    hexc = BAND_HEX.get(band, "#888")
    return f"""
    <div style="margin:6px 0 2px">
      <div style="height:14px;border-radius:999px;position:relative;opacity:.92;
        background:linear-gradient(90deg,#0ca30c 0%,#0ca30c 25%,#fab219 25%,#fab219 50%,
          #ec835a 50%,#ec835a 75%,#d03b3b 75%)">
        <div style="position:absolute;top:50%;left:{left:.1f}%;width:18px;height:18px;
          border-radius:50%;background:#fff;border:3px solid {hexc};
          transform:translate(-50%,-50%)"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#8a94a6;
        margin-top:3px"><span>0 low</span><span>0.25</span><span>0.5</span><span>0.75</span>
        <span>high 1</span></div>
    </div>"""


def _load_screener(task: str):
    from dvxr.serve.screener import Screener, fit_screener
    d = SCREENER_DIR / task
    if (d / "manifest.json").exists():
        return Screener.load(d)
    return fit_screener(task)


def _build_task(task_name: str, representation: str):
    """Build the cohort task (subject_ids + features/raw/events) WITHOUT embedding the whole cohort.

    The dropdown only needs subject_ids (+ y for case/control tags); the LaBraM compute is deferred
    to the per-subject live Run — that's the point of the demo. We only warm the encoder weights here
    so the first Run is snappy. Cached by the caller (one-time cohort load)."""
    import numpy as np
    from dvxr.bench.tasks import TASK_BUILDERS

    task = TASK_BUILDERS[task_name]()
    task.name = task_name
    task.extra["_representation"] = representation
    if representation == "labram_eeg":
        from dvxr.serve.live import get_encoder
        get_encoder()                              # load weights once (not the cohort embedding)
    return task, np.asarray(task.y), np.asarray(task.subject_ids)


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install the app extra:\n"
              "    venv/bin/pip install -e \".[app]\"\n"
              "then run:\n"
              "    venv/bin/streamlit run scripts/screen_app.py")
        return

    import numpy as np
    import pandas as pd

    from dvxr.serve.evidence import comparative_table
    from dvxr.serve.live import build_task_from_events, ingest_upload, run_screening_live

    st.set_page_config(page_title="DVXR Screen — Live", page_icon="🩺", layout="wide")

    build_task_cached = st.cache_resource(show_spinner=False)(_build_task)
    load_screener_cached = st.cache_resource(show_spinner=False)(_load_screener)

    st.title("🩺 DVXR Screen — live pipeline")
    st.caption("Real held-out subjects (or your own upload) scored **live** by the validated "
               "models: raw signal → LaBraM embedding → calibration → risk → explanation. "
               "Research-grade screening, **not a diagnosis**.")

    tab_run, tab_evidence = st.tabs(["▶ Live screening", "📊 Evidence"])

    with st.sidebar:
        st.header("Controls")
        task_name = st.selectbox("Screening task", [t for t, _ in TASKS],
                                 format_func=lambda t: dict(TASKS)[t])
        mode = st.radio("Subject source", ["Held-out cohort subject", "Upload a recording"])
        screener = load_screener_cached(task_name)

        subject = None
        upload = None
        if mode == "Held-out cohort subject":
            with st.spinner("Loading cohort (one-time)…"):
                task, y, subjects = build_task_cached(task_name, screener.representation)
            uniq = list(dict.fromkeys(subjects.tolist()))
            def _fmt(s):
                lab = int(round(float(np.mean(y[subjects == s]))))
                tag = {1: "case", 0: "control"}.get(lab, "?")
                return f"{s}  ({tag})"
            subject = st.selectbox("Held-out subject", uniq, index=len(uniq) - 1, format_func=_fmt)
        else:
            upload = st.file_uploader("EEG / wearable export", type=["edf", "bdf", "csv"])
            st.caption("Uploads are **illustrative / out-of-distribution** — the validated AUROC "
                       "applies to the research cohort, not an arbitrary recording.")
        run = st.button("▶ Run screening", type="primary")

    with tab_run:
        if not run:
            st.info("Pick a subject (or upload a file) in the sidebar, then **Run screening**.")
        else:
            stage_box = st.status("Running the pipeline live…", expanded=True)
            order = {"ingest": 0, "raw": 1, "embed": 2, "calibrate": 3, "score": 4,
                     "explain": 5, "done": 6}
            seen = {}
            def on_stage(key, msg):
                seen[key] = msg
                label = {"ingest": "Ingest", "raw": "Read raw signal", "embed": "Embed",
                         "calibrate": "Calibrate", "score": "Score", "explain": "Explain",
                         "done": "Done"}.get(key, key)
                stage_box.write(f"`{order.get(key,'?')}` **{label}** — {msg}")

            try:
                if mode == "Upload a recording":
                    if upload is None:
                        stage_box.update(label="No file uploaded", state="error")
                        st.stop()
                    tmp = ROOT / "outputs" / "product" / f"_upload{Path(upload.name).suffix}"
                    tmp.write_bytes(upload.getbuffer())
                    on_stage("ingest", f"ingesting {upload.name}…")
                    events = ingest_upload(str(tmp))
                    utask, usid = build_task_from_events(events, task_name=task_name)
                    out = run_screening_live(screener, utask, usid, on_stage=on_stage,
                                             validated=False, source="upload")
                else:
                    out = run_screening_live(screener, task, subject, on_stage=on_stage,
                                             validated=True, source="cohort")
                stage_box.update(label=f"Done in {out['stage_timings']['total']}s", state="complete")
            except Exception as e:  # noqa: BLE001 — surface the real reason to the user
                stage_box.update(label="Pipeline error", state="error")
                st.error(f"Could not run screening: {e}")
                st.stop()

            _render_result(st, pd, out, screener)

    with tab_evidence:
        st.subheader("Validated capabilities — every number traces to the benchmark scoreboard")
        rows = comparative_table()
        df = pd.DataFrame([{"Capability": r["task"], "AUROC": r["auroc"],
                            "95% CI": f"[{r['ci'][0]}, {r['ci'][1]}]",
                            "Winning model": r["winner_method"], "Source": r["source"]}
                           for r in rows])
        st.dataframe(df, hide_index=True)
        st.caption("Research-grade screening, never a diagnosis. Excluded by the honesty gate: "
                   "DEAP affect (chance), the learned CACMF fusion (loses), the LLM as a predictor, "
                   "mortality, the diabetes-leak numbers — see docs/MODEL_CARD.md.")


def _render_result(st, pd, out, screener):
    res = out["result"]
    band = res["risk_band"]
    if not out["validated"]:
        st.warning("**Illustrative result — out of distribution.** This upload was scored to "
                   "demonstrate the pipeline; the validated AUROC applies to the research cohort, "
                   "not an arbitrary recording.")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.metric(res["label"], f"{res['probability']*100:.0f}%",
                  f"{BAND_ICON.get(band,'')} {band.upper()}")
        st.markdown(_meter_html(res["probability"], band), unsafe_allow_html=True)
        st.caption(f"90% conformal interval [{res['interval'][0]:.2f}, {res['interval'][1]:.2f}] · "
                   f"{res['n_windows']} windows · {out['embed_meta'].get('encoder','')}")
        if out["validated"]:
            ci = res.get("heldout_auroc_ci") or []
            ci_s = f" (CI [{ci[0]}, {ci[1]}])" if len(ci) == 2 else ""
            st.success(f"Held-out benchmark AUROC {res['heldout_auroc']}{ci_s} — the same validated "
                       f"number, reproduced by this screener.")
    with c2:
        st.caption("Per-window calibrated probability (the pipeline ran window-by-window)")
        wp = out["window_probs"]
        st.line_chart(pd.DataFrame({"risk": wp}), height=190)

    if out["drivers"]:
        st.caption("Top drivers (standardized feature × head weight)")
        dd = pd.DataFrame(out["drivers"]).set_index("feature")["contribution"]
        st.bar_chart(dd, height=180)

    note = out["narrative"].get("clinician", "")
    if note:
        st.subheader("Grounded explanation")
        st.code(str(note), language="text")
    st.caption(f"⏱ stage timings (s): {out['stage_timings']}")


if __name__ == "__main__":
    main()
