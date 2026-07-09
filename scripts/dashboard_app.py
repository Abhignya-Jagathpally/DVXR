#!/usr/bin/env python
"""scripts/dashboard_app.py — Streamlit real-time replay of the DVXR monitor.

Reads the SAME deterministic replay JSON produced by scripts/build_dashboard.py
(outputs/dashboard/replay_<task>.json) and streams the steps through the panels of
the proposal's loop: fuse -> predict -> explain -> intervene.

Optional dependency: Streamlit is imported *inside* main() so this file stays
importable without it. To run the app:

    venv/bin/pip install streamlit        # optional dependency, not in requirements.txt
    venv/bin/streamlit run scripts/dashboard_app.py

If Streamlit is absent, running the file directly prints the install instruction.
Build the replay JSON first:  venv/bin/python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DASH_DIR = ROOT / "outputs" / "dashboard"

_BAND_ICON = {"low": "🟢", "watch": "🟡", "elevated": "🟠", "high": "🔴"}


def _load_replays() -> Dict[str, dict]:
    replays: Dict[str, dict] = {}
    for path in sorted(DASH_DIR.glob("replay_*.json")):
        rep = json.loads(path.read_text())
        replays[rep["task"]] = rep
    return replays


def _series(steps: List[dict], key: str, upto: int) -> List[float]:
    return [s[key] for s in steps[: upto + 1]]


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install the optional dependency with:\n"
              "    venv/bin/pip install streamlit\n"
              "then run:\n"
              "    venv/bin/streamlit run scripts/dashboard_app.py")
        return

    import pandas as pd

    st.set_page_config(page_title="DVXR Real-Time Monitor", page_icon="🩺", layout="wide")
    st.title("DVXR — Real-Time Multimodal Health Monitor")
    st.caption("Replaying real held-out-subject windows through the trained CACMF fusion "
               "model · fuse → predict → explain → intervene")

    replays = _load_replays()
    if not replays:
        st.error("No replay JSON found. Run:  venv/bin/python scripts/build_dashboard.py")
        return

    with st.sidebar:
        st.header("Controls")
        task = st.selectbox("Task", list(replays.keys()),
                            format_func=lambda t: replays[t]["title"])
        rep = replays[task]
        dropout = st.toggle("Simulate sensor dropout", value=False)
        run_key = "dropout" if dropout else "full"
        steps = rep["runs"][run_key]["steps"]
        speed = st.slider("Speed (steps / second)", 1, 20, 6)
        stream = st.button("▶ Stream", type="primary")
        st.caption(rep["runs"][run_key]["label"])
        st.divider()
        st.caption(f"Held-out subject **{rep['test_subject']}** · "
                   f"{len(steps)} steps · attribution: {rep.get('attribution_source', 'n/a')}")

    cls = rep["kind"] == "classification"
    n = len(steps)
    idx = st.slider("Step", 0, n - 1, 0)

    header = st.empty()
    kpi = st.empty()
    chart_box = st.empty()
    lights_box = st.empty()
    attr_box = st.empty()
    rec_box = st.empty()

    def draw(i: int) -> None:
        cur = steps[i]
        with header.container():
            c1, c2 = st.columns([1, 2])
            if cls:
                band = cur["stress_band"]
                c1.metric("Stress probability", f"{cur['stress_prob'] * 100:.0f}%",
                          f"{_BAND_ICON.get(band, '')} {band}")
                truth = "stress" if cur.get("y_true") else "calm"
                c2.markdown(f"**Ground truth:** {truth}  ·  **Model call:** {cur['stress_label']}")
            else:
                g = cur["glucose_now"]
                c1.metric("Glucose now", "— (sensor gap)" if g is None else f"{g:.0f} mg/dL",
                          None if cur["glucose_forecast"] is None
                          else f"30-min forecast {cur['glucose_forecast']:.0f}")
                if cur.get("glucose_target") is not None:
                    c2.markdown(f"**Actual +30 min:** {cur['glucose_target']:.0f} mg/dL  ·  "
                                f"**interval:** ±{rep.get('forecast_interval', 0):.0f} mg/dL")

        with kpi.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Sensors live", f"{len(cur['present_modalities'])}/{len(rep['modalities'])}")
            k2.metric("Step", f"{i + 1}/{n}")
            k3.metric("Active actions", len(cur["interventions"]))

        with chart_box.container():
            if cls:
                df = pd.DataFrame({"stress probability": _series(steps, "stress_prob", i)})
                st.line_chart(df, height=240)
            else:
                df = pd.DataFrame({
                    "glucose now": _series(steps, "glucose_now", i),
                    "30-min forecast": _series(steps, "glucose_forecast", i),
                })
                st.line_chart(df, height=240)

        with lights_box.container():
            present = set(cur["present_modalities"])
            st.write("**Sensor presence**")
            cols = st.columns(len(rep["modalities"]))
            for col, m in zip(cols, rep["modalities"]):
                col.markdown(f"{'🟢' if m in present else '⚪'} {m}"
                             + ("" if m in present else "  _(dropped)_"))

        with attr_box.container():
            if cur["attribution"]:
                st.write("**Modality attribution**")
                adf = (pd.DataFrame({"modality": list(cur["attribution"].keys()),
                                     "attribution": list(cur["attribution"].values())})
                       .set_index("modality"))
                st.bar_chart(adf, height=200)

        with rec_box.container():
            st.write("**Recommendations & narration**")
            if cur["interventions"]:
                for m in cur["interventions"]:
                    st.warning(m)
            else:
                st.success("All signals within range — no intervention.")
            st.info(cur["narration"])

    if stream:
        bar = st.progress(0.0)
        for i in range(idx, n):
            draw(i)
            bar.progress((i + 1) / n)
            time.sleep(1.0 / speed)
    else:
        draw(idx)

    st.divider()
    st.subheader(f"Grounded insight — {rep['title']}")
    st.code(rep.get("grounded_facts", ""), language="text")
    caveat = rep.get("insight", "")
    if "Caveat:" in caveat:
        st.caption("Caveat:" + caveat.split("Caveat:", 1)[1])


if __name__ == "__main__":
    main()
