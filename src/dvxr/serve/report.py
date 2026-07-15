"""dvxr.serve.report — a self-contained per-subject screening report.

Assembles ONE subject's screening result into a shareable, offline HTML report from pieces that
already exist: the live pipeline run (`serve.live.run_screening_live` → calibrated risk, per-window
trace, drivers, grounded note), the screener's held-out metrics (both granularities), and the
evidence layer's literature/caveats. Research-grade screening, never a diagnosis.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import numpy as np

_BAND_HEX = {"low": "#1a9850", "watch": "#e0952b", "elevated": "#e2673a", "high": "#cf3b2f"}


def subject_report(screener, task, sid, encoder=None, validated: bool = True) -> dict:
    """Run the live pipeline for one subject and attach evidence context for the report."""
    from dvxr.serve.live import run_screening_live
    out = run_screening_live(screener, task, sid, encoder=encoder,
                             validated=validated, source="cohort")
    h = screener.heldout
    out["evidence"] = {
        "window_auroc": h.get("auroc"), "window_ci": h.get("auroc_ci"),
        "subject_auroc": h.get("auroc_subject"), "subject_ci": h.get("auroc_subject_ci"),
        "ece": h.get("ece"), "protocol": h.get("protocol"),
        "n_subjects": h.get("n_subjects"), "literature": screener.meta.get("literature", []),
    }
    return out


def _sparkline(probs, w=280, hgt=48) -> str:
    p = np.asarray(probs, dtype=float)
    if len(p) < 2:
        p = np.repeat(p if len(p) else [0.0], 2)
    xs = np.linspace(0, w, len(p))
    ys = hgt - np.clip(p, 0, 1) * (hgt - 6) - 3
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    mid = hgt - 0.5 * (hgt - 6) - 3  # 0.5 reference line
    return (f'<svg viewBox="0 0 {w} {hgt}" width="100%" height="{hgt}" preserveAspectRatio="none" '
            f'aria-label="per-window risk trace">'
            f'<line x1="0" y1="{mid:.1f}" x2="{w}" y2="{mid:.1f}" stroke="#2a3152" '
            f'stroke-dasharray="3 3" stroke-width="1"/>'
            f'<polyline points="{pts}" fill="none" stroke="#3fd0c9" stroke-width="2"/></svg>')


def _meter(prob, band) -> str:
    left = max(0.0, min(1.0, float(prob))) * 100
    hexc = _BAND_HEX.get(band, "#888")
    return (f'<div class="meter"><div class="knob" style="left:{left:.1f}%;border-color:{hexc}">'
            f'</div></div>')


def render_report_html(report: dict, screener) -> str:
    res = report["result"]
    ev = report["evidence"]
    band = res["risk_band"]
    ood = "" if report.get("validated", True) else (
        '<div class="ood">Illustrative — out of distribution. This recording was scored to '
        'demonstrate the pipeline; the validated AUROC applies to the research cohort, not an '
        'arbitrary recording.</div>')
    drivers = "".join(
        f'<li><span class="d {d["direction"]}">{"▲" if d["direction"]=="raises" else "▼"}</span>'
        f'<code>{html.escape(str(d["feature"]))}</code>'
        f'<span class="c">{d["contribution"]:+.3f}</span></li>' for d in report["drivers"])
    subj = (f' · subject-level {ev["subject_auroc"]} (CI {ev["subject_ci"]})'
            if ev.get("subject_auroc") is not None
            else ' · within-subject task → epoch-level unit')
    lit = "".join(f"<li>{html.escape(x)}</li>" for x in ev.get("literature", []))
    note = html.escape(str(report["narrative"].get("clinician", "")))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVXR screening report — subject {html.escape(report['subject'])}</title>
<style>
 :root {{ --bg:#0f1220; --panel:#1a1f36; --ink:#e8eaf3; --muted:#98a2c0; --line:#2a3152;
   --accent:#3fd0c9; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); padding:24px;
   font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 .card {{ max-width:720px; margin:0 auto; }}
 h1 {{ font-size:20px; margin:0 0 2px; }} .sub {{ color:var(--muted); font-size:13px; }}
 .ood {{ background:#3a1d24; color:#ffb3c0; border:1px solid #cf3b2f; border-radius:8px;
   padding:10px 12px; margin:14px 0; font-size:13px; }}
 .risk {{ display:flex; align-items:baseline; gap:12px; margin-top:16px; }}
 .risk .v {{ font:700 44px ui-monospace,Menlo,monospace; color:{_BAND_HEX.get(band,'#888')};
   line-height:1; }}
 .risk .b {{ font-weight:700; color:#0b0e18; background:{_BAND_HEX.get(band,'#888')};
   border-radius:6px; padding:3px 10px; font-size:13px; }}
 .meter {{ height:12px; border-radius:999px; margin:14px 0 4px; position:relative; opacity:.92;
   background:linear-gradient(90deg,#1a9850 0%,#1a9850 25%,#e0952b 25%,#e0952b 50%,
     #e2673a 50%,#e2673a 75%,#cf3b2f 75%); }}
 .meter .knob {{ position:absolute; top:50%; width:16px; height:16px; border-radius:50%;
   background:#fff; border:3px solid; transform:translate(-50%,-50%); }}
 .scale {{ display:flex; justify-content:space-between; color:var(--muted); font-size:10px; }}
 .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:20px; }}
 .box {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
 .box h2 {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
   margin:0 0 8px; }}
 ul {{ list-style:none; padding:0; margin:0; font-size:13px; }}
 ul.dr li {{ display:flex; align-items:center; gap:8px; padding:3px 0; border-top:1px dashed var(--line); }}
 .d.raises {{ color:#ff7b7b; }} .d.lowers {{ color:#68d391; }} code {{ color:#cdd6ff; }}
 .c {{ margin-left:auto; color:var(--muted); font-family:ui-monospace,monospace; }}
 .kv {{ font-size:13px; }} .kv div {{ padding:2px 0; }} .kv b {{ color:var(--accent); }}
 pre {{ white-space:pre-wrap; background:#12172c; border-radius:8px; padding:12px; font-size:12.5px;
   color:var(--ink); overflow-x:auto; }}
 .lit {{ font-size:12.5px; color:var(--muted); }} .lit li {{ margin:4px 0; }}
 footer {{ color:var(--muted); font-size:12px; margin-top:20px; text-align:center; }}
</style></head><body>
 <div class="card">
   <h1>🩺 DVXR screening report</h1>
   <div class="sub">{html.escape(res['label'])} · subject <b>{html.escape(report['subject'])}</b> ·
     {res['n_windows']} windows · {html.escape(str(report['embed_meta'].get('encoder','')))}</div>
   {ood}
   <div class="risk"><span class="v">{res['probability']:.2f}</span>
     <span class="b">{band.upper()}</span>
     <span class="sub">90% interval [{res['interval'][0]:.2f}, {res['interval'][1]:.2f}]</span></div>
   <div class="meter">{_meter(res['probability'], band)}</div>
   <div class="scale"><span>0 low</span><span>0.25</span><span>0.5</span><span>0.75</span><span>high 1</span></div>

   <div class="grid">
     <div class="box"><h2>Per-window risk trace</h2>{_sparkline(report['window_probs'])}
       <div class="sub" style="margin-top:6px">the pipeline scored each window independently</div></div>
     <div class="box"><h2>Top drivers</h2><ul class="dr">{drivers}</ul></div>
   </div>

   <div class="box" style="margin-top:14px"><h2>Model evidence</h2>
     <div class="kv">
       <div>Held-out AUROC: <b>{ev['window_auroc']}</b> (window-level, CI {ev['window_ci']}){subj}</div>
       <div>Calibration ECE: {ev['ece']} · {html.escape(str(ev['protocol']))} ·
         {ev['n_subjects']} subjects</div>
     </div></div>

   <div class="box" style="margin-top:14px"><h2>Grounded explanation</h2><pre>{note}</pre></div>

   <div class="box" style="margin-top:14px"><h2>Literature</h2><ul class="lit">{lit}</ul></div>

   <footer>Research prototype — screening / decision-support only, <strong>not a diagnosis</strong>.
     A raised band is a prompt to consult a qualified clinician, never a conclusion.</footer>
 </div>
</body></html>"""


def write_subject_report(screener, task_name: str, sid=None, out_path: Optional[str] = None,
                         encoder=None) -> Path:
    """Build the cohort task, score `sid` (default: a held-out subject), write a self-contained HTML."""
    from dvxr.bench.tasks import TASK_BUILDERS
    task = TASK_BUILDERS[task_name]()
    task.name = task_name
    task.extra["_representation"] = screener.representation
    subjects = np.asarray(task.subject_ids)
    if sid is None:
        sid = list(dict.fromkeys(subjects.tolist()))[-1]
    elif sid not in subjects:
        raise ValueError(f"subject {sid!r} not in cohort {task_name}")
    report = subject_report(screener, task, sid, encoder=encoder)
    out = Path(out_path) if out_path else (
        Path("outputs/product") / f"report_{task_name}_{sid}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report_html(report, screener))
    return out
