"""dvxr.serve.batch — score a whole cohort and rank subjects for triage.

The single-subject path (`score_subject`) exists; this loops it over an entire cohort and returns a
risk-ranked table — the "who needs a clinician's attention first" view a screening tool needs. Reuses
`embed_cohort` (the batch embedding primitive) + `Screener.score_subject` + calibrated risk bands.
Emits a sorted CSV and a self-contained HTML triage board. Research-grade screening, not diagnosis.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import numpy as np

_BAND_HEX = {"low": "#1a9850", "watch": "#e0952b", "elevated": "#e2673a", "high": "#cf3b2f"}
_BAND_ORDER = {"high": 0, "elevated": 1, "watch": 2, "low": 3}


def triage_cohort(screener, task_name: Optional[str] = None, seed: int = 7):
    """Score every subject in a cohort and return a risk-ranked pandas DataFrame.

    Columns: subject, probability, risk_band, interval_low, interval_high, n_windows,
    cohort_label (ground truth, for evaluation). Sorted by probability descending (highest risk
    first). `task_name` defaults to the screener's task.
    """
    import pandas as pd
    from dvxr.serve.screener import embed_cohort

    task_name = task_name or screener.task
    emb, y, subjects, _ = embed_cohort(task_name, screener.representation)
    rows = []
    for sid in dict.fromkeys(subjects.tolist()):
        mask = subjects == sid
        res = screener.score_subject(emb[mask])
        rows.append({
            "subject": str(sid),
            "probability": res["probability"],
            "risk_band": res["risk_band"],
            "interval_low": res["interval"][0],
            "interval_high": res["interval"][1],
            "n_windows": res["n_windows"],
            "cohort_label": int(round(float(np.mean(y[mask])))),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _row_html(r) -> str:
    band = r["risk_band"]
    hexc = _BAND_HEX.get(band, "#888")
    pct = max(1.0, min(100.0, float(r["probability"]) * 100))
    truth = {1: "case", 0: "control"}.get(int(r["cohort_label"]), "?")
    return f"""
      <tr>
        <td class="num">{int(r['rank'])}</td>
        <td class="mono">{html.escape(str(r['subject']))}</td>
        <td class="num">{float(r['probability']):.3f}</td>
        <td><span class="band" style="background:{hexc}">{band.upper()}</span></td>
        <td><div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{hexc}"></div></div></td>
        <td class="num muted">[{float(r['interval_low']):.2f}, {float(r['interval_high']):.2f}]</td>
        <td class="num muted">{int(r['n_windows'])}</td>
        <td class="muted">{truth}</td>
      </tr>"""


def render_triage_html(df, screener, task_name: Optional[str] = None) -> str:
    """Self-contained (no external resources) HTML triage board — highest risk first."""
    task_name = task_name or screener.task
    h = screener.heldout
    label = screener.meta.get("label", task_name)
    subj = (f" · subject-level {h.get('auroc_subject')}"
            if h.get("auroc_subject") is not None else "")
    counts = df["risk_band"].value_counts().to_dict()
    chips = "".join(
        f'<span class="chip" style="border-color:{_BAND_HEX[b]};color:{_BAND_HEX[b]}">'
        f'{counts.get(b, 0)} {b}</span>' for b in ["high", "elevated", "watch", "low"])
    rows = "".join(_row_html(r) for _, r in df.iterrows())
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVXR triage — {html.escape(label)}</title>
<style>
 :root {{ --bg:#0f1220; --panel:#1a1f36; --ink:#e8eaf3; --muted:#98a2c0; --line:#2a3152; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); padding:24px;
   font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); font-size:13px; max-width:75ch; }}
 .chips {{ margin:12px 0 4px; display:flex; gap:8px; flex-wrap:wrap; }}
 .chip {{ font-size:12px; border:1px solid; border-radius:20px; padding:2px 10px; }}
 .wrap {{ overflow-x:auto; margin-top:12px; }}
 table {{ border-collapse:collapse; width:100%; min-width:640px; }}
 th {{ text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase;
   letter-spacing:.05em; border-bottom:1px solid var(--line); padding:8px; }}
 td {{ padding:8px; border-bottom:1px solid var(--line); }}
 td.num, .mono {{ font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
 .muted {{ color:var(--muted); }}
 .band {{ color:#0b0e18; font-weight:700; font-size:11px; border-radius:6px; padding:2px 8px; }}
 .bar {{ background:#12172c; border-radius:5px; height:12px; width:120px; }}
 .fill {{ height:100%; border-radius:5px; }}
 footer {{ color:var(--muted); font-size:12px; margin-top:18px; }}
</style></head><body>
 <h1>🩺 DVXR triage — {html.escape(label)}</h1>
 <div class="sub">{len(df)} subjects scored and ranked by calibrated risk (highest first). Held-out
   AUROC {h.get('auroc')}{subj}. Research-grade screening for prioritization — <strong>not a
   diagnosis</strong>; a high band is a prompt to review, never a conclusion.</div>
 <div class="chips">{chips}</div>
 <div class="wrap"><table>
   <thead><tr><th>#</th><th>subject</th><th>risk</th><th>band</th><th>level</th>
     <th>90% interval</th><th>windows</th><th>cohort label</th></tr></thead>
   <tbody>{rows}</tbody>
 </table></div>
 <footer>DVXR Lab · research prototype · screening / decision-support only.</footer>
</body></html>"""


def write_triage(screener, out_dir: str | Path, task_name: Optional[str] = None):
    """Score + rank a cohort and write triage.csv + triage.html. Returns the DataFrame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = triage_cohort(screener, task_name)
    df.to_csv(out_dir / "triage.csv", index=False)
    (out_dir / "triage.html").write_text(render_triage_html(df, screener, task_name))
    return df
