#!/usr/bin/env python3
"""Build the DVXR Screen demo: a self-contained HTML page that runs the *validated* screeners
(dvxr.serve.Screener) on real held-out subjects and shows the product as a user would meet it.

Unlike the older CACMF dashboard (which animates the *losing* fusion), this scores real cohort
subjects through the models that actually win the benchmark — headlined by depression screening from
resting EEG (real LaBraM foundation model, held-out AUROC ~0.96), with cognitive-workload and
wearable-stress as supporting multimodal panels. Each panel shows a calibrated risk gauge, the
conformal interval, the benchmark-reproduced AUROC + CI, the top drivers, and the honest caveat.

    venv/bin/python scripts/build_screen_demo.py [--out outputs/product] [--tasks depression,stress]

Offline / CPU / deterministic. Screeners are cached under <out>/screeners/ so re-builds are fast.
The depression panel needs LaBRaM weights + the Mumtaz cohort; panels whose data/weights are absent
are skipped with an honest note rather than faked.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path("outputs/product")

# The product's headline set — EEG mental-health screening first, supporting multimodal after.
# (order matters: it's the panel order on the page.)
PANELS = [
    ("mumtaz_depression", "Depression screen from resting EEG", "🧠",
     "The headline: a short resting-EEG recording, embedded by the real LaBraM EEG foundation "
     "model, screens major-depressive-disorder vs. healthy."),
    ("eegmat_workload", "Cognitive-workload screen from EEG", "🧩",
     "Rest vs. mental-arithmetic load from EEG (LaBraM). A supporting EEG capability."),
    ("wesad_stress", "Acute-stress screen from wearables", "⌚",
     "Wrist/chest physiology (band-power) screens acute stress — the multimodal, no-EEG path."),
]


def _data_ready(task: str) -> bool:
    from dvxr.serve.screener import REPRESENTATION_BY_TASK
    rep = REPRESENTATION_BY_TASK.get(task, "bandpower_concat")
    if rep == "labram_eeg":
        try:
            from dvxr.bench.labram_bench import _weights_reachable
        except Exception:
            return False
        cohort = "mumtaz_mdd" if task == "mumtaz_depression" else "eegmat"
        return _weights_reachable() and (Path("data/real") / cohort).exists()
    if task in ("wesad_stress",):
        return Path("data/real/WESAD").exists()
    return Path("data/real/noneeg").exists()


def _get_screener(task: str, out_dir: Path, seed: int):
    """Load a cached screener or fit + cache one."""
    from dvxr.serve.screener import Screener, fit_screener
    cache = out_dir / "screeners" / task
    if (cache / "manifest.json").exists():
        return Screener.load(cache)
    s = fit_screener(task, seed=seed)
    s.save(cache)
    return s


def _pick_subjects(subjects: np.ndarray, y: np.ndarray, k: int = 2):
    """Deterministically pick up to k subjects, preferring one positive + one negative."""
    uniq = list(dict.fromkeys(subjects.tolist()))
    pos = [s for s in uniq if int(round(float(y[subjects == s].mean()))) == 1]
    neg = [s for s in uniq if int(round(float(y[subjects == s].mean()))) == 0]
    picked = []
    if pos:
        picked.append(pos[-1])
    if neg:
        picked.append(neg[-1])
    for s in uniq[::-1]:
        if len(picked) >= k:
            break
        if s not in picked:
            picked.append(s)
    return picked[:k]


def build_panel(task: str, title: str, icon: str, blurb: str, out_dir: Path, seed: int) -> dict:
    from dvxr.serve.screener import embed_cohort
    from dvxr.serve.explain import top_feature_attribution

    s = _get_screener(task, out_dir, seed)
    emb, y, subjects, _ = embed_cohort(task, s.representation)
    cards = []
    for sid in _pick_subjects(subjects, y):
        mask = subjects == sid
        res = s.score_subject(emb[mask])
        truth = int(round(float(np.mean(y[mask]))))
        attr = top_feature_attribution(s, emb[mask], k=4)
        cards.append({"subject": str(sid), "truth": truth, "result": res, "drivers": attr})
    return {
        "task": task, "title": title, "icon": icon, "blurb": blurb,
        "encoder": s.meta.get("encoder", s.representation),
        "heldout": s.heldout, "caveat": s.meta.get("caveat", ""),
        "literature": s.meta.get("literature", []),
        "band_thresholds": s.meta.get("band_thresholds", {}),
        "cards": cards,
    }


# ------------------------------------------------------------------ rendering
_BAND_COLOR = {"low": "#1a9850", "watch": "#f4a836", "elevated": "#f46d43", "high": "#d73027"}


def _gauge_svg(prob: float, band: str, lo: float, hi: float) -> str:
    """A 180° semicircular risk gauge, self-contained SVG."""
    import math
    color = _BAND_COLOR.get(band, "#888")
    def pt(frac, r):
        ang = math.pi * (1 - frac)
        return 100 + r * math.cos(ang), 100 - r * math.sin(ang)
    def arc(f0, f1, r, w, col):
        x0, y0 = pt(f0, r); x1, y1 = pt(f1, r)
        large = 1 if (f1 - f0) > 0.5 else 0
        return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 {large} 1 {x1:.1f} {y1:.1f}" '
                f'fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>')
    segs = "".join([arc(a, b, 78, 16, c) for a, b, c in [
        (0.00, 0.25, "#1a9850"), (0.25, 0.50, "#f4a836"),
        (0.50, 0.75, "#f46d43"), (0.75, 1.00, "#d73027")]])
    # interval band + needle
    ib = arc(max(0, lo), min(1, hi), 78, 5, "rgba(20,20,30,.55)")
    nx, ny = pt(prob, 62)
    needle = f'<line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#111" stroke-width="3"/>'
    hub = '<circle cx="100" cy="100" r="5" fill="#111"/>'
    label = (f'<text x="100" y="86" text-anchor="middle" font-size="30" font-weight="700" '
             f'fill="{color}">{prob:.2f}</text>'
             f'<text x="100" y="104" text-anchor="middle" font-size="12" '
             f'fill="#555" letter-spacing="1">{band.upper()}</text>')
    return (f'<svg viewBox="0 12 200 100" width="220" height="118" role="img" '
            f'aria-label="risk {prob:.2f} {band}">{segs}{ib}{needle}{hub}{label}</svg>')


def _card_html(card: dict, thresholds: dict) -> str:
    r = card["result"]
    g = _gauge_svg(r["probability"], r["risk_band"], r["interval"][0], r["interval"][1])
    truth_txt = {1: "case", 0: "control"}.get(card["truth"], "?")
    drivers = "".join(
        f'<li><span class="dir {d["direction"]}">{"▲" if d["direction"]=="raises" else "▼"}</span>'
        f'<code>{html.escape(str(d["feature"]))}</code>'
        f'<span class="contrib">{d["contribution"]:+.2f}</span></li>'
        for d in card["drivers"])
    return f"""
    <div class="card">
      <div class="card-head">
        <span class="subj">subject {html.escape(card['subject'])}</span>
        <span class="truth truth-{card['truth']}">cohort label: {truth_txt}</span>
      </div>
      <div class="gauge">{g}</div>
      <div class="ivl">90% conformal interval [{r['interval'][0]:.2f}, {r['interval'][1]:.2f}]
        · {r['n_windows']} windows</div>
      <ul class="drivers">{drivers}</ul>
    </div>"""


def _panel_html(panel: dict) -> str:
    h = panel["heldout"]
    ci = h.get("auroc_ci", [None, None])
    cards = "".join(_card_html(c, panel["band_thresholds"]) for c in panel["cards"])
    lit = "".join(f"<li>{html.escape(x)}</li>" for x in panel["literature"])
    return f"""
  <section class="panel">
    <div class="panel-head">
      <h2><span class="icon">{panel['icon']}</span> {html.escape(panel['title'])}</h2>
      <div class="metric">
        <span class="auroc">AUROC {h.get('auroc')}</span>
        <span class="ci">CI [{ci[0]}, {ci[1]}]</span>
        <span class="proto">{html.escape(str(h.get('protocol','')))}</span>
      </div>
    </div>
    <p class="blurb">{html.escape(panel['blurb'])}</p>
    <div class="cards">{cards}</div>
    <div class="evidence">
      <div><strong>Model:</strong> {html.escape(panel['encoder'])}</div>
      <div><strong>Cohort:</strong> {h.get('n_subjects')} subjects · {h.get('n_windows')} windows ·
           ECE {h.get('ece')}</div>
      <details><summary>Literature &amp; caveat</summary>
        <ul class="lit">{lit}</ul>
        <p class="caveat">{html.escape(panel['caveat'])}</p>
      </details>
    </div>
  </section>"""


def render_html(panels: list, skipped: list) -> str:
    body = "".join(_panel_html(p) for p in panels)
    skip = ""
    if skipped:
        items = "".join(f"<li>{html.escape(t)} — {html.escape(reason)}</li>"
                        for t, reason in skipped)
        skip = (f'<section class="panel skipped"><h2>Not built this run</h2>'
                f'<ul>{items}</ul><p class="blurb">Panels are skipped, never faked, when their '
                f'data or model weights are unavailable.</p></section>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVXR Screen — evidence demo</title>
<style>
 :root {{ --bg:#0f1220; --card:#1a1f36; --ink:#e8eaf3; --muted:#9aa3c0; --line:#2a3152; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg);
        color:var(--ink); }}
 header.top {{ padding:28px 24px 18px; border-bottom:1px solid var(--line);
        background:linear-gradient(180deg,#151a30,#0f1220); }}
 header.top h1 {{ margin:0 0 4px; font-size:26px; letter-spacing:.3px; }}
 header.top .sub {{ color:var(--muted); font-size:14px; max-width:70ch; }}
 header.top .pill {{ display:inline-block; margin-top:10px; padding:3px 10px; border-radius:20px;
        background:#20264a; color:#a9b6ff; font-size:12px; border:1px solid var(--line); }}
 main {{ max-width:1000px; margin:0 auto; padding:20px 16px 60px; }}
 .panel {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
        padding:18px 20px; margin:18px 0; }}
 .panel-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:16px;
        flex-wrap:wrap; }}
 .panel h2 {{ margin:0; font-size:19px; }} .icon {{ margin-right:6px; }}
 .metric {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
 .auroc {{ font-weight:700; color:#7ee0a1; font-size:16px; }}
 .ci {{ color:var(--muted); font-size:13px; }} .proto {{ color:var(--muted); font-size:12px; }}
 .blurb {{ color:var(--muted); margin:8px 0 14px; }}
 .cards {{ display:flex; gap:16px; flex-wrap:wrap; }}
 .card {{ flex:1 1 220px; background:#141830; border:1px solid var(--line); border-radius:12px;
        padding:12px 14px; min-width:220px; }}
 .card-head {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); }}
 .truth {{ padding:1px 7px; border-radius:10px; }}
 .truth-1 {{ background:#3a1d24; color:#ff9db0; }} .truth-0 {{ background:#173226; color:#8fe6b6; }}
 .gauge {{ text-align:center; margin:4px 0; }}
 .ivl {{ text-align:center; font-size:12px; color:var(--muted); margin-bottom:8px; }}
 ul.drivers {{ list-style:none; padding:0; margin:6px 0 0; font-size:12.5px; }}
 ul.drivers li {{ display:flex; align-items:center; gap:8px; padding:2px 0;
        border-top:1px dashed var(--line); }}
 .dir.raises {{ color:#ff7b7b; }} .dir.lowers {{ color:#68d391; }}
 ul.drivers code {{ color:#cdd6ff; }} .contrib {{ margin-left:auto; color:var(--muted); }}
 .evidence {{ margin-top:14px; font-size:13px; color:var(--muted); border-top:1px solid var(--line);
        padding-top:12px; }}
 .evidence strong {{ color:var(--ink); }}
 details {{ margin-top:8px; }} summary {{ cursor:pointer; color:#a9b6ff; }}
 ul.lit {{ font-size:12.5px; }} .caveat {{ font-style:italic; color:#c7a86b; }}
 .skipped {{ opacity:.75; }}
 footer {{ text-align:center; color:var(--muted); font-size:12px; padding:24px; }}
</style></head>
<body>
<header class="top">
  <h1>🩺 DVXR Screen</h1>
  <div class="sub">Research-grade multimodal clinical-risk <strong>screening</strong> — every score
   below is produced live by the model that <em>wins</em> the benchmark for that task, on a real
   held-out subject, with the same subject-held-out AUROC the benchmark reports. Not a diagnosis.</div>
  <span class="pill">offline · CPU · deterministic · numbers trace to the committed scoreboard</span>
</header>
<main>
{body}
{skip}
</main>
<footer>DVXR Lab · research prototype · screening / decision-support only, never a diagnostic claim.
</footer>
</body></html>"""


def build(out_dir: Path, tasks: list, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {t for t in tasks} if tasks else None
    panels, skipped, manifest = [], [], []
    for task, title, icon, blurb in PANELS:
        short = task.split("_")[-1] if "_" in task else task
        if want and task not in want and short not in want:
            continue
        if not _data_ready(task):
            skipped.append((title, "data or model weights unavailable"))
            continue
        print(f"[demo] building panel: {task} …", file=sys.stderr)
        p = build_panel(task, title, icon, blurb, out_dir, seed)
        panels.append(p)
        manifest.append({"task": task, "heldout": p["heldout"],
                         "subjects": [c["subject"] for c in p["cards"]]})
    if not panels:
        print("[demo] no panels built (no cohorts/weights available).", file=sys.stderr)
    html_str = render_html(panels, skipped)
    out_html = out_dir / "index.html"
    out_html.write_text(html_str)
    (out_dir / "demo_manifest.json").write_text(json.dumps(
        {"format": "dvxr-screen-demo/1", "panels": manifest, "skipped": skipped}, indent=2))
    print(f"[demo] wrote {out_html} ({len(panels)} panels, {len(skipped)} skipped)", file=sys.stderr)
    return out_html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--tasks", default="", help="comma-separated subset (e.g. depression,stress)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    build(Path(args.out), tasks, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
