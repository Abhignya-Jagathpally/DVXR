"""dvxr.serve.glassbox_render — render a PipelineTrace into a self-contained, offline glass-box page.

Takes one or more traces (from `serve.glassbox.trace_pipeline(...).to_dict()`) and produces a complete
standalone HTML document: the proposed multimodal fLLM and the validated single-modality winner
side-by-side, stage by stage, with the honest full-observation scoreboard, the sensor-dropout crossover,
and the LLM-generated assessment. When several traces are embedded, a selector switches between them
(pre-rendered blocks toggled by a few lines of inline JS — no re-render, no external code).

Hard constraints (mirrors the evidence page, enforced by the honesty audit):
  * CSP-safe: no external resource loads (no <script src>, <link>, @import, url(http…), src=).
  * Theme-aware via prefers-color-scheme + [data-theme] overrides.
  * Every page carries the not-a-diagnosis disclaimer; a user sample is labelled out-of-distribution;
    the proposed path's underperformance is shown, never reframed as a win.
"""
from __future__ import annotations

import html
import json
from typing import Dict, List

from dvxr.serve.glassbox import DISCLAIMER


def _esc(x) -> str:
    return html.escape(str(x))


def _num(x, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return "—"


# ------------------------------------------------------------------ small SVG bits
def _sparkline(vals: List[float], w: int = 220, h: int = 40) -> str:
    vals = [float(v) for v in (vals or [])]
    if not vals:
        return "<span class='muted'>no windows</span>"
    n = len(vals)
    def X(i): return 4 + i * (w - 8) / max(n - 1, 1)
    def Y(v): return h - 4 - v * (h - 8)          # v in [0,1]
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    dots = "".join(f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='1.6' fill='currentColor'/>"
                   for i, v in enumerate(vals))
    return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
            f"aria-label='per-window probabilities'>"
            f"<line x1='4' y1='{Y(0.5):.1f}' x2='{w-4}' y2='{Y(0.5):.1f}' stroke='currentColor' "
            f"stroke-opacity='0.2'/>"
            f"<polyline points='{pts}' fill='none' stroke='var(--accent)' stroke-width='1.6'/>"
            f"{dots}</svg>")


def _bars(weights: Dict[str, float], label: str) -> str:
    if not weights:
        return "<span class='muted'>—</span>"
    mx = max(weights.values()) or 1.0
    rows = []
    for m, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * float(v) / mx
        rows.append(
            f"<div class='bar'><span class='bk'>{_esc(m)}</span>"
            f"<span class='bt'><span class='bf' style='width:{pct:.1f}%'></span></span>"
            f"<span class='bv'>{_num(v)}</span></div>")
    return f"<div class='bars' aria-label='{_esc(label)}'>" + "".join(rows) + "</div>"


def _vq_heatmap(vq: Dict[str, Dict]) -> str:
    if not vq:
        return "<span class='muted'>—</span>"
    rows = []
    for m, d in vq.items():
        n = max(int(d.get("n_codes", 64)), 1)
        cells = "".join(
            f"<i class='cell' title='code {int(c)}' "
            f"style='background:hsl({(int(c) * 360 / n):.0f} 70% 50%)'></i>"
            for c in (d.get("codes") or [])[:40])
        rows.append(
            f"<div class='vqrow'><span class='bk'>{_esc(m)}</span>"
            f"<span class='cells'>{cells}</span>"
            f"<span class='bv' title='codebook perplexity'>ppl {_num(d.get('perplexity'), 1)}"
            f"<span class='muted'>/{n}</span></span></div>")
    return f"<div class='vq'>{''.join(rows)}</div>"


# ------------------------------------------------------------------ per-trace block
def _winner_col(w: Dict) -> str:
    from dvxr.serve.utility import render_decision_curve_svg
    interval = w.get("interval") or []
    iv = f"[{_num(interval[0])}, {_num(interval[1])}]" if len(interval) == 2 else "—"
    dca = w.get("decision_curve")
    dca_svg = ""
    if isinstance(dca, dict):
        try:
            dca_svg = (f"<div class='dca'><div class='cap'>Decision curve — net benefit "
                       f"(Vickers &amp; Elkin), vs treat-all / treat-none</div>"
                       f"{render_decision_curve_svg(dca)}</div>")
        except Exception:  # noqa: BLE001
            dca_svg = ""
    narrative = (w.get("narrative") or {}).get("clinician", "") or w.get("caveat", "")
    drivers = w.get("drivers") or []
    drv = "".join(f"<li>{_esc(d.get('feature', d) if isinstance(d, dict) else d)}</li>"
                  for d in drivers[:5])
    auroc = w.get("heldout_auroc")
    auroc_s = w.get("heldout_auroc_subject")
    return f"""
    <section class="col winner">
      <div class="coltag">WINNER · single modality (validated)</div>
      <h3>{_esc(w.get('label',''))}</h3>
      <p class="enc">{_esc(w.get('encoder',''))}</p>
      <div class="big"><span class="prob">{_num(w.get('probability'))}</span>
        <span class="band band-{_esc((w.get('risk_band') or '').lower())}">{_esc(w.get('risk_band',''))}</span>
        <span class="ci">90% CI {iv}</span></div>
      <div class="cap">Per-window calibrated probability ({_esc(w.get('n_windows','?'))} windows)</div>
      <div class="ink">{_sparkline(w.get('window_probs'))}</div>
      <div class="metrics">
        <span>Held-out AUROC <b>{_num(auroc)}</b>{'' if auroc_s is None else f' · subject <b>{_num(auroc_s)}</b>'}</span>
        <span>ECE <b>{_num(w.get('ece'))}</b></span>
      </div>
      {dca_svg}
      {'<div class="drivers"><div class="cap">Top drivers</div><ul>'+drv+'</ul></div>' if drv else ''}
      <div class="llm"><div class="cap">LLM-generated assessment (grounded, explanation-only)</div>
        <p>{_esc(narrative)}</p></div>
    </section>"""


def _proposed_col(p: Dict) -> str:
    llm = p.get("llm") or {}
    llm_html = "<span class='muted'>LLM path not included</span>"
    if llm.get("included"):
        llm_html = (f"<div class='sub'>backend <code>{_esc(llm.get('backend',''))}</code>, "
                    f"pooled dim {_esc(llm.get('pooled_dim','?'))}</div>"
                    f"<div class='cap'>Modality attribution (L2 shift when a modality drops)</div>"
                    f"{_bars(llm.get('attribution') or {}, 'modality attribution')}"
                    f"<p class='role'>{_esc(llm.get('role',''))}</p>")
    prob = p.get("probability")
    prob_html = (f"<span class='prob'>{_num(prob)}</span>"
                 if prob is not None else "<span class='muted'>n/a</span>")
    return f"""
    <section class="col proposed">
      <div class="coltag">PROPOSED · multimodal fLLM (shown as-is)</div>
      <h3>VQ tokens → cross-modal attention → frozen-LLM soft prompts</h3>
      <div class="stage"><div class="cap">1 · Per-modality VQ tokenization (code indices + codebook perplexity)</div>
        {_vq_heatmap(p.get('vq') or {})}</div>
      <div class="stage"><div class="cap">2 · Cross-modal fusion attention (α over modalities, sums to 1)</div>
        {_bars(p.get('attention') or {}, 'cross-modal attention')}</div>
      <div class="stage"><div class="cap">3 · Frozen-LLM soft-prompt reader</div>{llm_html}</div>
      <div class="stage"><div class="cap">Proposed probability
        <span class="muted">(subject-held-out, single subject — cohort verdict below)</span></div>
        <div class="big">{prob_html}</div>
        <div class="note">{_esc(p.get('probability_note',''))}</div></div>
      <div class="asis">{_esc(p.get('note',''))}</div>
    </section>"""


def _scoreboard(sb: Dict) -> str:
    fo = (sb or {}).get("full_observation")
    dx = (sb or {}).get("dropout_crossover")
    if fo:
        verdict = fo.get("verdict", "")
        fo_html = (
            f"<div class='sbrow'><span class='sbk'>Best single-modality baseline</span>"
            f"<span class='sbv'>{_esc(fo.get('best_baseline','?'))} · error {_num(fo.get('base_err'))}</span></div>"
            f"<div class='sbrow'><span class='sbk'>Proposed fusion (full observation)</span>"
            f"<span class='sbv'>error {_num(fo.get('proposed_err'))} · "
            f"RER {_num(fo.get('rer_pct'),1)}%</span></div>"
            f"<div class='sbverdict'>{_esc(verdict)}</div>"
            f"<div class='muted src'>source: {_esc(fo.get('source_file',''))}</div>")
    else:
        fo_html = "<span class='muted'>no committed full-observation row for this task</span>"
    if dx and dx.get("crossover") is not None:
        model = f" ({_esc(dx.get('model'))})" if dx.get("model") else ""
        dx_html = (f"<div class='sbrow'><span class='sbk'>Sensor-dropout crossover</span>"
                   f"<span class='sbv good'>proposed wins{model} from {_esc(dx.get('crossover'))} "
                   f"dropped modalities</span></div><div class='muted'>{_esc(dx.get('note',''))}</div>")
    elif dx and dx.get("degradation"):
        # measured, but no CI-backed win — report the honest graceful-degradation nuance
        d = dx["degradation"]
        if d.get("narrows"):
            body = (f"no CI-backed win — floor leads at every level, but the gap narrows from "
                    f"RER {_num(d.get('rer_at_0_dropped'), 1)}% (0 dropped) to "
                    f"{_num(d.get('best_rer'), 1)}% ({_esc(d.get('best_k'))} dropped): graceful degradation")
        else:
            body = "no CI-backed win — the floor leads at every dropout level"
        dx_html = (f"<div class='sbrow'><span class='sbk'>Sensor-dropout robustness</span>"
                   f"<span class='sbv muted'>{body}</span></div>")
    else:
        dx_html = ("<div class='sbrow'><span class='sbk'>Sensor-dropout crossover</span>"
                   "<span class='sbv muted'>not recorded for this task "
                   "(run the streaming showdown)</span></div>")
    return f"""
    <section class="scoreboard">
      <div class="coltag">HONEST SCOREBOARD · full-observation cohort verdict</div>
      {fo_html}
      {dx_html}
      <p class="sbnote">The proposed multimodal path loses on full-observation accuracy; its genuine
        edge is graceful degradation under missing sensors. Both facts are shown here rather than hidden.</p>
    </section>"""


def _trace_block(t: Dict, idx: int) -> str:
    ood = (t.get("source") == "upload") or (not t.get("validated"))
    badge = ("<span class='ood'>⚠ out-of-distribution sample — a pipeline demonstration, "
             "NOT the validated cohort number</span>" if ood
             else "<span class='ok'>✓ held-out cohort subject — carries the validated number</span>")
    return f"""
    <div class="trace" data-trace="{idx}" {'style="display:none"' if idx else ''}>
      <div class="subhead"><span class="sid">subject <b>{_esc(t.get('subject','?'))}</b></span>{badge}</div>
      <div class="cols">
        {_winner_col(t.get('winner') or {})}
        {_proposed_col(t.get('proposed') or {})}
      </div>
      {_scoreboard(t.get('scoreboard') or {})}
    </div>"""


# ------------------------------------------------------------------ page
def render_glassbox(traces: List[Dict], title: str = "DVXR Screen — Glass-box") -> str:
    if not traces:
        raise ValueError("render_glassbox needs at least one trace")
    options = "".join(
        f"<option value='{i}'>subject {_esc(t.get('subject','?'))}"
        f"{' (sample)' if t.get('source') == 'upload' else ''}</option>"
        for i, t in enumerate(traces))
    selector = (f"<label class='sel'>Subject / sample entry "
                f"<select id='pick' onchange='pick(this.value)'>{options}</select></label>"
                if len(traces) > 1 else "")
    blocks = "".join(_trace_block(t, i) for i, t in enumerate(traces))
    synth = any(t.get("note") for t in traces)
    synth_banner = ("<div class='synth'>Rendered from a synthetic fixture (torch/data unavailable) — "
                    "numbers are illustrative; the scoreboard still reads the committed board.</div>"
                    if synth else "")
    js = ("function pick(v){document.querySelectorAll('.trace').forEach(function(d){"
          "d.style.display=(d.getAttribute('data-trace')===v)?'block':'none';});}"
          if len(traces) > 1 else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root {{ --paper:#f5f7fa; --panel:#ffffff; --fg:#16202e; --muted:#5b6b80; --line:#dde4ee;
  --accent:#0e8f8c; --accent-soft:#d6efee; --ink-soft:#eef2f7; --good:#1a9850; --warn:#e0952b; --bad:#cf3b2f; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --paper:#0c1220; --panel:#141d2e; --fg:#e9eef6; --muted:#93a1b8; --line:#243247;
    --accent:#3fd0c9; --accent-soft:#123634; --ink-soft:#101a2a; }} }}
:root[data-theme="dark"] {{ --paper:#0c1220; --panel:#141d2e; --fg:#e9eef6; --muted:#93a1b8;
  --line:#243247; --accent:#3fd0c9; --accent-soft:#123634; --ink-soft:#101a2a; }}
:root[data-theme="light"] {{ --paper:#f5f7fa; --panel:#ffffff; --fg:#16202e; --muted:#5b6b80;
  --line:#dde4ee; --accent:#0e8f8c; --accent-soft:#d6efee; --ink-soft:#eef2f7; }}
* {{ box-sizing:border-box; }}
body {{ --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg); background:var(--paper);
  margin:0; line-height:1.5; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 20px 72px; }}
.num, .prob, .ci, .bv, .sbv, .metrics b {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); }}
.hero {{ margin-top:30px; padding:26px 24px; border:1px solid var(--line); border-radius:16px;
  background:linear-gradient(160deg,var(--panel),var(--ink-soft)); }}
.hero h1 {{ font-size:28px; margin:8px 0 6px; letter-spacing:-.4px; text-wrap:balance; }}
.hero p {{ color:var(--muted); max-width:70ch; margin:0; }}
.disclaimer {{ margin-top:14px; padding:10px 12px; border-left:3px solid var(--warn);
  background:var(--ink-soft); border-radius:6px; font-size:13px; color:var(--fg); }}
.sel {{ display:inline-flex; gap:8px; align-items:center; margin:22px 0 6px; font-size:13px; color:var(--muted); }}
.sel select {{ font:inherit; padding:4px 8px; border-radius:8px; border:1px solid var(--line);
  background:var(--panel); color:var(--fg); }}
.synth {{ margin:14px 0; padding:8px 12px; border:1px dashed var(--line); border-radius:8px;
  color:var(--muted); font-size:12px; }}
.subhead {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:18px 0 10px; }}
.sid {{ font-size:14px; }}
.ok {{ color:var(--good); font-size:12px; }} .ood {{ color:var(--warn); font-size:12px; font-weight:600; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:820px) {{ .cols {{ grid-template-columns:1fr; }} }}
.col {{ border:1px solid var(--line); border-radius:14px; padding:16px 16px 18px; background:var(--panel); }}
.col.proposed {{ background:linear-gradient(180deg,var(--panel),var(--ink-soft)); }}
.coltag {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--accent); margin-bottom:6px; }}
.col h3 {{ font-size:16px; margin:2px 0 4px; }} .enc {{ color:var(--muted); font-size:12.5px; margin:0 0 10px; }}
.big {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.prob {{ font-size:30px; font-weight:700; color:var(--accent); }}
.band {{ font-size:11px; padding:2px 8px; border-radius:20px; border:1px solid var(--line); text-transform:capitalize; }}
.band-elevated,.band-high {{ color:var(--bad); border-color:var(--bad); }}
.band-watch {{ color:var(--warn); border-color:var(--warn); }} .band-low {{ color:var(--good); border-color:var(--good); }}
.ci {{ font-size:12px; color:var(--muted); }}
.cap {{ font-size:11px; color:var(--muted); margin:12px 0 5px; }}
.ink {{ color:var(--fg); }}
.metrics {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; color:var(--muted); margin-top:8px; }}
.metrics b {{ color:var(--fg); }}
.drivers ul {{ margin:2px 0 0; padding-left:18px; font-size:12.5px; }}
.llm {{ margin-top:12px; padding:10px 12px; border-radius:8px; background:var(--ink-soft); }}
.llm p {{ margin:2px 0 0; font-size:13px; }}
.stage {{ margin-top:6px; }}
.bars {{ display:flex; flex-direction:column; gap:5px; }}
.bar {{ display:grid; grid-template-columns:64px 1fr 52px; align-items:center; gap:8px; }}
.bk {{ font-size:12px; color:var(--muted); }} .bv {{ font-size:11.5px; text-align:right; color:var(--fg); }}
.bt {{ height:9px; border-radius:5px; background:var(--ink-soft); overflow:hidden; }}
.bf {{ display:block; height:100%; background:var(--accent); }}
.vq {{ display:flex; flex-direction:column; gap:5px; }}
.vqrow {{ display:grid; grid-template-columns:64px 1fr auto; align-items:center; gap:8px; }}
.cells {{ display:flex; flex-wrap:wrap; gap:2px; }}
.cell {{ width:11px; height:11px; border-radius:2px; display:inline-block; }}
.sub {{ font-size:12px; color:var(--muted); }} .sub code {{ font-family:var(--mono); }}
.role {{ font-size:11.5px; color:var(--muted); margin:6px 0 0; }}
.note {{ font-size:11.5px; color:var(--muted); }}
.asis {{ margin-top:12px; padding:8px 10px; border-left:3px solid var(--warn); background:var(--ink-soft);
  border-radius:6px; font-size:12px; }}
.scoreboard {{ margin-top:16px; border:1px solid var(--line); border-radius:14px; padding:16px; background:var(--panel); }}
.sbrow {{ display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px solid var(--line); font-size:13px; }}
.sbk {{ color:var(--muted); }} .sbv.good {{ color:var(--good); }} .sbv.muted {{ color:var(--muted); }}
.sbverdict {{ margin-top:8px; font-weight:600; color:var(--bad); }}
.src {{ font-size:11px; }} .sbnote {{ font-size:12.5px; color:var(--muted); margin:10px 0 0; }}
.muted {{ color:var(--muted); }}
code {{ font-family:var(--mono); }}
svg {{ max-width:100%; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">Glass-box · proposed multimodal fLLM vs the winning model</div>
    <h1>How the proposed model actually works — shown honestly, side by side</h1>
    <p>The same subject flows through both pipelines. Left: the validated single-modality screener that
      actually wins. Right: the proposed multimodal fLLM — VQ tokenization, cross-modal attention, and a
      frozen-LLM soft-prompt reader — shown exactly as it performs. The scoreboard states the honest
      verdict: fusion loses on full-observation accuracy; its real edge is missing-sensor robustness.</p>
    <div class="disclaimer">{_esc(DISCLAIMER)}</div>
  </header>
  {selector}
  {synth_banner}
  {blocks}
</div>
{f'<script>{js}</script>' if js else ''}
</body>
</html>"""
