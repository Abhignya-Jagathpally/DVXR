"""dvxr.serve.utility — clinical-utility (decision-curve / net-benefit) analysis.

AUROC says the ranking is good; it does not say whether *acting* on the screen helps. Decision-curve
analysis answers that directly: across the range of decision thresholds a clinician might use, is the
net benefit of screening-then-acting higher than the two default policies — treat everyone
(screen-all) and treat no one (screen-none)?

Net benefit (Vickers & Elkin, 2006, Med Decis Making, doi:10.1177/0272989X06295361):

    NB(p_t) = TP/n  -  (FP/n) · p_t/(1 - p_t)

where a subject is flagged when its calibrated risk ≥ p_t, and p_t/(1-p_t) is the odds at the
threshold — i.e. how many false alarms a clinician trades for one true catch. `treat-all` is
`prevalence - (1-prevalence)·p_t/(1-p_t)`; `treat-none` is 0. A screener is *useful* over the
threshold band where its curve sits above BOTH defaults.

Computed from the screener's held-out out-of-fold predictions (the same ones behind the AUROC), so it
never leaks and always traces. Honest by construction: we report the band where it helps AND note
where it doesn't (falls to treat-all/none). Research-grade screening, not diagnosis.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# A clinically plausible threshold band. Below ~5% almost anything beats treat-none; above ~75% the
# odds weight explodes and few real screening decisions live there.
_DEFAULT_THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(15)]  # 0.05 … 0.75

# A win only counts as clinically meaningful if it clears the best default by at least this much net
# benefit (≈ half a net true case per 100 patients) and holds across ≥2 thresholds — a single-point,
# hair-thin crossing is noise, not utility, and must not read as "useful".
_MEANINGFUL_GAIN = 5e-3


def net_benefit(y: Sequence[int], prob: Sequence[float], p_t: float) -> float:
    """Net benefit of flagging `prob >= p_t` at threshold `p_t` (Vickers & Elkin, 2006)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    n = len(y)
    if n == 0 or p_t <= 0.0 or p_t >= 1.0:
        return 0.0
    flag = prob >= p_t
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    w = p_t / (1.0 - p_t)
    return tp / n - (fp / n) * w


def _gain(y: np.ndarray, prob: np.ndarray, p_t: float, prev: float) -> float:
    """Net-benefit gain of the model over the stronger default (treat-all / treat-none) at p_t."""
    nb_all = prev - (1.0 - prev) * (p_t / (1.0 - p_t))
    return net_benefit(y, prob, p_t) - max(nb_all, 0.0)


def decision_curve(y: Sequence[int], prob: Sequence[float],
                   thresholds: Optional[Sequence[float]] = None,
                   n_boot: int = 500, seed: int = 7) -> dict:
    """Net-benefit curve for the model vs treat-all vs treat-none over a threshold band.

    Returns ``{"prevalence", "n", "points": [{"threshold","model","all","none"}...], "summary": {...}}``
    where ``summary`` states the threshold band over which the model beats BOTH defaults and the best
    net-benefit gain. ``useful`` is only True when that advantage is **bootstrap-stable** — the point
    estimate can win by chance (a random score's expected net benefit is dominated by the best default,
    yet a single draw fluctuates), so we require the peak gain's one-sided 95% bootstrap lower bound to
    stay positive. Honest by construction: noise-level advantages read as not useful.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    m = np.isfinite(prob)
    y, prob = y[m], prob[m]
    n = int(len(y))
    prev = float(np.mean(y == 1)) if n else 0.0
    ths = list(thresholds) if thresholds is not None else _DEFAULT_THRESHOLDS

    points, wins = [], []
    for p_t in ths:
        w = p_t / (1.0 - p_t)
        nb_model = net_benefit(y, prob, p_t)
        nb_all = prev - (1.0 - prev) * w
        points.append({"threshold": round(float(p_t), 4), "model": round(nb_model, 5),
                       "all": round(nb_all, 5), "none": 0.0})
        if nb_model > max(nb_all, 0.0) + _MEANINGFUL_GAIN:
            wins.append((p_t, nb_model - max(nb_all, 0.0)))

    useful, band, best_t, best_gain, lo_gain = False, None, None, 0.0, 0.0
    if len(wins) >= 2:
        band = (round(min(w for w, _ in wins), 4), round(max(w for w, _ in wins), 4))
        best_t, best_gain = max(wins, key=lambda t: t[1])
        # bootstrap the peak-threshold gain; keep "useful" only if it stays positive at the 5th pct.
        if n >= 2 and n_boot > 0:
            rng = np.random.default_rng(seed)
            boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                yb, pb = y[idx], prob[idx]
                boot[b] = _gain(yb, pb, best_t, float(np.mean(yb == 1)))
            lo_gain = float(np.percentile(boot, 5))
            useful = lo_gain > 0.0
        else:
            useful = True

    if useful:
        note = (f"Screening beats both treat-all and treat-none for decision thresholds "
                f"{int(band[0]*100)}–{int(band[1]*100)}%; peak added net benefit {best_gain:.3f} at "
                f"{int(best_t*100)}% (bootstrap 95% lower bound {lo_gain:.3f} > 0) — ≈ {best_gain:.3f} "
                f"extra true cases caught per patient at no added false-alarm cost vs the best default.")
    else:
        note = ("Over the evaluated band the screener does not show a bootstrap-stable net-benefit "
                "advantage over treat-all / treat-none — reported honestly, not hidden.")
    summary = {"useful": useful, "useful_band": band if useful else None,
               "best_threshold": round(float(best_t), 4) if useful else None,
               "best_gain": round(float(best_gain), 5) if useful else 0.0,
               "best_gain_lo": round(float(lo_gain), 5), "note": note}
    return {"prevalence": round(prev, 4), "n": n, "points": points, "summary": summary}


def render_decision_curve_svg(curve: dict, width: int = 460, height: int = 240) -> str:
    """A self-contained inline SVG of the net-benefit curve (no external resources)."""
    pts = curve.get("points", [])
    if not pts:
        return "<svg width='0' height='0'></svg>"
    pad_l, pad_b, pad_t, pad_r = 42, 30, 16, 12
    xs = [p["threshold"] for p in pts]
    ys = [v for p in pts for v in (p["model"], p["all"], p["none"])]
    y_hi = max(ys + [0.01]); y_lo = min(ys + [0.0])
    span = (y_hi - y_lo) or 1.0

    def X(t): return pad_l + (t - xs[0]) / ((xs[-1] - xs[0]) or 1.0) * (width - pad_l - pad_r)
    def Y(v): return pad_t + (y_hi - v) / span * (height - pad_t - pad_b)

    def path(key, color, dash=""):
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['threshold']):.1f},{Y(p[key]):.1f}"
                     for i, p in enumerate(pts))
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'{da}/>"

    # Axis text + grid inherit the container's ink via currentColor (theme-neutral: readable on the
    # report's dark panel and the evidence page's light one); the three curves keep fixed brand hues.
    zero_y = Y(0.0)
    grid = (f"<line x1='{pad_l}' y1='{zero_y:.1f}' x2='{width-pad_r}' y2='{zero_y:.1f}' "
            f"stroke='currentColor' stroke-opacity='0.25' stroke-width='1'/>")
    xticks = "".join(
        f"<text x='{X(t):.1f}' y='{height-8}' font-size='9' fill='currentColor' fill-opacity='0.7' "
        f"text-anchor='middle'>{int(t*100)}%</text>"
        for t in (xs[0], xs[len(xs)//2], xs[-1]))
    axis_lbl = (f"<text x='{pad_l}' y='{pad_t-4}' font-size='9' fill='currentColor' "
                f"fill-opacity='0.7'>net benefit</text>"
                f"<text x='{width-pad_r}' y='{height-8}' font-size='9' fill='currentColor' "
                f"fill-opacity='0.7' text-anchor='end'>decision threshold</text>")
    legend = ("<g font-size='9' fill='currentColor'>"
              f"<rect x='{pad_l}' y='{pad_t}' width='10' height='3' fill='#3b82f6'/>"
              f"<text x='{pad_l+14}' y='{pad_t+4}'>DVXR screen</text>"
              f"<rect x='{pad_l+86}' y='{pad_t}' width='10' height='3' fill='#f59e0b'/>"
              f"<text x='{pad_l+100}' y='{pad_t+4}'>treat all</text>"
              f"<rect x='{pad_l+156}' y='{pad_t}' width='10' height='3' fill='#94a3b8'/>"
              f"<text x='{pad_l+170}' y='{pad_t+4}'>treat none</text></g>")
    return (f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            f"style='max-width:100%;height:auto' "
            f"role='img' aria-label='Decision-curve analysis: net benefit vs decision threshold'>"
            f"{grid}{path('none', '#94a3b8')}{path('all', '#f59e0b', '4 3')}"
            f"{path('model', '#3b82f6')}{xticks}{axis_lbl}{legend}</svg>")


def subject_aggregate(subjects: Sequence, prob: Sequence[float], y: Sequence[int]):
    """Collapse window-level (prob, y) to one row per subject (mean prob, subject label).

    Only valid when each subject carries a single class (subject-level-diagnosis tasks). Returns
    ``(subject_y, subject_prob)`` aligned by unique subject.
    """
    subjects = np.asarray(subjects)
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y).astype(int)
    uniq = list(dict.fromkeys(subjects.tolist()))
    sy, sp = [], []
    for s in uniq:
        mask = subjects == s
        sy.append(int(round(float(np.mean(y[mask])))))
        sp.append(float(np.mean(prob[mask])))
    return np.asarray(sy), np.asarray(sp)
