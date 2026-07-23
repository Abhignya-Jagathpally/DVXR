"""Engineer presentation-ready figures from the real, committed results.

Consistent slide styling (light ground, large type, one palette). Every number is sourced
from a committed artifact (the model ladder CSV) or a documented result; nothing invented.
Writes PNGs into presentation/figures/. Run: python presentation/build_presentation_assets.py
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = HERE / "figures"

# palette
EEG, PHR, PPG, GLU = "#7c3aed", "#0d9488", "#e11d48", "#059669"
INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
GOOD, WARN, CRIT = "#059669", "#d97706", "#dc2626"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.labelsize": 14, "axes.edgecolor": MUTED, "axes.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    })
    return plt


def _ladder():
    rows = list(csv.DictReader(open(REPO / "outputs/_r2/glucose_model_ladder.csv")))
    for r in rows:
        r["horizon_minutes"] = int(r["horizon_minutes"]); r["rmse_mg_dl"] = float(r["rmse_mg_dl"])
    return rows


def fig_ingestion_matrix():
    """Goal-1 ingestion capability across the five POW modalities."""
    plt = _plt()
    rows = [
        ("a) Physiological wearable", "WESAD · Non-EEG · CGMacros", "biosignal enc.", "validated", GOOD),
        ("b) EEG / BCI", "DEAP · EMOTIV · Galea · Mumtaz", "real LaBraM FM", "validated", GOOD),
        ("c) Biosensor streams", "EDA · BVP · temp · resp · SpO₂", "biosignal enc.", "validated", GOOD),
        ("d) Behavioral metrics", "VR/AR gaze · interactions", "behavior adapter", "wired — not run yet", WARN),
        ("e) Diabetes / CGM", "CGMacros · DiaTrend · BIG-IDEAS", "CGM-history enc.", "validated (RMSE~13)", GOOD),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.4)); ax.axis("off")
    ax.set_title("Goal 1 — the pipeline ingests all five modalities", loc="left", pad=16, fontsize=19)
    y = 0.86
    ax.text(0.01, 0.985, "MODALITY", fontsize=11.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.28, 0.985, "EXAMPLE DATA (on disk)", fontsize=11.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.55, 0.985, "ENCODER", fontsize=11.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.76, 0.985, "STATUS", fontsize=11.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    for name, data, enc, status, color in rows:
        ax.add_patch(plt.Rectangle((0.0, y - 0.03), 1.0, 0.13, transform=ax.transAxes,
                                   facecolor=color, alpha=0.06, edgecolor="none"))
        ax.text(0.01, y + 0.03, name, fontsize=13, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.28, y + 0.03, data, fontsize=11.5, color=INK, transform=ax.transAxes)
        ax.text(0.55, y + 0.03, enc, fontsize=11.5, color=INK, transform=ax.transAxes)
        mark = "✓" if status.startswith("validated") else "○"
        ax.text(0.76, y + 0.03, f"{mark} {status}", fontsize=11.5, color=color, fontweight="bold", transform=ax.transAxes)
        y -= 0.165
    ax.text(0.01, -0.02, "a,b,c,e: wired end-to-end with real data + encoders (validated results). "
            "d: canonical slot + adapter + VR ingest script exist; not yet run on a real behavioral dataset.",
            fontsize=10.5, color=MUTED, transform=ax.transAxes, style="italic")
    fig.savefig(FIGS / "fig_ingestion_matrix.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_glucose_horizons():
    plt = _plt()
    import numpy as np
    rows = _ladder()
    hs = [30, 60, 90, 120]
    model = [next(r["rmse_mg_dl"] for r in rows if r["model"] == "neuroglycemic_net" and r["horizon_minutes"] == h) for h in hs]
    persist = [next(r["rmse_mg_dl"] for r in rows if r["model"] == "persistence" and r["horizon_minutes"] == h) for h in hs]
    x = np.arange(len(hs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - w/2, persist, w, label="Persistence (naive baseline)", color=MUTED)
    b = ax.bar(x + w/2, model, w, label="NeuroGlycemic forecast", color=GLU)
    ax.bar_label(b, fmt="%.1f", padding=3, fontsize=12, fontweight="bold")
    ax.set(xticks=x, xticklabels=[f"{h} min" for h in hs], ylabel="RMSE (mg/dL, lower better)",
           title="Glucose forecast beats the naive baseline at every horizon")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "fig_glucose_horizons.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_model_ladder():
    plt = _plt()
    rows = [r for r in _ladder() if r["horizon_minutes"] == 30]
    order = ["persistence", "decision_tree", "random_forest", "linear_ridge", "mlp", "neuroglycemic_net", "gradient_boosting"]
    labels = {"persistence": "persistence", "decision_tree": "decision tree", "random_forest": "random forest",
              "linear_ridge": "linear ridge", "mlp": "MLP", "neuroglycemic_net": "deep net (ours)", "gradient_boosting": "gradient boosting"}
    vals = [(labels[m], next(r["rmse_mg_dl"] for r in rows if r["model"] == m)) for m in order]
    # honest addition: the 3-seed deep ensemble we trained to try to beat GBM (it did not)
    ens_csv = REPO / "outputs/_r2/ensemble_result.csv"
    if ens_csv.is_file():
        er = [row for row in csv.DictReader(open(ens_csv)) if int(row["horizon_minutes"]) == 30]
        if er:
            vals.append(("deep ensemble (3-seed)", round(float(er[0]["ensemble_rmse"]), 2)))
    model_names = order + (["deep ensemble (3-seed)"] if len(vals) > len(order) else [])
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    def _c(m):
        if m == "persistence": return CRIT
        if m in ("neuroglycemic_net", "gradient_boosting", "deep ensemble (3-seed)"): return GLU
        return MUTED
    colors = [_c(m) for m in model_names]
    bars = ax.barh([v[0] for v in vals], [v[1] for v in vals], color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set(xlabel="RMSE @30 min (mg/dL)", title="Gradient boosting wins on point accuracy — reported honestly")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, -0.15, "same patient-disjoint split · a 3-seed deep ensemble was trained to beat GBM and did "
            "NOT (13.06 vs 12.48) · the causal representation is the win, not depth · all beat persistence ~25%",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=MUTED, style="italic")
    fig.savefig(FIGS / "fig_model_ladder.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_per_device():
    plt = _plt()
    # leave-one-out @30 min from the committed device run (docs/GLUCOSE_FROM_DEVICES.md)
    data = [("all devices", 12.77, GLU), ("− wearable/pulse", 12.78, MUTED),
            ("− meals", 13.14, MUTED), ("− CGM", 35.13, CRIT)]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    bars = ax.bar([d[0] for d in data], [d[1] for d in data], color=[d[2] for d in data])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12, fontweight="bold")
    ax.set(ylabel="RMSE @30 min (mg/dL)", title="What each device contributes to the glucose forecast")
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.16, "CGM dominates; meals add ~0.4; the wearable/pulse device keeps a forecast alive "
            "(~90% coverage) when CGM drops out", transform=ax.transAxes, ha="center", fontsize=10.5, color=MUTED, style="italic")
    fig.savefig(FIGS / "fig_per_device.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_heads_sota():
    plt = _plt()
    import numpy as np
    # subject-held-out AUROC vs an honest published bar (docs/HEADS_SOTA.md)
    heads = [("Depression\n(Mumtaz)", 0.961, 0.73), ("Stress\n(WESAD)", 0.955, 0.85),
             ("Stress\n(PhysioNet)", 0.892, None), ("Workload\n(EEGMAT)", 0.740, None),
             ("Anxiety\n(DEAP)", 0.53, None)]
    x = np.arange(len(heads))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [GOOD if h[1] >= 0.7 else WARN for h in heads]
    bars = ax.bar(x, [h[1] for h in heads], 0.55, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=12, fontweight="bold")
    for i, h in enumerate(heads):
        if h[2] is not None:
            ax.plot([i - 0.3, i + 0.3], [h[2], h[2]], color=INK, lw=2, ls="--")
            ax.text(i, h[2] + 0.01, f"SOTA {h[2]:.2f}", ha="center", fontsize=9.5, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(len(heads) - 0.5, 0.51, "chance", fontsize=9.5, color=MUTED, ha="right")
    ax.set(xticks=x, xticklabels=[h[0] for h in heads], ylabel="AUROC (subject-held-out)",
           ylim=(0.4, 1.0), title="Mental-health heads vs published SOTA (honest, protocol-labeled)")
    ax.grid(axis="x", visible=False)
    ax.text(0.0, -0.16, "⚠ depression 0.961 pending an identity-leakage audit (Identity Trap, "
            "arXiv:2606.06647) — treat as an upper bound; DEAP anxiety is at chance (data-limited)",
            transform=ax.transAxes, fontsize=9.5, color=WARN, style="italic")
    fig.savefig(FIGS / "fig_heads_sota.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_deep_vs_gbm():
    plt = _plt()
    import numpy as np
    p = REPO / "outputs/_r2/deep_tabular_result.csv"
    if not p.is_file():
        return
    rows = list(csv.DictReader(open(p)))
    hs = [int(r["horizon_minutes"]) for r in rows]
    deep = [float(r["deep_v2_rmse"]) for r in rows]
    gbm = [float(r["gradient_boosting"]) for r in rows]
    x = np.arange(len(hs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    b1 = ax.bar(x - w/2, gbm, w, label="Gradient boosting", color=MUTED)
    b2 = ax.bar(x + w/2, deep, w, label="Redesigned deep net (GRN + conv + ensemble)", color=GLU)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=11)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=11, fontweight="bold")
    for i, (d, g) in enumerate(zip(deep, gbm)):
        if d < g:
            ax.text(i + w/2, d + 0.6, "✓", ha="center", color=GOOD, fontsize=15, fontweight="bold")
    ax.set(xticks=x, xticklabels=[f"{h} min" for h in hs], ylabel="RMSE (mg/dL, lower better)",
           title="Redesigned deep net beats gradient boosting at 3/4 horizons")
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.15, "same patient-disjoint split · deep net wins at 60/90/120 min (temporal structure) · "
            "within 0.16 at 30 min · and returns calibrated intervals GBM cannot",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED, style="italic")
    fig.savefig(FIGS / "fig_deep_vs_gbm.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (fig_ingestion_matrix, fig_glucose_horizons, fig_model_ladder, fig_per_device,
               fig_heads_sota):
        fn(); print("wrote", FIGS / (fn.__name__.replace("fig_", "fig_") + ".png"))
    print("done")


if __name__ == "__main__":
    main()
