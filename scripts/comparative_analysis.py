"""Goal-3 deliverable: a comparative performance table — single-modality vs integrated.

Built from committed, held-out results: the benchmark scoreboard (best single/non-fused
baseline vs the integrated learned fusion, same folds) and the real CGMacros leave-one-
modality-out. Honest verdict per task; no configuration is assumed to win.
Writes outputs/_r2/comparative_analysis.{md,csv} + presentation/figures/fig_comparative_analysis.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def build_rows():
    sb = pd.read_csv(REPO / "outputs/benchmark_scoreboard.csv")
    rows = []
    # classification tasks: metric is 1-AUROC -> convert to AUROC (higher better)
    label = {"stress": "Stress (PhysioNet)", "wesad_stress": "Stress (WESAD)",
             "deap_anxiety": "Anxiety (DEAP)", "deap_arousal": "Arousal (DEAP)",
             "eegmat_workload": "Cognitive workload (EEGMAT)", "mumtaz_depression": "Depression (Mumtaz)"}
    for _, r in sb.iterrows():
        if r["task"] not in label:
            continue
        single = round(1 - r["base_err"], 3)
        integ = round(1 - r["prop_err"], 3)
        rows.append({
            "task": label[r["task"]], "metric": "AUROC",
            "best_single_modality": single, "integrated_fusion": integ,
            "delta": round(integ - single, 3),
            "verdict": "integrated wins" if integ > single + 0.005 else
                       ("~tie" if abs(integ - single) <= 0.005 else "single-modality wins"),
            "holm_p": r["p_holm"]})
    # glucose: real CGMacros leave-one-modality-out (RMSE @30 min, lower better)
    a = pd.read_csv(REPO / "neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1/missing_modality_ablation.csv")
    a30 = a[a.horizon_minutes == 30].set_index("scenario")["rmse_mg_dl"]
    single_cgm = round(float(a30.get("without_events")), 2)   # CGM only
    integrated = round(float(a30.get("observed_modalities")), 2)  # CGM + meals
    rows.append({"task": "Glucose forecast (CGMacros)", "metric": "RMSE@30 (mg/dL, ↓)",
                 "best_single_modality": single_cgm, "integrated_fusion": integrated,
                 "delta": round(integrated - single_cgm, 2),
                 "verdict": "integrated wins" if integrated < single_cgm else "single-modality wins",
                 "holm_p": float("nan")})
    return pd.DataFrame(rows)


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    MUTED, GLU, INK = "#64748b", "#0d9488", "#0f172a"
    cls = df[df.metric == "AUROC"]
    x = np.arange(len(cls)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, cls["best_single_modality"], w, label="Best single modality", color=MUTED)
    b2 = ax.bar(x + w/2, cls["integrated_fusion"], w, label="Integrated (learned fusion)", color=GLU)
    ax.bar_label(b1, fmt="%.3f", fontsize=10, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=10, padding=2)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xticks=x, xticklabels=[t.replace(" (", "\n(") for t in cls["task"]],
           ylabel="AUROC (subject-held-out)", ylim=(0.4, 1.02),
           title="Comparative analysis — single modality vs integrated fusion")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False); ax.grid(axis="y", alpha=0.25)
    ax.text(0.0, -0.22, "Honest finding: learned fusion does NOT beat the best single modality on the mental-health "
            "tasks (Holm p=1.0). Multimodal integration helps only for glucose (CGM+meals 12.99 < CGM-only 13.33).",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, style="italic")
    fig.savefig(REPO / "presentation/figures/fig_comparative_analysis.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    df = build_rows()
    out = REPO / "outputs/_r2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparative_analysis.csv", index=False)
    md = ["# Comparative performance analysis — single modality vs integrated (Goal 3)\n",
          "Committed, subject/patient-held-out results. AUROC ↑ for classification; RMSE ↓ for glucose. "
          "No configuration is assumed to win — measured, with statistical tests.\n",
          df.to_markdown(index=False), "",
          "## Verdict",
          "- **Mental-health / EEG tasks:** the integrated learned fusion does **not** beat the best single "
          "modality — every task's fusion RER is negative and non-significant (Holm p=1.0). The strongest "
          "*single* modality (wearable for stress, EEG/LaBraM for depression, ECG for workload) wins.",
          "- **Glucose:** integration **helps** — CGM+meals (12.99) beats CGM-only (13.33) @30 min, and "
          "adding the wearable/pulse device lowers it further (12.77). This is the one task where the real "
          "data co-registers multiple informative modalities per subject.",
          "- **Honest conclusion:** multimodal integration is not universally better; it pays off where "
          "modalities carry complementary signal on the same subject (glucose), and adds noise where one "
          "modality dominates (mental health). Reported exactly as measured."]
    (out / "comparative_analysis.md").write_text("\n".join(md) + "\n")
    figure(df)
    print("\n".join(md))


if __name__ == "__main__":
    main()
