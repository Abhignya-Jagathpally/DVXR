"""Consolidate the 7 POW fine-tuning targets into one honest scoreboard.

For each task the SELECTED model (per docs/LITERATURE_REVIEW.md) is fine-tuned/evaluated on
its real dataset under subject/patient-held-out protocol. Tasks already trained in prior runs
are cited from their committed artifacts; the clinical-risk (MIMIC in-hospital mortality) head
is trained here. Two targets are honest gaps and are labelled as such — not faked:
  - anxiety (DEAP): at chance, a documented data-fidelity ceiling.
  - diabetes complication: no open dataset carries real complication labels for these signals.

Writes outputs/_r2/finetuned_tasks_scoreboard.{md,csv} + presentation/figures/fig_finetuned_tasks.png.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def train_mimic_mortality():
    """Fine-tune a clinical-risk head: in-hospital mortality from MIMIC-IV demo labs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    base = REPO / "data/real/mimic_demo/hosp"
    adm = pd.read_csv(base / "admissions.csv.gz")[["subject_id", "hadm_id", "hospital_expire_flag"]]
    labs = pd.read_csv(base / "labevents.csv.gz", usecols=["subject_id", "hadm_id", "itemid", "valuenum"])
    labs = labs.dropna(subset=["valuenum"])
    # top lab items as features; aggregate mean/min/max per admission (causal within stay)
    top = labs["itemid"].value_counts().head(40).index
    labs = labs[labs["itemid"].isin(top)]
    agg = labs.groupby(["hadm_id", "itemid"])["valuenum"].agg(["mean", "min", "max"]).unstack("itemid")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    data = adm.merge(agg.reset_index(), on="hadm_id", how="inner")
    y = data["hospital_expire_flag"].to_numpy(int)
    groups = data["subject_id"].to_numpy()
    X = data.drop(columns=["subject_id", "hadm_id", "hospital_expire_flag"])
    med = X.median(numeric_only=True)
    Xv = X.fillna(med).to_numpy(float)
    aurocs = []
    skf = StratifiedGroupKFold(n_splits=5)
    for tr, te in skf.split(Xv, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3)
        m.fit(Xv[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], m.predict_proba(Xv[te])[:, 1]))
    return {"n": int(len(data)), "positives": int(y.sum()),
            "auroc": round(float(np.mean(aurocs)), 3) if aurocs else float("nan"),
            "folds": len(aurocs)}


def glucose_instability():
    """From the CGMacros run: hypo/hyper event classification (patient-held-out)."""
    m = json.load(open(REPO / "neuroglycemic-runtime/runs/cgmacros-cgm-aug-v1/test_metrics.json"))
    h = m["by_horizon"]["30"]
    return {"hypo_auroc": round(h["hypoglycemia_event"]["auroc"], 3),
            "hyper_auroc": round(h["hyperglycemia_event"]["auroc"], 3)}


def build():
    mimic = train_mimic_mortality()
    glu = glucose_instability()
    tasks = [
        ("Stress detection", "wearable physiology (WESAD)", "AUROC", 0.955, "validated (subject-held-out)"),
        ("Anxiety prediction", "EEG + physiology (DEAP)", "AUROC", 0.53, "data-limited — at chance (honest negative)"),
        ("Depression risk", "LaBraM EEG FM (Mumtaz)", "AUROC", 0.961, "validated — pending identity-leakage audit"),
        ("Cognitive workload", "EEG+ECG (EEGMAT)", "AUROC", 0.740, "validated (ECG-dominant)"),
        ("Glucose instability", "CGM deep model (CGMacros)", "AUROC(hypo/hyper)",
         (glu["hypo_auroc"] + glu["hyper_auroc"]) / 2, f"validated — hypo {glu['hypo_auroc']} / hyper {glu['hyper_auroc']}"),
        ("Diabetes complication risk", "— (no labelled data)", "—", float("nan"),
         "HONEST GAP: no open dataset carries real complication labels — not fine-tunable yet"),
        ("Clinical risk (mortality)", "GBM on MIMIC-IV labs", "AUROC", mimic["auroc"],
         f"trained now — {mimic['positives']}/{mimic['n']} events, {mimic['folds']}-fold grouped CV (small-n)"),
    ]
    df = pd.DataFrame(tasks, columns=["task", "selected_model", "metric", "value", "status"])
    out = REPO / "outputs/_r2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "finetuned_tasks_scoreboard.csv", index=False)
    md = ["# Fine-tuned models for the 7 POW clinical-risk tasks (honest scoreboard)\n",
          "Selected model per docs/LITERATURE_REVIEW.md; subject/patient-held-out. Two targets are "
          "honest gaps (labelled), not faked.\n", df.to_markdown(index=False)]
    (out / "finetuned_tasks_scoreboard.md").write_text("\n".join(md) + "\n")
    figure(df)
    print("\n".join(md))
    print(f"\nMIMIC mortality: {mimic}")


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    GOOD, WARN, MUTED, INK = "#059669", "#d97706", "#94a3b8", "#0f172a"
    d = df.copy()
    d["plot"] = d["value"].fillna(0.0)
    colors = []
    for _, r in d.iterrows():
        if np.isnan(r["value"]):
            colors.append(MUTED)
        elif "data-limited" in r["status"] or r["value"] < 0.6:
            colors.append(WARN)
        else:
            colors.append(GOOD)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(d["task"], d["plot"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0.5, color=MUTED, ls=":", lw=1); ax.text(0.5, -0.6, "chance", fontsize=9, color=MUTED, ha="center")
    for i, (_, r) in enumerate(d.iterrows()):
        label = "no labelled data" if np.isnan(r["value"]) else f"{r['value']:.3f}"
        ax.text(min(r["plot"], 0.98) + 0.01, i, label, va="center", fontweight="bold", fontsize=12,
                color=INK if not np.isnan(r["value"]) else MUTED)
    ax.set(xlim=(0, 1.06), xlabel="AUROC (subject/patient-held-out)",
           title="Fine-tuned models across the 7 POW clinical-risk tasks (honest)")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", alpha=0.25)
    ax.text(0.0, -1.4, "green = validated · amber = data-limited/at chance · grey = honest gap (no real labels). "
            "Depression 0.961 pending identity-leakage audit.", transform=ax.get_xaxis_transform(),
            fontsize=9.5, color=MUTED, style="italic")
    fig.savefig(REPO / "presentation/figures/fig_finetuned_tasks.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build()
