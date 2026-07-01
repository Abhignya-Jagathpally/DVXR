"""Goal 2 (multimodal late fusion) + Goal 3 (single-vs-multimodal ablation).

On the EMOTIV 4-class command task (Left/Right/Push/Pull), decode from each
modality stream captured for the SAME windows / SAME label:

  * EEG     — per-channel band power (Welch + Emotiv FFT)
  * Motion  — MOT.* accelerometer / quaternion / magnetometer (mean+std)
  * PM      — PM.* affective performance metrics (stress/engagement/...)

then combine per-modality decoders by **weighted late fusion** (confidence-weighted
average of out-of-fold class probabilities). All modalities share identical
leakage-controlled CV folds (StratifiedGroupKFold by command-trial), so the fused
out-of-fold probabilities are honestly held-out and the ablation is apples-to-apples.

Run:  venv/bin/python scripts/run_fusion_ablation.py
Outputs: outputs/bci/ablation.csv, ablation.png, fusion.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score

from goal1_pipeline.bci_real import ingest_emotiv, epoch_emotiv, feature_cols

OUT = ROOT / "outputs" / "bci"
OUT.mkdir(parents=True, exist_ok=True)
EMOTIV_ZIP = ROOT / "data" / "EmotivBCI-AJ_EPOCX_142080_2026.06.08T15.15.46.05.00.zip"
CMD = ["Left", "Right", "Push", "Pull"]
PALETTE = {"EEG": "#2456c7", "Motion": "#1c9e77", "PM (affective)": "#e8a33d",
           "Fusion (mean)": "#7a4fd6", "Fusion (conf-weighted)": "#d6455d"}


def _pipe():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def oof_proba(X, y, cv, groups, classes):
    """Out-of-fold class probabilities, columns aligned to `classes`."""
    proba = cross_val_predict(_pipe(), X, y, cv=cv, groups=groups, method="predict_proba")
    fit_classes = list(_pipe().fit(X, y).named_steps["c"].classes_)
    return proba[:, [fit_classes.index(c) for c in classes]]


def score(y, proba, classes):
    pred = np.array(classes)[proba.argmax(1)]
    return (balanced_accuracy_score(y, pred),
            f1_score(y, pred, average="macro", labels=classes))


def main():
    print("[fusion] ingest + epoch (EEG + motion + PM) ...")
    emo = ingest_emotiv(EMOTIV_ZIP)
    win = epoch_emotiv(emo, win_s=2.0, step_s=0.5, power_thresh=0.05)
    win = win[win["label"].isin(CMD)].reset_index(drop=True)
    y = win["label"].to_numpy()
    groups = win["trial_id"].to_numpy()
    print(f"    {len(win)} engaged windows | classes { {c:int((y==c).sum()) for c in CMD} }")

    modalities = {
        "EEG": feature_cols(win, "eeg"),
        "Motion": feature_cols(win, "motion"),
        "PM (affective)": feature_cols(win, "pm"),
    }
    modalities = {k: v for k, v in modalities.items() if v}
    n_splits = int(max(2, min(5, pd.Series(groups).groupby(y).nunique().min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    chance = 1.0 / len(CMD)

    # Per-modality out-of-fold probabilities (identical folds across modalities).
    probas, rows = {}, []
    for name, fc in modalities.items():
        p = oof_proba(win[fc].to_numpy(), y, cv, groups, CMD)
        probas[name] = p
        ba, mf = score(y, p, CMD)
        rows.append({"modality": name, "n_features": len(fc),
                     "balanced_acc": round(ba, 4), "macro_f1": round(mf, 4)})
        print(f"    [{name:16s}] feats={len(fc):3d}  balAcc={ba:.3f}  macroF1={mf:.3f}")

    # Late fusion — simple mean of per-modality probabilities.
    stack = np.stack(list(probas.values()))           # (M, N, C)
    p_mean = stack.mean(0)
    ba_m, mf_m = score(y, p_mean, CMD)
    rows.append({"modality": "Fusion (mean)", "n_features": sum(len(v) for v in modalities.values()),
                 "balanced_acc": round(ba_m, 4), "macro_f1": round(mf_m, 4)})
    print(f"    [Fusion mean      ] balAcc={ba_m:.3f}  macroF1={mf_m:.3f}")

    # Late fusion — confidence-weighted (weight ∝ modality CV balanced-acc above chance).
    w = np.array([max(1e-3, r["balanced_acc"] - chance) for r in rows[:len(modalities)]])
    w = w / w.sum()
    p_cw = (stack * w[:, None, None]).sum(0)
    ba_c, mf_c = score(y, p_cw, CMD)
    rows.append({"modality": "Fusion (conf-weighted)",
                 "n_features": sum(len(v) for v in modalities.values()),
                 "balanced_acc": round(ba_c, 4), "macro_f1": round(mf_c, 4)})
    print(f"    [Fusion conf-wtd  ] balAcc={ba_c:.3f}  macroF1={mf_c:.3f}  "
          f"weights={dict(zip(modalities, np.round(w,2)))}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation.csv", index=False)
    best_single = table.iloc[:len(modalities)]["balanced_acc"].max()
    summary = {
        "task": "EMOTIV 4-class command (Left/Right/Push/Pull)",
        "chance": chance, "n_windows": int(len(win)), "cv_splits": n_splits,
        "rows": rows, "best_single_modality_balAcc": float(best_single),
        "fusion_conf_weighted_balAcc": float(ba_c),
        "fusion_gain_vs_best_single": round(float(ba_c - best_single), 4),
        "weights": {k: round(float(wi), 3) for k, wi in zip(modalities, w)},
    }
    (OUT / "fusion.json").write_text(json.dumps(summary, indent=2))

    # Ablation figure
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = table["modality"].tolist()
    vals = table["balanced_acc"].tolist()
    colors = [PALETTE.get(n, "#888") for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(chance, ls="--", color="#888", label=f"chance = {chance:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("balanced accuracy (held-out)")
    ax.set_title("Goal 3 ablation — single modality vs late fusion\n(EMOTIV 4-class command, leakage-controlled CV)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=130)
    plt.close(fig)
    print(f"[done] best single={best_single:.3f}  fusion(conf)={ba_c:.3f}  "
          f"gain={ba_c-best_single:+.3f} -> {OUT}/ablation.png")


if __name__ == "__main__":
    main()
