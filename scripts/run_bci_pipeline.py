"""End-to-end BCI pipeline on the REAL collected EMOTIV + Galea recordings.

Decodes intended cube movement (Neutral / Left / Right / Push / Pull) from EMOTIV
EEG — the wearable-BCI analog of the real-time neural-manifold avatar decoding in
Busch et al. (avatarRT / MRAE / TPHATE) — and produces tangible artifacts:

  outputs/bci/manifold_emotiv.png        neural manifold (PHATE) colored by command
  outputs/bci/confusion_matrix.png       leakage-controlled decoding confusion matrix
  outputs/bci/per_class_accuracy.png     per-class recall vs chance
  outputs/bci/realtime_decode.png        streaming P(command) control signal over time
  outputs/bci/channel_band_importance.png explainable EEG biomarkers
  outputs/bci/galea_quality.png          Galea multi-device signal-quality + rest manifold
  outputs/bci/metrics.json               all metrics
  outputs/bci/dashboard.html             self-contained dashboard embedding everything

Run:  venv/bin/python scripts/run_bci_pipeline.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score, f1_score,
)

from goal1_pipeline.bci_real import (
    ingest_emotiv, ingest_galea, epoch_emotiv, feature_cols,
    temporal_diffusion_map, COMMAND_CLASSES, EPOCX_CHANNELS, EEG_BANDS,
)

OUT = ROOT / "outputs" / "bci"
OUT.mkdir(parents=True, exist_ok=True)
EMOTIV_ZIP = ROOT / "data" / "EmotivBCI-AJ_EPOCX_142080_2026.06.08T15.15.46.05.00.zip"
GALEA_ZIP = ROOT / "data" / "OpenBCISession_2026-06-08_11-23-34.zip"

PALETTE = {"Neutral": "#8a94a6", "Left": "#2456c7", "Right": "#d6455d",
           "Push": "#1c9e77", "Pull": "#e8a33d"}


def manifold_embed(X, times, n_components=3):
    """Use real PHATE if available, else the built-in temporal diffusion map."""
    try:
        import phate
        op = phate.PHATE(n_components=n_components, knn=15, decay=20,
                         t="auto", verbose=False, random_state=0, n_jobs=1)
        return op.fit_transform(X), "PHATE"
    except Exception:
        return temporal_diffusion_map(X, n_components=n_components, k=15,
                                      t_diffusion=8, temporal_weight=0.0,
                                      times=times), "diffusion-map"


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    t_start = time.time()
    report = {"generated_by": "scripts/run_bci_pipeline.py"}
    imgs: dict[str, str] = {}

    # === 1. INGEST =========================================================
    print("[1] Ingesting real recordings ...")
    emo = ingest_emotiv(EMOTIV_ZIP)
    print(f"    EMOTIV {emo.meta['serial']}: {len(emo.eeg):,} samples @ {emo.fs:.0f} Hz "
          f"({emo.duration_s:.0f}s), {len(emo.ch_names)} EEG ch, {len(emo.mc):,} MC samples")
    report["emotiv"] = {"serial": emo.meta["serial"], "fs": emo.fs,
                        "duration_s": round(emo.duration_s, 1),
                        "n_eeg_samples": len(emo.eeg), "n_channels": len(emo.ch_names)}

    # === 2. EPOCH + LABEL ==================================================
    print("[2] Epoching into labeled windows ...")
    win = epoch_emotiv(emo, win_s=2.0, step_s=0.5, power_thresh=0.05)
    win = win[win["label"].isin(COMMAND_CLASSES)].reset_index(drop=True)
    dist = win["label"].value_counts().reindex(COMMAND_CLASSES).fillna(0).astype(int)
    print(f"    {len(win):,} windows | label distribution: {dist.to_dict()}")
    print(f"    {win['trial_id'].nunique()} leakage-control trials")
    report["windows"] = {"n": int(len(win)), "distribution": dist.to_dict(),
                         "n_trials": int(win["trial_id"].nunique())}

    y = win["label"].to_numpy()
    groups = win["trial_id"].to_numpy()
    times = win["t_center"].to_numpy()

    # === 3. DECODER: leakage-controlled cross-validation ===================
    print("[3] Decoding (StratifiedGroupKFold by trial) across feature sets ...")
    # n_splits limited by the rarest class's trial count.
    trials_per_class = (win.groupby("label")["trial_id"].nunique())
    n_splits = int(max(2, min(5, trials_per_class.min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    classes = COMMAND_CLASSES

    feat_results = {}
    proba_for_cm = None
    for kind in ["welch", "pow", "all"]:
        fc = feature_cols(win, kind)
        if not fc:
            continue
        Xf = win[fc].to_numpy()
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        proba = cross_val_predict(pipe, Xf, y, cv=cv, groups=groups,
                                  method="predict_proba")
        pred = np.array(classes)[proba.argmax(1)]
        # align proba columns to `classes` order
        fit_classes = list(pipe.fit(Xf, y).named_steps["c"].classes_)
        col_order = [fit_classes.index(c) for c in classes]
        proba_ord = proba[:, col_order]
        bal_acc = balanced_accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, average="macro", labels=classes)
        try:
            auroc = roc_auc_score(pd.get_dummies(y)[classes].to_numpy(), proba_ord,
                                  average="macro", multi_class="ovr")
        except Exception:
            auroc = float("nan")
        feat_results[kind] = {"n_features": len(fc), "balanced_acc": round(float(bal_acc), 4),
                              "macro_f1": round(float(macro_f1), 4), "macro_auroc": round(float(auroc), 4)}
        print(f"    [{kind:5s}] feats={len(fc):3d}  balAcc={bal_acc:.3f}  "
              f"macroF1={macro_f1:.3f}  macroAUROC={auroc:.3f}")
        if kind == "welch":
            proba_for_cm = (proba_ord, pred)
    report["decoding"] = feat_results
    report["chance_balanced_acc"] = round(1.0 / len(classes), 4)
    report["cv"] = {"scheme": "StratifiedGroupKFold by command-trial", "n_splits": n_splits}

    # === 3b. TWO-STAGE DECODE (avatar-control framing) =====================
    # Stage A: engaged (any command) vs Neutral. Stage B: which of the 4 commands.
    print("[3b] Two-stage decode: engaged-detector + 4-class command ...")
    pow_fc = feature_cols(win, "pow")
    best_fc = pow_fc if pow_fc else feature_cols(win, "welch")
    Xb = win[best_fc].to_numpy()

    # Stage A — engaged vs neutral (binary AUROC), leakage-controlled.
    y_eng = (y != "Neutral").astype(int)
    cvb = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    pa = cross_val_predict(
        Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        Xb, y_eng, cv=cvb, groups=groups, method="predict_proba")[:, 1]
    eng_auroc = roc_auc_score(y_eng, pa)
    print(f"     stage-A engaged-vs-neutral AUROC = {eng_auroc:.3f}")

    # Stage B — command-only 4-class among engaged windows.
    cmd_classes = ["Left", "Right", "Push", "Pull"]
    em = np.isin(y, cmd_classes)
    yc = y[em]
    gc_grp = groups[em]
    Xc = win.loc[em, best_fc].to_numpy()
    tc_per = pd.Series(yc).groupby(yc).count()
    nsc = int(max(2, min(5, pd.Series(gc_grp).groupby(yc).nunique().min())))
    cvc = StratifiedGroupKFold(n_splits=nsc, shuffle=True, random_state=0)
    pc = cross_val_predict(
        Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        Xc, yc, cv=cvc, groups=gc_grp, method="predict_proba")
    fit_c = list(LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xc, yc).classes_)
    predc = np.array(fit_c)[pc.argmax(1)]
    cmd_bal = balanced_accuracy_score(yc, predc)
    cmd_f1 = f1_score(yc, predc, average="macro", labels=cmd_classes)
    print(f"     stage-B 4-class command balAcc = {cmd_bal:.3f}  macroF1 = {cmd_f1:.3f}  "
          f"(chance {1/4:.2f}, n={len(yc)})")
    report["two_stage"] = {
        "engaged_vs_neutral_auroc": round(float(eng_auroc), 4),
        "command_4class_balanced_acc": round(float(cmd_bal), 4),
        "command_4class_macro_f1": round(float(cmd_f1), 4),
        "command_chance": 0.25, "n_engaged_windows": int(em.sum()),
        "feature_set": "pow" if pow_fc else "welch",
    }
    # command-only confusion matrix
    cmc = confusion_matrix(yc, predc, labels=cmd_classes)
    cmcn = cmc / cmc.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cmcn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(cmd_classes, rotation=20, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(cmd_classes)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmcn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmcn[i, j] > 0.5 else "#333", fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true command")
    ax.set_title(f"Stage B — 4-class command decode\n(engaged windows only, balAcc={cmd_bal:.2f}, chance=0.25)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "command_confusion.png", dpi=130)
    imgs["command_confusion"] = fig_to_b64(fig)

    # === 3c. RIGOR: temporal-block CV + lateralization sanity check ========
    # Drift control: split EACH command's windows chronologically into K contiguous
    # blocks; fold k tests block-k of every class and trains on the rest. This keeps
    # all 4 classes in train AND test while separating test windows in time from
    # their training windows -> if accuracy holds, it is not nearby-in-time leakage.
    print("[3c] Rigor: temporal-block CV (per-class chronological) ...")
    tc = times[em]
    K = 4
    test_fold = np.full(len(yc), -1, dtype=int)
    for cls in cmd_classes:
        idx = np.where(yc == cls)[0]
        idx = idx[np.argsort(tc[idx])]              # chronological within class
        for k, chunk in enumerate(np.array_split(idx, K)):
            test_fold[chunk] = k
    predt = np.empty(len(yc), dtype=object)
    for k in range(K):
        tr, te = test_fold != k, test_fold == k
        if te.sum() == 0 or len(np.unique(yc[tr])) < 2:
            continue
        clf = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xc[tr], yc[tr])
        predt[te] = clf.predict(Xc[te])
    valid = predt != None  # noqa: E711
    tcmd_bal = balanced_accuracy_score(yc[valid], predt[valid].astype(str))
    tcmd_f1 = f1_score(yc[valid], predt[valid].astype(str), average="macro", labels=cmd_classes)
    print(f"     temporal-block 4-class balAcc = {tcmd_bal:.3f}  macroF1 = {tcmd_f1:.3f} "
          f"(vs trial-grouped {cmd_bal:.3f})")

    # Lateralization sanity: Left vs Right from a single hemisphere-contrast feature.
    LEFT_H = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]
    RIGHT_H = ["AF4", "F8", "F4", "FC6", "T8", "P8", "O2"]
    def hemi_beta(side):
        cols = [f"eeg_{ch}_{b}" for ch in side for b in ("betaL", "betaH")]
        cols = [c for c in cols if c in win.columns]
        return win.loc[em, cols].to_numpy().mean(1)
    lat = hemi_beta(RIGHT_H) - hemi_beta(LEFT_H)     # >0 = relatively more right-hemi beta
    lr = np.isin(yc, ["Left", "Right"])
    lat_auroc = float("nan")
    if lr.sum() > 10 and len(np.unique(yc[lr])) == 2:
        ybin = (yc[lr] == "Right").astype(int)
        s = lat[lr]
        lat_auroc = roc_auc_score(ybin, s)
        lat_auroc = max(lat_auroc, 1 - lat_auroc)    # direction-agnostic
    print(f"     Left-vs-Right lateralization AUROC (single hemi-beta contrast) = {lat_auroc:.3f}")

    report["two_stage"]["temporal_block_balanced_acc"] = round(float(tcmd_bal), 4)
    report["two_stage"]["temporal_block_macro_f1"] = round(float(tcmd_f1), 4)
    report["two_stage"]["lateralization_lr_auroc"] = round(float(lat_auroc), 4)

    # === 4. NEURAL LATENT (MRAE-style autoencoder) =========================
    print("[4] Neural manifold latent (BIOT/MRAE-style autoencoder) ...")
    welch_fc = feature_cols(win, "welch")
    latent_decode = None
    try:
        from goal1_pipeline.neural_encoders import NeuralBiosignalEncoder
        enc = NeuralBiosignalEncoder(embedding_dim=16, hidden_dim=64, n_layers=2,
                                     n_heads=4, epochs=40, seed=7)
        emb = enc.fit_transform(win, welch_fc)
        enc.save(str(OUT / "emotiv_encoder.pt"))
        latent = emb.to_numpy()
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        proba = cross_val_predict(pipe, latent, y, cv=cv, groups=groups, method="predict_proba")
        pred = np.array(classes)[proba.argmax(1)]
        latent_decode = {"balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
                         "macro_f1": round(float(f1_score(y, pred, average="macro", labels=classes)), 4),
                         "embedding_dim": latent.shape[1]}
        print(f"    latent[{latent.shape[1]}d] balAcc={latent_decode['balanced_acc']:.3f} "
              f"macroF1={latent_decode['macro_f1']:.3f}")
        report["neural_latent_decode"] = latent_decode
    except Exception as e:  # torch missing or other
        print(f"    (skipped neural latent: {e})")
        latent = win[welch_fc].to_numpy()

    # === 5. MANIFOLD FIGURE ===============================================
    print("[5] Manifold embedding for visualization ...")
    coords, method = manifold_embed(win[best_fc].to_numpy(), times, n_components=3)
    report["manifold_method"] = method
    # command-only manifold (engaged windows) so the command geometry is legible
    cmd_coords, _ = manifold_embed(win.loc[em, best_fc].to_numpy(),
                                   times[em], n_components=2)
    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    # full manifold: Neutral faint background, commands bold on top
    mneu = y == "Neutral"
    ax1.scatter(coords[mneu, 0], coords[mneu, 1], coords[mneu, 2], s=5, alpha=0.12,
                color=PALETTE["Neutral"], label="Neutral")
    for cls in cmd_classes:
        m = y == cls
        if m.any():
            ax1.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=16, alpha=0.85,
                        color=PALETTE[cls], label=cls)
    ax1.set_title(f"Full EEG manifold ({method}) — commands vs neutral", fontsize=11, fontweight="bold")
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
    ax1.legend(fontsize=8, loc="upper left")
    # command-only 2D manifold
    for cls in cmd_classes:
        m = yc == cls
        if m.any():
            ax2.scatter(cmd_coords[m, 0], cmd_coords[m, 1], s=22, alpha=0.8,
                        color=PALETTE[cls], label=cls)
    ax2.set_title("Command-only manifold (engaged windows)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("dim 1"); ax2.set_ylabel("dim 2"); ax2.legend(fontsize=9, loc="best")
    fig.suptitle("Intended-movement structure in the EEG manifold (avatarRT / TPHATE analog)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "manifold_emotiv.png", dpi=130)
    imgs["manifold"] = fig_to_b64(fig)

    # === 6. CONFUSION MATRIX ==============================================
    proba_ord, pred = proba_for_cm
    cm = confusion_matrix(y, pred, labels=classes)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#333", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true command")
    ax.set_title(f"Leakage-controlled decoding confusion matrix\n(row-normalized, "
                 f"balAcc={feat_results['welch']['balanced_acc']:.2f}, chance={1/len(classes):.2f})",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=130)
    imgs["confusion"] = fig_to_b64(fig)

    # === 7. PER-CLASS RECALL vs CHANCE ====================================
    rep = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    recalls = [rep[c]["recall"] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(classes, recalls, color=[PALETTE[c] for c in classes])
    ax.axhline(1 / len(classes), ls="--", color="#888", label=f"chance = {1/len(classes):.2f}")
    for b, r in zip(bars, recalls):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall (held-out)")
    ax.set_title("Per-command decoding recall vs chance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_accuracy.png", dpi=130)
    imgs["per_class"] = fig_to_b64(fig)

    # === 8. REAL-TIME DECODE TRACE ========================================
    # Uses the HELD-OUT (out-of-fold) probabilities so the trace matches the
    # reported metrics rather than an optimistic in-sample fit. Two compact
    # full-session strips (true vs decoded command) + a readable zoomed segment.
    print("[6] Streaming decode trace (held-out) ...")
    from matplotlib.colors import ListedColormap
    order = np.argsort(times)
    ts_o = times[order]
    proba_o = proba_ord[order]                      # held-out, columns == classes
    y_o = y[order]
    dec_o = np.array(classes)[proba_o.argmax(1)]
    cmap = ListedColormap([PALETTE[c] for c in classes])
    code = {c: i for i, c in enumerate(classes)}
    true_codes = np.array([[code[c] for c in y_o]])
    dec_codes = np.array([[code[c] for c in dec_o]])
    extent = [ts_o.min(), ts_o.max(), 0, 1]

    # smoothed per-command probability (rolling mean ~2.5 s) for the zoom panel
    pdf = pd.DataFrame(proba_o, columns=classes).rolling(5, center=True, min_periods=1).mean()

    # pick the ~90 s segment richest in command variety for the zoom
    seg = 90.0
    best_s, best_score = ts_o.min(), -1
    for s in np.arange(ts_o.min(), ts_o.max() - seg, 15.0):
        msk = (ts_o >= s) & (ts_o < s + seg)
        nm = y_o[msk] != "Neutral"
        sc = nm.sum() + 25 * len(set(y_o[msk][nm]))
        if sc > best_score:
            best_score, best_s = sc, s
    zmask = (ts_o >= best_s) & (ts_o < best_s + seg)

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.5, 3.2], hspace=0.55)
    axt = fig.add_subplot(gs[0]); axd = fig.add_subplot(gs[1], sharex=axt)
    axz = fig.add_subplot(gs[2])
    for a, codes, lab in [(axt, true_codes, "true"), (axd, dec_codes, "decoded")]:
        a.imshow(codes, aspect="auto", cmap=cmap, vmin=0, vmax=len(classes) - 1,
                 extent=extent, interpolation="nearest")
        a.set_yticks([0.5]); a.set_yticklabels([lab], fontsize=10)
        a.set_xlim(extent[0], extent[1])
    axt.set_title("Full-session command timeline — true (top) vs held-out decoded (bottom)",
                  fontsize=11, fontweight="bold")
    axt.tick_params(labelbottom=False)
    axd.axvspan(best_s, best_s + seg, color="k", alpha=0.0)
    axd.add_patch(plt.Rectangle((best_s, 0), seg, 1, fill=False, edgecolor="k", lw=1.6))
    axd.set_xlabel("time (s)")
    # zoom: smoothed P(command) lines + true-command shading
    for cls in cmd_classes:
        axz.plot(ts_o[zmask], pdf[cls].to_numpy()[zmask], color=PALETTE[cls], lw=2.0, label=cls)
    for i in np.where(zmask)[0]:
        if y_o[i] != "Neutral":
            axz.axvspan(ts_o[i] - 0.25, ts_o[i] + 0.25, color=PALETTE[y_o[i]], alpha=0.12)
    axz.set_xlim(best_s, best_s + seg); axz.set_ylim(0, 1)
    axz.set_xlabel("time (s)"); axz.set_ylabel("P(command), smoothed")
    axz.set_title(f"Zoom: {seg:.0f}s segment [{best_s:.0f}–{best_s+seg:.0f}s] — decoder tracking command onsets "
                  f"(shading = true command)", fontsize=11, fontweight="bold")
    axz.legend(ncol=4, fontsize=9, loc="upper right")
    fig.suptitle("Real-time streaming decode — avatar control signal (held-out probabilities)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "realtime_decode.png", dpi=130, bbox_inches="tight")
    imgs["realtime"] = fig_to_b64(fig)

    # === 9. CHANNEL x BAND IMPORTANCE (explainable biomarkers) ============
    print("[7] Explainable channel x band importance ...")
    full = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(
        win[welch_fc].to_numpy(), y)
    clf = full.named_steps["c"]
    coef = np.abs(clf.coef_).mean(0)  # mean |weight| across classes per feature
    imp = pd.Series(coef, index=welch_fc)
    bands = list(EEG_BANDS.keys())
    mat = np.zeros((len(EPOCX_CHANNELS), len(bands)))
    for fi, fname in enumerate(welch_fc):
        # eeg_<ch>_<band>
        parts = fname.split("_")
        ch, band = parts[1], parts[2]
        if ch in EPOCX_CHANNELS and band in bands:
            mat[EPOCX_CHANNELS.index(ch), bands.index(band)] = imp[fname]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)
    ax.set_yticks(range(len(EPOCX_CHANNELS))); ax.set_yticklabels(EPOCX_CHANNELS)
    ax.set_title("Explainable EEG biomarkers\n(mean |decoder weight|, channel × band)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "channel_band_importance.png", dpi=130)
    imgs["importance"] = fig_to_b64(fig)
    top_feats = imp.sort_values(ascending=False).head(10)
    report["top_biomarkers"] = {k: round(float(v), 4) for k, v in top_feats.items()}

    # === 10. GALEA (multi-device ingestion + quality + rest manifold) =====
    print("[8] Galea multi-device ingestion + signal quality ...")
    try:
        gal = ingest_galea(GALEA_ZIP, max_seconds=120)
        usable = gal.quality[gal.quality["usable"]]["channel"].tolist()
        print(f"    Galea: {len(gal.eeg):,} samples @ {gal.fs:.0f} Hz, "
              f"{len(usable)}/{len(gal.ch_names)} usable channels")
        report["galea"] = {"fs": gal.fs, "n_samples": len(gal.eeg),
                           "n_channels": len(gal.ch_names), "n_usable": len(usable),
                           "usable_channels": usable}
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        q = gal.quality
        colors = ["#1c9e77" if u else "#d6455d" for u in q["usable"]]
        axes[0].bar(q["channel"], np.log10(q["std_uv"].clip(lower=1)), color=colors)
        axes[0].set_title("Galea per-channel signal quality\n(green=usable, red=railed/poor)",
                          fontsize=11, fontweight="bold")
        axes[0].set_ylabel("log10 std (uV)"); axes[0].tick_params(axis="x", rotation=90)
        # rest manifold from short band-power windows on usable channels
        if usable:
            from scipy import signal as sps
            t = gal.eeg["t"].to_numpy()
            G = gal.eeg[usable].to_numpy()
            wn, st = int(2 * gal.fs), int(1 * gal.fs)
            rows = []
            s = 0
            while s + wn <= len(t):
                seg = G[s:s + wn]
                feats = []
                for ci in range(seg.shape[1]):
                    f, psd = sps.welch(seg[:, ci], fs=gal.fs, nperseg=min(wn, int(gal.fs)))
                    tot = np.trapezoid(psd, f) + 1e-12
                    for lo, hi in EEG_BANDS.values():
                        mm = (f >= lo) & (f < hi)
                        feats.append(np.trapezoid(psd[mm], f[mm]) / tot if mm.any() else 0)
                rows.append(feats)
                s += st
            GR = np.array(rows)
            gc, gm = manifold_embed(GR, np.arange(len(GR)), n_components=2)
            sc = axes[1].scatter(gc[:, 0], gc[:, 1], c=np.arange(len(gc)), cmap="viridis", s=14)
            axes[1].set_title(f"Galea resting-EEG manifold ({gm})\ncolored by time",
                              fontsize=11, fontweight="bold")
            axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
            fig.colorbar(sc, ax=axes[1], label="time index", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT / "galea_quality.png", dpi=130)
        imgs["galea"] = fig_to_b64(fig)
    except Exception as e:
        print(f"    (Galea step skipped: {e})")

    # === 10b. MULTIMODAL FUSION + ABLATION (Goal 2/3) =====================
    print("[9] Multimodal late fusion + ablation (EEG / motion / affective) ...")
    try:
        import run_fusion_ablation
        run_fusion_ablation.main()
        abl_png = OUT / "ablation.png"
        if abl_png.exists():
            imgs["ablation"] = base64.b64encode(abl_png.read_bytes()).decode("ascii")
        if (OUT / "fusion.json").exists():
            report["fusion_ablation"] = json.loads((OUT / "fusion.json").read_text())
    except Exception as e:
        print(f"    (fusion ablation skipped: {e})")

    # === 11. SAVE METRICS + WINDOWS =======================================
    report["runtime_s"] = round(time.time() - t_start, 1)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    win.to_csv(OUT / "emotiv_windows.csv", index=False)

    # === 12. DASHBOARD ====================================================
    build_dashboard(report, imgs, feat_results, latent_decode)
    print(f"[done] {report['runtime_s']}s -> {OUT}/dashboard.html")


def build_dashboard(report, imgs, feat_results, latent_decode):
    def img_block(key, title):
        if key not in imgs:
            return ""
        return f'<div class="card"><h3>{title}</h3><img src="data:image/png;base64,{imgs[key]}"/></div>'

    rows = ""
    for k, v in feat_results.items():
        rows += (f"<tr><td>{k}</td><td>{v['n_features']}</td>"
                 f"<td>{v['balanced_acc']:.3f}</td><td>{v['macro_f1']:.3f}</td>"
                 f"<td>{v['macro_auroc']:.3f}</td></tr>")
    if latent_decode:
        rows += (f"<tr><td>neural latent (AE)</td><td>{latent_decode['embedding_dim']}</td>"
                 f"<td>{latent_decode['balanced_acc']:.3f}</td><td>{latent_decode['macro_f1']:.3f}</td>"
                 f"<td>—</td></tr>")
    chance = report["chance_balanced_acc"]
    dist = report["windows"]["distribution"]
    emo = report["emotiv"]
    ts = report.get("two_stage", {})
    # Goal 2/3 ablation table
    abl = report.get("fusion_ablation", {})
    abl_rows = ""
    for r in abl.get("rows", []):
        hl = "background:#1d3a2b" if r["modality"].startswith("Fusion") else ""
        abl_rows += (f"<tr style='{hl}'><td>{r['modality']}</td><td>{r['n_features']}</td>"
                     f"<td>{r['balanced_acc']:.3f}</td><td>{r['macro_f1']:.3f}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DVXR BCI Pipeline — Real EMOTIV + Galea</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0e1726;color:#e6edf6}}
header{{padding:26px 34px;background:linear-gradient(120deg,#15233b,#1d3a6b)}}
h1{{margin:0 0 6px;font-size:23px}} header p{{margin:2px 0;color:#a8b9d4;font-size:14px}}
.wrap{{padding:22px 34px;max-width:1280px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}}
.card{{background:#16223a;border:1px solid #243757;border-radius:12px;padding:14px}}
.card h3{{margin:0 0 10px;font-size:15px;color:#cfe0ff}}
.card img{{width:100%;border-radius:8px;background:#fff}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.kpi{{background:#16223a;border:1px solid #243757;border-radius:12px;padding:14px 18px;min-width:150px}}
.kpi b{{display:block;font-size:26px;color:#7fd1a8}} .kpi span{{color:#9fb1cf;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:7px 10px;border-bottom:1px solid #243757;text-align:left}} th{{color:#9fb1cf}}
code{{background:#0c1322;padding:2px 6px;border-radius:5px;color:#9ad0ff}}
.full{{grid-column:1/-1}}
</style></head><body>
<header>
<h1>DVXR — Multimodal BCI Pipeline · Real Collected Data</h1>
<p>EMOTIV EPOC X mental-command decoding (Neutral / Left / Right / Push / Pull) — the wearable-BCI analog of real-time neural-manifold avatar decoding (avatarRT · MRAE · TPHATE).</p>
<p>EMOTIV serial <code>{emo['serial']}</code> · {emo['duration_s']:.0f}s · {emo['fs']:.0f} Hz · {emo['n_channels']} EEG channels · manifold method <code>{report.get('manifold_method','?')}</code> · generated in {report['runtime_s']}s</p>
</header>
<div class="wrap">
<div class="kpis">
  <div class="kpi"><b>{ts.get('command_4class_balanced_acc', float('nan')):.2f}</b><span>4-class command balAcc · trial-grouped CV (chance 0.25)</span></div>
  <div class="kpi"><b>{ts.get('temporal_block_balanced_acc', float('nan')):.2f}</b><span>4-class command balAcc · temporal-block CV (drift-controlled)</span></div>
  <div class="kpi"><b>{ts.get('lateralization_lr_auroc', float('nan')):.2f}</b><span>Left-vs-Right lateralization AUROC</span></div>
  <div class="kpi"><b>{ts.get('engaged_vs_neutral_auroc', float('nan')):.2f}</b><span>engaged-vs-neutral AUROC (≈chance is good)</span></div>
  <div class="kpi"><b>{report['windows']['n']}</b><span>EEG windows · {report['windows']['n_trials']} trials</span></div>
</div>

<div class="grid">
  <div class="card full"><h3>Decoding performance by feature set <span style="font-weight:400;color:#9fb1cf">— {report['cv']['scheme']}, {report['cv']['n_splits']} folds</span></h3>
    <table><tr><th>feature set</th><th># features</th><th>balanced acc</th><th>macro F1</th><th>macro AUROC</th></tr>{rows}</table>
    <p style="color:#9fb1cf;font-size:13px">Window label distribution: {dist} &nbsp;·&nbsp; chance balanced-accuracy = {chance:.2f}</p>
  </div>
  {img_block('manifold','Neural manifold colored by intended command')}
  {img_block('command_confusion','Stage B — 4-class command decode (engaged windows)')}
  {img_block('confusion','5-class confusion matrix (leakage-controlled)')}
  {img_block('per_class','Per-command recall vs chance')}
  {img_block('importance','Explainable EEG biomarkers (channel × band)')}
  {('<div class="card"><h3>Goal 2/3 — multimodal late fusion &amp; ablation</h3>'
    '<table><tr><th>modality</th><th># feat</th><th>bal acc</th><th>macro F1</th></tr>'
    + abl_rows + '</table><p style="color:#9fb1cf;font-size:12px">EMOTIV 4-class command · '
    'fusion = confidence-weighted late fusion of held-out probabilities</p></div>') if abl_rows else ''}
  {img_block('ablation','Single modality vs late fusion (balanced accuracy)')}
  <div class="card full">{('<h3>Real-time streaming decode</h3><img src="data:image/png;base64,'+imgs['realtime']+'"/>') if 'realtime' in imgs else ''}</div>
  <div class="card full">{('<h3>Galea — multi-device ingestion, signal quality &amp; resting manifold</h3><img src="data:image/png;base64,'+imgs['galea']+'"/>') if 'galea' in imgs else ''}</div>
</div>
<p style="color:#6e82a6;font-size:12px;margin-top:24px">DVXR Lab · Goal 1 (BCI/EEG ingestion → embeddings → real-time decoding → explainable biomarkers). Omics deferred. Leakage-controlled = windows from the same command trial never split across train/test.</p>
</div></body></html>"""
    (OUT / "dashboard.html").write_text(html)


if __name__ == "__main__":
    main()
