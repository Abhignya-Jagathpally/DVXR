"""Honest test: does full-rate (128 Hz) DEAP band-power beat the decimated chance result?

The canonical DVXR pipeline decimates DEAP to ~8 Hz, which aliases away the alpha/beta
oscillations that carry affect — every canonical config sits at chance (AUROC ~0.53). This
probe computes proper band-power on the ORIGINAL 128 Hz preprocessed signal (which retains
the full spectrum) and classifies valence/arousal under SUBJECT-HELD-OUT CV. If it clears
chance, decimation was the culprit and a full-rate re-export is warranted; if it stays at
chance, DEAP is confirmed fundamentally limited for this task.

No leakage: features are per-trial spectra; folds are grouped by subject (a subject's trials
never span train and test). Reports mean AUROC over folds — reported honestly either way.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
FS = 128.0
N_EEG = 32  # first 32 channels are EEG in DEAP preprocessed


def _bandpower_features(trials: np.ndarray) -> np.ndarray:
    """trials: (n_trials, 32, samples) -> (n_trials, 32*len(BANDS)) relative band power."""
    from scipy.signal import welch

    freqs, psd = welch(trials[:, :N_EEG, :], fs=FS, nperseg=int(FS * 2), axis=-1)
    total = psd.sum(axis=-1, keepdims=True) + 1e-12
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(psd[:, :, mask].sum(axis=-1) / total[:, :, 0])
    # stack -> (n_trials, 32, n_bands) -> flatten channels x bands
    return np.concatenate([f[:, :, None] for f in feats], axis=-1).reshape(len(trials), -1)


def _load_subject(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    signal = np.asarray(data["data"], dtype=float)  # (40, 40, 8064)
    labels = np.asarray(data["labels"], dtype=float)  # (40, 4)
    feats = _bandpower_features(signal)
    valence = (labels[:, 0] > 5.0).astype(int)
    arousal = (labels[:, 1] > 5.0).astype(int)
    return feats, valence, arousal


def _evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    results = {}
    for name, make in {
        "logistic": lambda: LogisticRegression(max_iter=1000, C=0.5),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05),
    }.items():
        aurocs = []
        gkf = GroupKFold(n_splits=5)
        for train_idx, test_idx in gkf.split(x, y, groups):
            if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
                continue
            scaler = StandardScaler().fit(x[train_idx])
            xt, xv = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
            model = make()
            model.fit(xt, y[train_idx])
            score = (model.predict_proba(xv)[:, 1] if hasattr(model, "predict_proba")
                     else model.predict(xv))
            aurocs.append(roc_auc_score(y[test_idx], score))
        results[name] = float(np.mean(aurocs)) if aurocs else float("nan")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/real/deap/data_preprocessed_python"))
    parser.add_argument("--max-subjects", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("outputs/_r2/deap_fullrate_probe.md"))
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("s*.dat"))[: args.max_subjects]
    all_feats, all_val, all_aro, groups = [], [], [], []
    for i, path in enumerate(files):
        feats, valence, arousal = _load_subject(path)
        all_feats.append(feats)
        all_val.append(valence)
        all_aro.append(arousal)
        groups.append(np.full(len(feats), i))
        print(f"  {path.stem}: {feats.shape[0]} trials, {feats.shape[1]} features")
    x = np.concatenate(all_feats)
    groups = np.concatenate(groups)
    valence = _evaluate(x, np.concatenate(all_val), groups)
    arousal = _evaluate(x, np.concatenate(all_aro), groups)

    lines = [
        "# DEAP full-rate (128 Hz) band-power probe — subject-held-out\n",
        f"Subjects: {len(files)} | features: {x.shape[1]} (32 EEG ch x 4 bands, relative power) | "
        "5-fold GroupKFold by subject. Chance = 0.50; the decimated canonical pipeline sits ~0.53.\n",
        "| target | logistic AUROC | gradient-boosting AUROC |",
        "|---|---:|---:|",
        f"| valence (high vs low) | {valence['logistic']:.3f} | {valence['gradient_boosting']:.3f} |",
        f"| arousal (high vs low) | {arousal['logistic']:.3f} | {arousal['gradient_boosting']:.3f} |",
    ]
    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
