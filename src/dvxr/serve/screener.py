"""dvxr.serve.screener — a trained, persistable clinical-risk screener.

`fit_screener(task)` trains a calibrated head on top of the VALIDATED representation for that
task — the real LaBraM EEG foundation model for EEG screening (depression, workload), or the
band-power features for wearable stress — under subject-held-out CV, and records the held-out
AUROC + bootstrap CI so the product's reported accuracy is the *same* number as the committed
benchmark. `Screener.save/load` persist a portable artifact; `predict_windows` /
`score_subject` turn a new subject's signals into a calibrated probability, a risk band, and a
conformal interval.

Honesty: this is research-grade screening, not diagnosis. The held-out AUROC a Screener carries
is its own estimate on subject-disjoint folds; on depression it reproduces the ~0.96 that the
LaBraM benchmark reports (dataset-specific, fidelity-limited — see docs/MODEL_CARD.md).
Deterministic, offline/CPU (LaBraM weights cached or DVXR_LABRAM_ALLOW_DOWNLOAD).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Which validated representation each task's screener is built on.
REPRESENTATION_BY_TASK: Dict[str, str] = {
    "mumtaz_depression": "labram_eeg",      # real LaBraM EEG FM — the headline (AUROC ~0.96)
    "eegmat_workload": "labram_eeg",         # LaBraM EEG (ECG is stronger; EEG-screener view)
    "wesad_stress": "bandpower_concat",      # wearable physiology band-power (AUROC ~0.95)
    "stress": "bandpower_concat",            # peripheral physiology (AUROC ~0.89)
}

# Short, honest per-task label for reports (never a diagnostic claim).
TASK_LABEL: Dict[str, str] = {
    "mumtaz_depression": "Depression screen (MDD vs healthy) from resting EEG",
    "eegmat_workload": "Cognitive-workload screen (rest vs task) from EEG",
    "wesad_stress": "Acute-stress screen from wearable physiology",
    "stress": "Stress screen from peripheral physiology",
}


def _embed_task(task, representation: str) -> np.ndarray:
    """(N, d) representation for every window of a BenchTask, using the validated encoder."""
    if representation == "labram_eeg":
        from dvxr.bench.labram_bench import labram_embeddings
        return np.asarray(labram_embeddings(task), dtype=float)
    if representation == "bandpower_concat":
        from dvxr.bench.representations import _concat
        return np.asarray(_concat(task), dtype=float)
    raise ValueError(f"unknown representation {representation!r}")


def embed_cohort(task_name: str, representation: Optional[str] = None):
    """Build a cohort task and return (emb, y, subject_ids, task) using its validated encoder.

    Shared by fit_screener, the CLI, and the demo so a subject is embedded exactly the way the
    benchmark scores it. Embeddings are per window; ``subject_ids`` aligns row-for-row so a single
    subject's windows are ``emb[subject_ids == sid]``.
    """
    from dvxr.bench.tasks import TASK_BUILDERS
    representation = representation or REPRESENTATION_BY_TASK.get(task_name, "bandpower_concat")
    task = TASK_BUILDERS[task_name]()
    emb = _embed_task(task, representation)
    y = np.asarray(task.y, dtype=int)
    subjects = np.asarray(task.subject_ids)
    return emb, y, subjects, task


def _fit_head(emb: np.ndarray, y: np.ndarray, seed: int):
    """StandardScaler + balanced logistic head (the same shared head the benchmark scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(emb)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed).fit(sc.transform(emb), y)
    return sc, clf


def _head_proba(sc, clf, emb: np.ndarray) -> np.ndarray:
    classes = list(clf.classes_)
    if len(classes) < 2:
        return np.full(len(emb), float(classes[0]))
    pos = classes.index(1) if 1 in classes else 1
    return clf.predict_proba(sc.transform(emb))[:, pos]


@dataclass
class Screener:
    """A trained, persistable screener wired to a validated representation + calibrated head."""
    task: str
    representation: str
    scaler: object                       # sklearn StandardScaler
    head: object                         # sklearn LogisticRegression
    calibrator: object                   # dvxr.calibration.BinaryCalibrator
    conformal: float                     # +/- radius on the calibrated probability
    heldout: dict                        # auroc, auroc_ci, ece, n_subjects, n_windows, metric
    meta: dict = field(default_factory=dict)   # encoder id, label, caveats, literature, thresholds

    # ---- inference ----
    def predict_windows(self, emb: np.ndarray) -> np.ndarray:
        """Calibrated positive-class probability per window/embedding row."""
        from dvxr.calibration import BinaryCalibrator  # noqa: F401 (type ref)
        p = _head_proba(self.scaler, self.head, np.asarray(emb, dtype=float))
        return np.asarray(self.calibrator.predict(p), dtype=float)

    def score_subject(self, emb: np.ndarray) -> dict:
        """Aggregate a subject's window embeddings into one screening result."""
        from dvxr.calibration import risk_band
        probs = self.predict_windows(emb)
        p = float(np.mean(probs))
        lo = float(max(0.0, p - self.conformal))
        hi = float(min(1.0, p + self.conformal))
        return {
            "task": self.task,
            "label": self.meta.get("label", self.task),
            "probability": round(p, 4),
            "risk_band": risk_band(p),
            "interval": [round(lo, 4), round(hi, 4)],
            "n_windows": int(len(probs)),
            "heldout_auroc": self.heldout.get("auroc"),
            "heldout_auroc_ci": self.heldout.get("auroc_ci"),
            "basis": self.meta.get("encoder", self.representation),
            "caveat": self.meta.get("caveat", ""),
        }

    # ---- persistence ----
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "head": self.head,
                     "calibrator": self.calibrator}, path / "model.joblib")
        manifest = {"task": self.task, "representation": self.representation,
                    "conformal": self.conformal, "heldout": self.heldout, "meta": self.meta,
                    "format": "dvxr-screener/1"}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Screener":
        import joblib
        path = Path(path)
        m = json.loads((path / "manifest.json").read_text())
        blob = joblib.load(path / "model.joblib")
        return cls(task=m["task"], representation=m["representation"],
                   scaler=blob["scaler"], head=blob["head"], calibrator=blob["calibrator"],
                   conformal=float(m["conformal"]), heldout=m["heldout"], meta=m["meta"])


def fit_screener(task_name: str, n_repeats: int = 3, n_folds: int = 5,
                 seed: int = 7, representation: Optional[str] = None) -> Screener:
    """Train + calibrate a Screener on a cohort, capturing its subject-held-out AUROC.

    The held-out CV both (a) yields the honest accuracy the product reports and (b) produces
    out-of-fold probabilities used to fit the calibrator and the conformal radius. The final
    deployable head is then refit on ALL windows.
    """
    from sklearn.metrics import roc_auc_score

    from dvxr.bench.protocol import bootstrap_ci, repeated_group_folds
    from dvxr.calibration import (conformal_radius, expected_calibration_error,
                                  fit_platt_calibrator, risk_band)

    representation = representation or REPRESENTATION_BY_TASK.get(task_name, "bandpower_concat")
    emb, y, subjects, task = embed_cohort(task_name, representation)

    folds = repeated_group_folds(subjects, n_repeats, n_folds, seed)
    fold_auroc: List[float] = []
    oof_prob = np.full(len(y), np.nan)
    oof_cnt = np.zeros(len(y))
    oof_sum = np.zeros(len(y))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        sc, clf = _fit_head(emb[tr], y[tr], seed)
        p = _head_proba(sc, clf, emb[te])
        oof_sum[te] += p
        oof_cnt[te] += 1
        if len(np.unique(y[te])) >= 2:
            fold_auroc.append(float(roc_auc_score(y[te], p)))
    covered = oof_cnt > 0
    oof_prob[covered] = oof_sum[covered] / oof_cnt[covered]

    auroc = float(np.mean(fold_auroc)) if fold_auroc else float("nan")
    ci = bootstrap_ci(fold_auroc, seed=seed) if len(fold_auroc) >= 2 else (auroc, auroc)

    # calibrate on out-of-fold predictions (honest, uses no test leakage into the head fit)
    cov = covered & np.isfinite(oof_prob)
    calibrator = fit_platt_calibrator(oof_prob[cov], y[cov])
    cal_oof = calibrator.predict(oof_prob[cov])
    ece = float(expected_calibration_error(y[cov], cal_oof))
    conformal = float(conformal_radius(np.abs(y[cov] - cal_oof), alpha=0.10))

    # final deployable head on ALL windows
    scaler, head = _fit_head(emb, y, seed)

    encoder = ("real LaBraM EEG foundation model (frozen, vendored forward)"
               if representation == "labram_eeg" else "band-power physiology features")
    caveat = ("Research-grade screening, not a diagnosis. Held-out AUROC is a subject-disjoint "
              "estimate on a research cohort" +
              (" (dataset-specific; EEG sampled at 64 Hz, below LaBraM's 200 Hz training)."
               if representation == "labram_eeg" else "."))
    screener = Screener(
        task=task_name, representation=representation, scaler=scaler, head=head,
        calibrator=calibrator, conformal=conformal,
        heldout={"metric": "AUROC", "auroc": round(auroc, 4),
                 "auroc_ci": [round(ci[0], 4), round(ci[1], 4)], "ece": round(ece, 4),
                 "n_subjects": int(len(np.unique(subjects))), "n_windows": int(len(y)),
                 "n_folds": int(len(fold_auroc)), "protocol": f"{n_repeats}x{n_folds} subject-held-out CV"},
        meta={"label": TASK_LABEL.get(task_name, task_name), "encoder": encoder,
              "caveat": caveat, "band_thresholds": {"low": 0.25, "watch": 0.50, "elevated": 0.75},
              "literature": _LITERATURE.get(representation, [])})
    return screener


_LITERATURE = {
    "labram_eeg": [
        "LaBraM — Jiang et al., ICLR 2024, arXiv:2405.18765 (EEG foundation model; real weights "
        "braindecode/labram-pretrained)",
        "Mumtaz et al., 2016 — MDD-vs-healthy resting-EEG cohort (labels)",
    ],
    "bandpower_concat": [
        "WESAD — Schmidt et al., 2018 (wearable stress cohort)",
        "PhysioNet Non-EEG — Birjandtalab et al., 2016 (peripheral physiology, stress states)",
    ],
}
