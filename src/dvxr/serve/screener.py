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


def _fit_personalization(subjects, oof_prob, y, covered, seed: int = 7):
    """Fit a PersonalizedCalibrator on OOF probs and honestly measure its ECE gain.

    Per-subject recalibration needs BOTH classes within a subject, so it applies to within-subject
    STATE tasks (stress/workload) — not subject-level-diagnosis tasks (depression: one class per
    subject → no per-subject model fits, falls back to population, a no-op). The reported ECE is on a
    per-subject held-out split (calibrate on half a subject's windows, evaluate on the other half),
    so the gain is honest, not fit-on-what-you-test. Returns (calibrator_or_None, metrics_dict).
    """
    from dvxr.calibration import expected_calibration_error
    from dvxr.personalization import PersonalizedCalibrator

    m = np.asarray(covered) & np.isfinite(oof_prob)
    subs, p, yy = np.asarray(subjects)[m].astype(str), np.asarray(oof_prob)[m], np.asarray(y)[m]
    # a subject is personalizable only if it carries both classes
    per_subj_both = [s for s in np.unique(subs) if len(np.unique(yy[subs == s])) >= 2]
    if not per_subj_both:
        return None, {"applicable": False,
                      "note": "subject-level-label task (one class per subject) — per-subject "
                              "recalibration does not apply; population calibration used"}
    rng = np.random.default_rng(seed)
    tr_mask = np.zeros(len(p), dtype=bool)
    for s in np.unique(subs):
        idx = np.where(subs == s)[0]
        rng.shuffle(idx)
        tr_mask[idx[: max(1, len(idx) // 2)]] = True   # first half (shuffled) calibrates
    te_mask = ~tr_mask
    pop_ece = float(expected_calibration_error(yy[te_mask], p[te_mask]))
    cal = PersonalizedCalibrator()
    cal.fit(subs[tr_mask], p[tr_mask], yy[tr_mask])
    pers_ece = float(expected_calibration_error(yy[te_mask], cal.predict(subs[te_mask], p[te_mask])))
    # the DEPLOYED calibrator is fit on ALL OOF data (more per-subject signal at serve time)
    deployed = PersonalizedCalibrator()
    deployed.fit(subs, p, yy)
    metrics = {"applicable": True, "population_ece": round(pop_ece, 4),
               "personalized_ece": round(pers_ece, 4),
               "ece_improvement": round(pop_ece - pers_ece, 4),
               "n_personalized_subjects": len(per_subj_both),
               "note": "per-subject held-out split (calibrate on half a subject's windows, "
                       "evaluate on the other half) — honest gain"}
    return deployed, metrics


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


def _subject_level_auroc(subjects, oof_prob, y, covered, seed: int = 7, n_boot: int = 1000):
    """Subject-level AUROC from out-of-fold window probabilities — only for subject-level-label tasks.

    Window-level AUROC pools every held-out window, so correlated windows from the same subject
    inflate it relative to the clinically meaningful question ("is THIS SUBJECT a case?"). For a
    subject-level-label task (e.g. depression: every window of a subject shares the diagnosis) this
    aggregates each subject's OOF windows to one probability (the same mean `score_subject` uses) and
    scores AUROC over subjects — the conservative number, with a subject-resampling bootstrap CI.

    For a WITHIN-subject state task (workload rest-vs-task, stress rest-vs-stress: a subject carries
    BOTH classes), the label is per-window state, not a per-subject diagnosis, so subject-level AUROC
    is undefined and the epoch/window-level number is the appropriate unit — we return that note.
    Returns (auroc, ci_lo, ci_hi, n_subjects_scored, note).
    """
    from sklearn.metrics import roc_auc_score
    m = np.asarray(covered) & np.isfinite(oof_prob)
    subs, p, yy = np.asarray(subjects)[m], np.asarray(oof_prob)[m], np.asarray(y)[m]
    uniq = np.unique(subs)
    single_class = all(len(np.unique(yy[subs == s])) == 1 for s in uniq)
    if not single_class:
        return None, None, None, int(len(uniq)), (
            "within-subject state task (a subject carries both classes); the epoch/window-level "
            "AUROC is the appropriate unit — subject-level AUROC does not apply")
    sp = np.array([p[subs == s].mean() for s in uniq])
    sy = np.array([int(round(float(yy[subs == s].mean()))) for s in uniq])
    if len(np.unique(sy)) < 2:
        return None, None, None, int(len(uniq)), "only one subject-level class present"
    point = float(roc_auc_score(sy, sp))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(uniq))
    boots = [roc_auc_score(sy[b], sp[b]) for b in
             (rng.choice(idx, size=len(idx), replace=True) for _ in range(n_boot))
             if len(np.unique(sy[b])) >= 2]
    lo, hi = ((float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))
              if boots else (point, point))
    return point, lo, hi, int(len(uniq)), "subject-level (one prediction per subject)"


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
    personal: object = None              # optional PersonalizedCalibrator (within-subject tasks)

    # ---- inference ----
    def predict_windows(self, emb: np.ndarray, subject_id=None) -> np.ndarray:
        """Calibrated positive-class probability per window/embedding row.

        If a ``subject_id`` is given AND this screener carries a per-subject calibrator, the
        subject's own recalibrator is applied (unseen subjects fall back to the global one);
        otherwise the population calibrator is used."""
        emb = np.asarray(emb, dtype=float)
        p = _head_proba(self.scaler, self.head, emb)
        if subject_id is not None and self.personal is not None:
            return np.asarray(self.personal.predict([str(subject_id)] * len(p), p), dtype=float)
        return np.asarray(self.calibrator.predict(p), dtype=float)

    def score_subject(self, emb: np.ndarray, subject_id=None) -> dict:
        """Aggregate a subject's window embeddings into one screening result."""
        from dvxr.calibration import risk_band
        probs = self.predict_windows(emb, subject_id=subject_id)
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
            "heldout_auroc_subject": self.heldout.get("auroc_subject"),
            "heldout_auroc_subject_ci": self.heldout.get("auroc_subject_ci"),
            "basis": self.meta.get("encoder", self.representation),
            "caveat": self.meta.get("caveat", ""),
        }

    # ---- persistence ----
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {"scaler": self.scaler, "head": self.head, "calibrator": self.calibrator}
        if self.personal is not None:
            blob["personal"] = self.personal
        joblib.dump(blob, path / "model.joblib")
        manifest = {"task": self.task, "representation": self.representation,
                    "conformal": self.conformal, "heldout": self.heldout, "meta": self.meta,
                    "personalized": self.personal is not None,
                    "format": "dvxr-screener/2" if self.personal is not None else "dvxr-screener/1"}
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
                   conformal=float(m["conformal"]), heldout=m["heldout"], meta=m["meta"],
                   personal=blob.get("personal"))  # v1 artifacts have no 'personal' → None


def fit_screener(task_name: str, n_repeats: int = 3, n_folds: int = 5,
                 seed: int = 7, representation: Optional[str] = None,
                 personalize: bool = False) -> Screener:
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

    # subject-level AUROC (the conservative, clinically meaningful number) alongside window-level
    auroc_subj, subj_lo, subj_hi, n_subj_scored, subj_note = _subject_level_auroc(
        subjects, oof_prob, y, covered, seed=seed)

    # calibrate on out-of-fold predictions (honest, uses no test leakage into the head fit)
    cov = covered & np.isfinite(oof_prob)
    calibrator = fit_platt_calibrator(oof_prob[cov], y[cov])
    cal_oof = calibrator.predict(oof_prob[cov])
    ece = float(expected_calibration_error(y[cov], cal_oof))
    conformal = float(conformal_radius(np.abs(y[cov] - cal_oof), alpha=0.10))

    # optional serve-time personalization (within-subject tasks only) + honest ECE gain
    personal, personal_metrics = (None, {"applicable": False, "enabled": False})
    if personalize:
        personal, personal_metrics = _fit_personalization(subjects, oof_prob, y, covered, seed=seed)
        personal_metrics["enabled"] = True

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
        heldout={"metric": "AUROC", "auroc": round(auroc, 4), "auroc_level": "window",
                 "auroc_ci": [round(ci[0], 4), round(ci[1], 4)],
                 "auroc_subject": (round(auroc_subj, 4) if auroc_subj is not None else None),
                 "auroc_subject_ci": ([round(subj_lo, 4), round(subj_hi, 4)]
                                      if auroc_subj is not None else None),
                 "auroc_subject_note": subj_note, "n_subjects_scored": n_subj_scored,
                 "ece": round(ece, 4), "personalization": personal_metrics,
                 "n_subjects": int(len(np.unique(subjects))), "n_windows": int(len(y)),
                 "n_folds": int(len(fold_auroc)), "protocol": f"{n_repeats}x{n_folds} subject-held-out CV"},
        meta={"label": TASK_LABEL.get(task_name, task_name), "encoder": encoder,
              "caveat": caveat, "band_thresholds": {"low": 0.25, "watch": 0.50, "elevated": 0.75},
              "literature": _LITERATURE.get(representation, [])},
        personal=personal)
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
