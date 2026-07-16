#!/usr/bin/env python3
"""run_clinical_notes_bench.py — the honest floor-vs-FM benchmark for the REAL
unstructured clinical-notes path (POW Goal 2).

Corpus: MTSamples (real de-identified transcribed medical reports, public domain).
Two tasks — binary (Surgery vs rest) and 40-way specialty. Each is evaluated under
note-held-out grouped CV comparing:

  * majority           — no-skill floor
  * tfidf+lr           — classical bag-of-words floor (TF-IDF fit per-fold on TRAIN)
  * hashing+gbm        — the harness's stateless single:ehr_notes floor
  * clinicalbert+lr    — a FROZEN real clinical transformer (Bio_ClinicalBERT, chunk-
                         pooled CLS over >512-token notes) as a feature extractor

The clinical-transformer embedding is label-free and computed once over all rows (no
transductive leak); every supervised head (LR/GBM) and the per-fold TF-IDF vocabulary
are fit on TRAIN indices only. We do NOT assume the transformer wins — the measured
relativity is reported as-is. Writes outputs/clinical_notes_scoreboard.{md,csv}.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.bench.tasks import (  # noqa: E402
    clinical_notes_specialty_task, clinical_notes_surgery_task)
from dvxr.encoders.base import clinical_notes_available  # noqa: E402


def _clinicalbert_embeddings(texts):
    """Frozen Bio_ClinicalBERT embeddings for every note (label-free, computed once)."""
    import pandas as pd
    from dvxr.config import DEFAULTS
    from dvxr.encoders.base import make_primary_backend
    cfg = DEFAULTS.with_(d=16, use_real_weights=True, allow_download=True, seed=7)
    backend = make_primary_backend("ehr_notes", cfg)
    if backend is None or not hasattr(backend, "_embed"):
        return None, ""
    frame = pd.DataFrame({"note_text": texts})
    emb = np.asarray(backend._embed(frame, ["note_text"]), dtype=float)
    return emb, backend.name


def _binary_metrics(y_true, prob):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan")}
    return {"auroc": float(roc_auc_score(y_true, prob))}


def _multiclass_metrics(y_true, proba, classes):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true)
    pred = classes[np.argmax(proba, axis=1)]
    out = {"macro_f1": float(f1_score(y_true, pred, average="macro")),
           "accuracy": float(accuracy_score(y_true, pred))}
    try:  # macro AUROC needs every class present in y_true
        if set(np.unique(y_true)) == set(classes):
            out["macro_auroc"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=classes))
        else:
            out["macro_auroc"] = float("nan")
    except Exception:
        out["macro_auroc"] = float("nan")
    return out


def _lr(seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)


def _eval_task(task, emb, kind, n_folds=5, seed=7):
    """Grouped CV over notes; return {config: {metric: mean}} and the fold table."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = task.y
    texts = task.extra["notes_text"]
    Xhash = task.features["ehr_notes"]
    classes = np.unique(y)
    metric_fn = _binary_metrics if kind == "binary" else \
        (lambda yt, p: _multiclass_metrics(yt, p, classes))

    def _proba(clf, Xtr, ytr, Xte):
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)
        if kind == "binary":
            pos = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            return p[:, pos]
        # align columns to the global class order
        col = {c: i for i, c in enumerate(clf.classes_)}
        return np.column_stack([p[:, col[c]] if c in col else np.zeros(len(Xte))
                                for c in classes])

    folds = list(GroupKFold(n_splits=n_folds).split(y, y, task.subject_ids))
    rows = {c: [] for c in (["majority", "tfidf+lr", "hashing+gbm"]
                            + (["clinicalbert+lr"] if emb is not None else []))}
    for tr, te in folds:
        # majority
        if kind == "binary":
            base = np.full(len(te), float(np.mean(y[tr])))
            rows["majority"].append(metric_fn(y[te], base))
        else:
            maj = np.bincount(
                [list(classes).index(v) for v in y[tr]], minlength=len(classes))
            proba = np.tile(maj / maj.sum(), (len(te), 1))
            rows["majority"].append(metric_fn(y[te], proba))
        # tfidf + lr (vocabulary fit on TRAIN only)
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              sublinear_tf=True, stop_words="english")
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        rows["tfidf+lr"].append(metric_fn(y[te], _proba(_lr(seed), Xtr, y[tr], Xte)))
        # hashing + gbm (the harness single:ehr_notes floor)
        gbm = HistGradientBoostingClassifier(random_state=seed)
        rows["hashing+gbm"].append(metric_fn(y[te], _proba(gbm, Xhash[tr], y[tr], Xhash[te])))
        # clinicalbert + lr (frozen FM features, standardized)
        if emb is not None:
            sc = StandardScaler().fit(emb[tr])
            e_tr, e_te = sc.transform(emb[tr]), sc.transform(emb[te])
            rows["clinicalbert+lr"].append(metric_fn(y[te], _proba(_lr(seed), e_tr, y[tr], e_te)))

    summary = {}
    for cfg, fold_metrics in rows.items():
        keys = fold_metrics[0].keys()
        summary[cfg] = {k: float(np.nanmean([fm[k] for fm in fold_metrics])) for k in keys}
    return summary


def _fmt(summary, primary):
    order = sorted(summary, key=lambda c: -summary[c].get(primary, float("-inf")))
    metrics = list(next(iter(summary.values())).keys())
    header = "| config | " + " | ".join(metrics) + " |"
    sep = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    lines = [header, sep]
    for c in order:
        lines.append("| " + c + " | " + " | ".join(f"{summary[c][m]:.4f}" for m in metrics) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="clinical-notes floor-vs-FM benchmark")
    ap.add_argument("--max-notes", type=int, default=None,
                    help="cap corpus size (default: full 4,499)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    print(f"[clinical-notes] clinical_notes_available={clinical_notes_available()}")
    surgery = clinical_notes_surgery_task(max_notes=args.max_notes)
    specialty = clinical_notes_specialty_task(max_notes=args.max_notes)
    prov = surgery.extra.get("provenance", "")
    print(f"[clinical-notes] n={surgery.n} notes; provenance: {prov}")

    emb, backend_name = _clinicalbert_embeddings(surgery.extra["notes_text"])
    if emb is None:
        print("[clinical-notes] real clinical transformer unavailable — floor-only run")
    else:
        print(f"[clinical-notes] frozen FM: {backend_name}  emb={emb.shape}")

    surg_sum = _eval_task(surgery, emb, "binary", n_folds=args.folds)
    spec_sum = _eval_task(specialty, emb, "multiclass", n_folds=args.folds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Clinical Notes Scoreboard (real unstructured EHR text)",
        "",
        f"- Corpus: {prov}",
        f"- Notes: **{surgery.n}**   |   grouped CV folds: {args.folds}   |   each note = its own CV group",
        f"- Frozen clinical transformer: `{backend_name or 'UNAVAILABLE'}`",
        "- Honest relativity: the transformer is NOT assumed to beat the classical floors; "
        "numbers are measured. Label-free FM embedding computed once (no leak); heads + "
        "TF-IDF vocabulary fit on TRAIN folds only.",
        "",
        "## Task 1 — Surgery vs rest (binary, AUROC ↑)",
        "",
        _fmt(surg_sum, "auroc"),
        "",
        "## Task 2 — 40-way specialty (multi-class, macro-F1 ↑)",
        "",
        _fmt(spec_sum, "macro_f1"),
        "",
    ]
    md_path = out_dir / "clinical_notes_scoreboard.md"
    md_path.write_text("\n".join(md))

    import csv
    csv_path = out_dir / "clinical_notes_scoreboard.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "config", "metric", "value"])
        for task_name, summ in [("surgery", surg_sum), ("specialty", spec_sum)]:
            for cfg, metrics in summ.items():
                for m, v in metrics.items():
                    w.writerow([task_name, cfg, m, f"{v:.6f}"])

    print("\n".join(md))
    print(f"\n[clinical-notes] wrote {md_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
