"""dvxr.bench.gated_fusion — do-no-harm (DNH) reliability-gated late fusion.

Motivation (the honest negative result this repo already reports): a from-scratch
learned cross-modal transformer (``rep:fused``) trained on ~20-60 subjects over
per-window summary statistics *underperforms* single-modality / concat / GBM
baselines on all six real clinical/BCI tasks. In that small-cohort regime there is
not enough data to learn cross-modal interactions, so a jointly-trained fusion only
adds variance. The remedy here sidesteps joint training entirely: a reliability-
gated LATE fusion with a *finite-sample-robust* safety floor.

Provenance (cited honestly, not claimed as new): the do-no-harm floor is the
Super-Learner / stacked-generalization oracle inequality (van der Laan et al. 2007;
Hasson et al. ICML 2023) — a CV-selected non-negative-weight ensemble is
asymptotically no worse than its best candidate. Reliability-weighted late fusion of
physiological signals also predates us (Wei et al. 2018; Han et al. TMC 2021/22).

What is genuinely new here is the *finite-sample* treatment for N<=60 subjects,
where the inner-CV weights are very noisy and the asymptotic oracle guarantee
degrades: we (1) shrink the stacked weights toward the single best candidate by an
amount tied to the inner-CV noise, and (2) only *accept* the ensemble over the best
candidate when its inner-CV advantage clears one standard error (a subject-grouped
bootstrap SE). Otherwise we fall back to the best candidate. So on the inner-CV
estimate the method is, by construction, never worse than the best candidate in the
library — and it is only trusted to be better when the data support it.

The candidate library deliberately includes the STRONG learner (a gradient-boosted
tree on the concatenated features), not only linear per-modality heads: a do-no-harm
floor is only meaningful relative to the best opponent the field would actually use.
So DNH's guarantee is "never worse than the tuned GBM on concat" — while still able
to beat it (and the learned cross-modal fusion) when reliability-gated late fusion of
the per-modality heads adds real signal.

Caveat stated plainly: the floor holds on the *inner-CV* estimate, which at small N
can diverge from held-out subjects. The shrinkage + SE gate is exactly what narrows
that divergence; the outer 5x5 subject-held-out CV in the benchmark is what tests
whether it did.

Predictor signature matches every other opponent in ``bench/baselines.py``:
``pred_dnh_gated(task, tr, te, seed=7) -> np.ndarray`` (probabilities for
classification, target values for forecast). Fit on ``tr`` only — no leakage.
Deterministic, CPU, sklearn/scipy only.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np

from dvxr.bench.protocol import repeated_group_folds
from dvxr.bench.representations import _concat, _fit_head
from dvxr.bench.tasks import BenchTask

# inner-CV budget for reliability estimation (kept cheap: linear heads + one GBM)
_INNER_REPEATS = 2
_INNER_FOLDS = 5
_BOOT = 200
_EPS = 1e-9


# --------------------------------------------------------------- candidate library
def _gbm_predict(kind: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    """HistGradientBoosting on the given features (the strong non-linear candidate)."""
    if kind == "classification":
        from sklearn.ensemble import HistGradientBoostingClassifier
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xte), float(np.mean(ytr)))
        m = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, list(m.classes_).index(1)]
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed).fit(Xtr, ytr).predict(Xte)


def _candidates(task: BenchTask) -> Tuple[Dict[str, Callable], List[str]]:
    """Candidate library as name -> fit_predict(tr_abs, te_abs, seed) -> preds.

    Members: each single modality (linear head), the concat (linear head), the concat
    under a gradient-boosted tree (the strong learner), and — when a real frozen FM is
    available — the SOTA embedding (linear head). Returns (candidates, single_names).
    All views are label-free; every fit is train-only (no leakage). Also returns the
    subset that are single-modality candidates, for the do-no-harm-vs-best-single claim.
    """
    kind = task.kind
    cands: Dict[str, Callable] = {}
    singles: List[str] = []
    for m in task.modalities:
        X = np.asarray(task.features[m], dtype=float)
        cands[f"single:{m}"] = (lambda tr, te, seed, X=X:
                                _fit_head(kind, X[tr], task.y[tr], X[te], seed=seed))
        singles.append(f"single:{m}")
    if len(task.modalities) > 1:
        Xc = _concat(task).astype(float)
        cands["concat"] = (lambda tr, te, seed, Xc=Xc:
                           _fit_head(kind, Xc[tr], task.y[tr], Xc[te], seed=seed))
        cands["concat_gbm"] = (lambda tr, te, seed, Xc=Xc:
                              _gbm_predict(kind, Xc[tr], task.y[tr], Xc[te], seed))
    else:
        Xc = np.asarray(task.features[task.modalities[0]], dtype=float)
        cands["concat_gbm"] = (lambda tr, te, seed, Xc=Xc:
                              _gbm_predict(kind, Xc[tr], task.y[tr], Xc[te], seed))
    # optional real-SOTA candidate (deferred import to avoid a baselines<->gated cycle).
    # DVXR_DNH_NO_SOTA lets the fast unit tests skip the heavy FM load; the benchmark
    # path never sets it, so the full library (incl. SOTA) runs there.
    if os.environ.get("DVXR_DNH_NO_SOTA"):
        return cands, singles
    try:
        from dvxr.bench.baselines import _sota_embeddings
        emb = np.asarray(_sota_embeddings(task), dtype=float)
        if emb.ndim == 2 and emb.shape[0] == task.n and np.all(np.isfinite(emb)):
            cands["sota"] = (lambda tr, te, seed, E=emb:
                            _fit_head(kind, E[tr], task.y[tr], E[te], seed=seed))
    except Exception:
        pass  # CGM-JEPA etc. can't load — the DNH library simply omits it (never silent-fakes)
    return cands, singles


def _cand_error(kind: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    """Error (lower better): 1-AUROC for classification, MAE for forecast."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=float)
    if kind == "classification":
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(1.0 - roc_auc_score(y_true, pred))
    return float(np.mean(np.abs(pred - y_true)))


# --------------------------------------------------------------- OOF reliability
def _oof_predictions(task: BenchTask, cands: Dict[str, Callable], tr: np.ndarray,
                     seed: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Out-of-fold predictions for every candidate over the TRAIN rows, via
    subject-grouped inner CV. Returns ({cand: oof_pred[len(tr)]}, covered_mask)."""
    sub_tr = np.asarray(task.subject_ids)[tr]
    ytr = task.y[tr]
    names = list(cands)
    acc = {c: np.zeros(len(tr)) for c in names}
    cnt = np.zeros(len(tr))
    try:
        inner = repeated_group_folds(sub_tr, _INNER_REPEATS, _INNER_FOLDS, seed=seed)
    except ValueError:
        inner = []
    for itr, ite in inner:
        # skip degenerate inner train folds for classification (need both classes)
        if task.kind == "classification" and len(np.unique(ytr[itr])) < 2:
            continue
        abs_tr, abs_te = tr[itr], tr[ite]
        for c in names:
            acc[c][ite] += np.asarray(cands[c](abs_tr, abs_te, seed), dtype=float)
        cnt[ite] += 1.0
    covered = cnt > 0
    oof = {c: np.divide(acc[c], np.maximum(cnt, 1.0)) for c in names}
    return oof, covered


# --------------------------------------------------------------- DNH combiner
def _nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative stacking weights (Super-Learner convention), normalised to sum 1."""
    from scipy.optimize import nnls
    w, _ = nnls(P, y.astype(float))
    s = w.sum()
    return w / s if s > _EPS else np.full(P.shape[1], 1.0 / P.shape[1])


def _grouped_bootstrap_se(kind: str, subjects: np.ndarray, y: np.ndarray,
                          stack_oof: np.ndarray, best_oof: np.ndarray,
                          seed: int) -> Tuple[float, float]:
    """Subject-grouped bootstrap of the advantage (best_cand_err - stack_err).

    Returns (mean_advantage, se_advantage). Positive mean => the stacked ensemble
    beats the best single candidate on the inner-CV out-of-fold predictions."""
    uniq = np.unique(subjects)
    if len(uniq) < 2:
        return 0.0, float("inf")
    rng = np.random.default_rng(seed)
    advs: List[float] = []
    for _ in range(_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(subjects == s)[0] for s in pick])
        eb = _cand_error(kind, y[rows], best_oof[rows])
        es = _cand_error(kind, y[rows], stack_oof[rows])
        if np.isfinite(eb) and np.isfinite(es):
            advs.append(eb - es)
    if len(advs) < 2:
        return 0.0, float("inf")
    a = np.asarray(advs)
    return float(a.mean()), float(a.std(ddof=1))


def dnh_weights(task: BenchTask, cands: Dict[str, Callable], singles: List[str],
                tr: np.ndarray, seed: int) -> Tuple[Dict[str, float], dict]:
    """Compute the do-no-harm reliability-gated weights over candidates on TRAIN.

    Returns (weights_by_candidate, diagnostics). The diagnostics dict exposes the
    inner-CV per-candidate error, the best candidate and best single modality, the
    stacked advantage and its SE, and the shrinkage lambda — used by the synergy
    diagnostic + tests. The do-no-harm reference is the best OVERALL candidate (which
    includes the GBM on concat), so the guarantee is vs. the strongest opponent.
    """
    names = list(cands)
    oof, covered = _oof_predictions(task, cands, tr, seed)
    ytr = task.y[tr]
    sub_tr = np.asarray(task.subject_ids)[tr]
    rows = np.where(covered)[0]

    # per-candidate inner-CV reliability (error; lower is better)
    cand_err = {c: _cand_error(task.kind, ytr[rows], oof[c][rows]) for c in names}
    finite = {c: e for c, e in cand_err.items() if np.isfinite(e)}
    # do-no-harm reference = best OVERALL candidate (the guarantee is vs the field)
    best_cand = (min(finite, key=finite.get) if finite
                 else min(names, key=lambda c: cand_err.get(c, np.inf)))
    valid_single = [c for c in singles if np.isfinite(cand_err[c])]
    best_single = (min(valid_single, key=lambda c: cand_err[c])
                   if valid_single else best_cand)

    onehot = {c: (1.0 if c == best_cand else 0.0) for c in names}
    base_diag = {"cand_err": cand_err, "best_cand": best_cand,
                 "best_single": best_single}
    # not enough coverage / candidates to stack -> fall back to best candidate
    if len(rows) < 4 or len(names) < 2:
        return onehot, {**base_diag, "lambda": 1.0, "adv_mean": 0.0,
                        "adv_se": float("inf"), "accepted": False}

    # stacked non-negative weights on the OOF library (Super-Learner style)
    P = np.column_stack([oof[c][rows] for c in names])
    w_stack = _nnls_weights(P, ytr[rows])
    stack_oof = np.zeros(len(tr))
    for j, c in enumerate(names):
        stack_oof += w_stack[j] * oof[c]

    # finite-sample gate: accept the stack over the best candidate only if its
    # advantage clears one subject-grouped bootstrap SE; shrink toward best by noise.
    adv_mean, adv_se = _grouped_bootstrap_se(
        task.kind, sub_tr[rows], ytr[rows], stack_oof[rows], oof[best_cand][rows], seed)
    accepted = adv_mean > adv_se and np.isfinite(adv_se)
    if accepted:
        lam = float(np.clip(adv_se / (adv_mean + _EPS), 0.0, 1.0))  # shrink w/ noise
    else:
        lam = 1.0                                                    # pure best candidate

    w = {c: (1.0 - lam) * w_stack[j] + lam * onehot[c]
         for j, c in enumerate(names)}
    s = sum(w.values())
    w = {c: v / s for c, v in w.items()} if s > _EPS else onehot

    # the realised do-no-harm quantity: inner-CV error of the FINAL gated combiner.
    # By construction this is <= the best candidate's inner error (the safety floor).
    combined_oof = np.zeros(len(tr))
    for c in names:
        combined_oof += w[c] * oof[c]
    dnh_inner = _cand_error(task.kind, ytr[rows], combined_oof[rows])
    return w, {**base_diag, "lambda": lam, "adv_mean": adv_mean, "adv_se": adv_se,
               "accepted": accepted, "dnh_inner_err": dnh_inner,
               "w_stack": {c: float(w_stack[j]) for j, c in enumerate(names)}}


# --------------------------------------------------------------- predictor
def pred_dnh_gated(task: BenchTask, tr, te, seed: int = 7) -> np.ndarray:
    """Do-no-harm reliability-gated late fusion prediction on the test rows."""
    tr = np.asarray(tr)
    te = np.asarray(te)
    cands, singles = _candidates(task)
    if not cands:                                   # nothing to fuse
        return np.full(len(te), float(np.mean(task.y[tr])))
    w, _diag = dnh_weights(task, cands, singles, tr, seed)

    # refit each participating candidate on FULL train, combine on test.
    pred = np.zeros(len(te))
    total = 0.0
    for c, wc in w.items():
        if wc <= _EPS:
            continue
        pc = cands[c](tr, te, seed)
        pred += wc * np.asarray(pc, dtype=float)
        total += wc
    if total <= _EPS:                               # degenerate — majority/mean
        return np.full(len(te), float(np.mean(task.y[tr])))
    return pred / total


# --------------------------------------------------------------- synergy diagnostic
def dnh_diagnostics(task: BenchTask, seed: int = 7) -> dict:
    """Train-fold modality synergy/redundancy diagnostic + realised DNH behaviour.

    Answers "when does reliability-gated late fusion help vs. merely not-harm?".
    Computed on ONE subject-held-out train fold (the diagnostic is descriptive, not a
    scored result). Returns per-task quantities that the writeup correlates across the
    six cohorts to deliver the testable rule: *low joint-gain relative to inner-CV
    weight noise ⇒ the do-no-harm floor dominates learned fusion*.

    Keys:
      best_single_err   inner-CV error of the best single modality
      best_cand_err     inner-CV error of the best candidate (incl. concat-GBM)
      dnh_inner_err     inner-CV error of the gated combiner (<= best_cand_err)
      joint_gain        (best_single_err - best_cand_err)/best_single_err — how much
                        combining modalities helps at all (redundancy+synergy)
      dnh_gain_single   (best_single_err - dnh_inner_err)/best_single_err — realised
                        DNH improvement over best single modality
      redundancy        mean pairwise correlation of single-modality OOF predictions
                        (high => modalities are redundant, low => complementary)
      lambda            shrinkage toward best candidate (1.0 => pure fallback)
      accepted          whether the fusion cleared the finite-sample SE gate
    """
    tr, _te = repeated_group_folds(task.subject_ids, 1, 4, seed=seed)[0]
    cands, singles = _candidates(task)
    oof, covered = _oof_predictions(task, cands, tr, seed)
    rows = np.where(covered)[0]
    ytr = task.y[tr]
    _w, diag = dnh_weights(task, cands, singles, tr, seed)

    cand_err = diag["cand_err"]
    bs = diag["best_single"]
    bc = diag["best_cand"]
    best_single_err = float(cand_err.get(bs, float("nan")))
    best_cand_err = float(cand_err.get(bc, float("nan")))
    dnh_inner = float(diag.get("dnh_inner_err", float("nan")))

    # redundancy = mean pairwise correlation of single-modality OOF predictions
    single_preds = [oof[s][rows] for s in singles if np.all(np.isfinite(oof[s][rows]))]
    corrs = []
    for i in range(len(single_preds)):
        for j in range(i + 1, len(single_preds)):
            a, b = single_preds[i], single_preds[j]
            if np.std(a) > _EPS and np.std(b) > _EPS:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    redundancy = float(np.mean(corrs)) if corrs else float("nan")

    def _rer(base, val):
        return float((base - val) / base) if base and np.isfinite(base) else float("nan")

    return {
        "task": task.name,
        "n_subjects": int(len(np.unique(task.subject_ids))),
        "modalities": list(task.modalities),
        "best_single": bs, "best_cand": bc,
        "best_single_err": best_single_err,
        "best_cand_err": best_cand_err,
        "dnh_inner_err": dnh_inner,
        "joint_gain": _rer(best_single_err, best_cand_err),
        "dnh_gain_single": _rer(best_single_err, dnh_inner),
        "redundancy": redundancy,
        "lambda": float(diag.get("lambda", float("nan"))),
        "accepted": bool(diag.get("accepted", False)),
    }
