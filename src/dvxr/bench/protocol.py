"""dvxr.bench.protocol — repeated grouped CV + honest statistics.

Everything here operates on *error* metrics (lower is better): 1-AUROC,
1-balanced-accuracy, MAE, ... so relative error reduction (RER) is
(base_err - prop_err) / base_err, and "meets >=50%" means the fused model
cut the best baseline's error at least in half.

No leakage by construction: folds are subject/patient-disjoint. Significance is
a paired Wilcoxon signed-rank across folds; CIs are bootstrap over folds; Holm
corrects across tasks. The test split is evaluated once (frozen-test discipline
is enforced by the caller, not here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- CV folds
def repeated_group_folds(subject_ids: Sequence,
                         n_repeats: int = 5, n_folds: int = 5,
                         seed: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Subject-disjoint (train_idx, test_idx) folds, repeated with reshuffling.

    Groups (subjects) are partitioned into ``n_folds`` disjoint blocks; each block
    is the test set once per repeat. With fewer subjects than n_folds, n_folds is
    clamped so every fold still holds out >=1 whole subject.
    """
    sids = np.asarray(subject_ids)
    uniq = np.unique(sids)
    k = int(min(n_folds, len(uniq)))
    if k < 2:
        raise ValueError(f"need >=2 subjects for grouped CV, got {len(uniq)}")
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        order = uniq.copy()
        rng.shuffle(order)
        blocks = np.array_split(order, k)
        for b in blocks:
            test_subj = set(b.tolist())
            test_idx = np.array([i for i, s in enumerate(sids) if s in test_subj])
            train_idx = np.array([i for i, s in enumerate(sids) if s not in test_subj])
            if len(test_idx) and len(train_idx):
                folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 7) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_wilcoxon(prop_err: Sequence[float], base_err: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank p-value that prop_err < base_err (one-sided).

    Returns 1.0 when the test is undefined (all differences zero / <1 pair).
    """
    from scipy.stats import wilcoxon
    a = np.asarray(prop_err, dtype=float)
    b = np.asarray(base_err, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = b - a                                   # >0 when proposed has lower error
    if len(d) < 1 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, alternative="less").pvalue)
    except ValueError:
        return 1.0


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size for (a<b): fraction favouring a minus against.

    Positive => values in ``a`` (proposed error) tend to be lower than ``b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((lt - gt) / (len(a) * len(b)))


def holm_correction(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values keyed the same as the input."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)             # enforce monotonicity
        adj[k] = running
    return adj


# ---------------------------------------------------------------- RER
@dataclass
class RelativityResult:
    task: str
    metric: str
    best_baseline: str
    base_err: float
    prop_err: float
    delta_abs: float
    rer_pct: float
    rer_ci: Tuple[float, float]
    p_wilcoxon: float
    p_holm: float = float("nan")
    cliffs_delta: float = float("nan")
    n_folds: int = 0
    per_fold: Dict[str, List[float]] = field(default_factory=dict)

    def meets_target(self, target_pct: float = 50.0, alpha: float = 0.05) -> bool:
        """Strong claim: CI lower bound clears the target AND paired test is sig."""
        p = self.p_holm if np.isfinite(self.p_holm) else self.p_wilcoxon
        return (self.rer_ci[0] >= target_pct) and (p < alpha)

    def as_row(self, target_pct: float = 50.0) -> dict:
        return {
            "task": self.task, "metric": self.metric,
            "best_baseline": self.best_baseline,
            "base_err": round(self.base_err, 4),
            "prop_err": round(self.prop_err, 4),
            "delta_abs": round(self.delta_abs, 4),
            "RER_pct": round(self.rer_pct, 2),
            "RER_CI_low": round(self.rer_ci[0], 2),
            "RER_CI_high": round(self.rer_ci[1], 2),
            "p_wilcoxon": _fmt_p(self.p_wilcoxon),
            "p_holm": _fmt_p(self.p_holm),
            "cliffs_delta": round(self.cliffs_delta, 3),
            "n_folds": self.n_folds,
            f"meets_>={int(target_pct)}%": bool(self.meets_target(target_pct)),
        }


def _fmt_p(p: float) -> float:
    return float("nan") if p is None or not np.isfinite(p) else round(float(p), 5)


def relativity(task: str, metric: str, best_baseline: str,
               prop_fold_err: Sequence[float], base_fold_err: Sequence[float],
               seed: int = 7, n_boot: int = 2000) -> RelativityResult:
    """Assemble the RER result from paired per-fold error arrays.

    RER and its CI are computed on the paired fold means so the CI reflects
    subject-fold variability, not window count.
    """
    prop = np.asarray(prop_fold_err, dtype=float)
    base = np.asarray(base_fold_err, dtype=float)
    m = np.isfinite(prop) & np.isfinite(base)
    prop, base = prop[m], base[m]
    base_mean = float(np.mean(base)) if len(base) else float("nan")
    prop_mean = float(np.mean(prop)) if len(prop) else float("nan")
    rer = 100.0 * (base_mean - prop_mean) / base_mean if base_mean else float("nan")

    # bootstrap the RER over paired folds
    rng = np.random.default_rng(seed)
    rers = []
    if len(prop) >= 2:
        idx = np.arange(len(prop))
        for _ in range(n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            bm = base[s].mean()
            if bm:
                rers.append(100.0 * (bm - prop[s].mean()) / bm)
    ci = ((float(np.percentile(rers, 2.5)), float(np.percentile(rers, 97.5)))
          if rers else (float("nan"), float("nan")))

    return RelativityResult(
        task=task, metric=metric, best_baseline=best_baseline,
        base_err=base_mean, prop_err=prop_mean,
        delta_abs=base_mean - prop_mean, rer_pct=rer, rer_ci=ci,
        p_wilcoxon=paired_wilcoxon(prop, base),
        cliffs_delta=cliffs_delta(prop, base),
        n_folds=int(len(prop)),
        per_fold={"proposed": prop.tolist(), "baseline": base.tolist()},
    )
