"""dvxr.eval.splits — subject/patient-held-out split utilities (ARCHITECTURE §A7).

Honest metrics require that no subject appears in both train and test.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class InsufficientSubjectsError(ValueError):
    """Raised when there are too few unique subjects to form the requested held-out folds honestly."""


def subject_kfold(subject_ids, n_folds: int = 5, seed: int = 7):
    """Yield ``(train_idx, test_idx)`` for ``n_folds`` folds whose TEST subjects are disjoint and
    together cover every subject exactly once — so pooling test predictions across folds counts each
    participant once (no cross-fold subject leakage). Raises when there are fewer subjects than folds."""
    sids = np.asarray(subject_ids)
    unique = np.unique(sids)
    if len(unique) < n_folds:
        raise InsufficientSubjectsError(
            f"{len(unique)} unique subjects < {n_folds} folds — cannot form honest held-out folds")
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    fold_of = {s: (i % n_folds) for i, s in enumerate(order)}
    assigned = np.array([fold_of[s] for s in sids])
    out = []
    for k in range(n_folds):
        test_idx = np.where(assigned == k)[0]
        train_idx = np.where(assigned != k)[0]
        out.append((train_idx, test_idx))
    return out


def subject_holdout_split(subject_ids, test_frac: float = 0.3,
                          seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) with disjoint subjects (deterministic)."""
    sids = np.asarray(subject_ids)
    unique = np.unique(sids)
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    n_test = max(1, int(round(len(order) * test_frac)))
    test_subjects = set(order[:n_test].tolist())
    is_test = np.array([s in test_subjects for s in sids])
    test_idx = np.where(is_test)[0]
    train_idx = np.where(~is_test)[0]
    return train_idx, test_idx
