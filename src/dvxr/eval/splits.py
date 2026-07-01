"""dvxr.eval.splits — subject/patient-held-out split utilities (ARCHITECTURE §A7).

Honest metrics require that no subject appears in both train and test.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


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
