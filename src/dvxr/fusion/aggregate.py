"""dvxr.fusion.aggregate — prediction-level aggregation baselines (ARCHITECTURE §A4).

These operate on per-modality head PROBABILITIES ``{modality: array(B, C)}`` and are
pure NumPy (no torch, always-runnable). They are orthogonal to the five fusion
strategies (which combine latents).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _stack(probs: Dict[str, np.ndarray]):
    mods = list(probs.keys())
    if not mods:
        raise ValueError("aggregate: no modality probabilities provided")
    arr = np.stack([np.asarray(probs[m], dtype=np.float64) for m in mods], axis=0)
    return mods, arr  # (M, B, C)


def ensemble_avg(probs: Dict[str, np.ndarray]) -> np.ndarray:
    """p = mean_m p_m."""
    _mods, arr = _stack(probs)
    return arr.mean(axis=0)


def weighted_late(probs: Dict[str, np.ndarray],
                  weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """p = Σ_m w_m p_m with w normalized over present modalities."""
    mods, arr = _stack(probs)
    if weights is None:
        w = np.ones(len(mods))
    else:
        w = np.array([float(weights.get(m, 0.0)) for m in mods])
    s = w.sum()
    if s <= 0:
        w = np.ones(len(mods))
        s = w.sum()
    w = w / s
    return np.tensordot(w, arr, axes=([0], [0]))  # (B, C)


def normalized_entropy_confidence(p: np.ndarray) -> np.ndarray:
    """c = 1 - H(p)/log(C) per sample, in [0, 1] (1 = fully confident)."""
    p = np.asarray(p, dtype=np.float64)
    C = p.shape[1]
    H = -(p * np.log(p + 1e-12)).sum(axis=1)
    return 1.0 - H / np.log(C)


def confidence_weighted(probs: Dict[str, np.ndarray]) -> np.ndarray:
    """p = Σ_m c_m p_m / Σ_m c_m, c_m = normalized-entropy confidence per sample.

    Lets a low-confidence modality defer to the confident ones, sample by sample.
    """
    mods, arr = _stack(probs)                      # (M, B, C)
    conf = np.stack([normalized_entropy_confidence(arr[i]) for i in range(len(mods))],
                    axis=0)                        # (M, B)
    denom = conf.sum(axis=0) + 1e-12               # (B,)
    weighted = (conf[:, :, None] * arr).sum(axis=0)  # (B, C)
    return weighted / denom[:, None]


AGGREGATORS = {
    "weighted_late": weighted_late,
    "ensemble_avg": ensemble_avg,
    "confidence_weighted": confidence_weighted,
}
