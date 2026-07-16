"""dvxr.fusion.aggregate — prediction-level aggregation baselines (ARCHITECTURE §A4).

These operate on per-modality head PROBABILITIES ``{modality: array(B, C)}`` and are
pure NumPy (no torch, always-runnable). They are orthogonal to the five fusion
strategies (which combine latents).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class GatedFusionResult:
    """Quality-gated fusion outcome that ABSTAINS instead of manufacturing confidence. When every
    modality gate for a sample collapses to ~0 (all unavailable / stale / low-quality / OOD), the
    fused probability for that sample is None-equivalent (NaN) and ``abstained`` marks it — the caller
    must surface an abstention, never a mean (spec §8.7, §11)."""
    fused: Optional[np.ndarray]              # (B, C); NaN rows where abstained
    weights: Dict[str, float] = field(default_factory=dict)
    abstained: Optional[np.ndarray] = None   # (B,) bool per-sample
    all_abstained: bool = False
    abstain_reason: Optional[str] = None


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


def quality_gated(probs: Dict[str, np.ndarray],
                  quality: Optional[Dict[str, float]] = None,
                  freshness: Optional[Dict[str, float]] = None,
                  availability: Optional[Dict[str, float]] = None,
                  ood: Optional[Dict[str, float]] = None,
                  return_weights: bool = False,
                  strict_unknown: bool = False):
    """Quality-aware gated late fusion (spec §5 "confidence-aware fusion", §11).

    Each modality's gate multiplies signal reliability factors, all in [0, 1]:

        g_m = availability_m · quality_m · freshness_m · confidence_m · (1 - ood_m)

    where confidence is the per-sample normalized-entropy confidence. A reliability map that is not
    supplied at all leaves that factor neutral (1.0). With ``strict_unknown=True`` a map that IS
    supplied but omits a modality fails CLOSED for that modality (gate 0) — unknown quality/freshness/
    availability is never treated as perfect. The fused probability is
    the gate-weighted average over modalities, per sample. Unlike ``confidence_weighted``, this lets a
    STALE or LOW-QUALITY or OUT-OF-DISTRIBUTION modality be down-weighted even when it is confident —
    the failure modes spec §9 calls out (noisy EEG, stale CGM feed, clock drift, OOD participant).

    If every gate collapses to ~0 for a sample (all modalities unreliable) the honest outcome is to
    ABSTAIN, not to invent confidence. Such samples are returned as NaN here (never an unweighted
    mean); use :func:`gated_fusion` for the typed, abstention-aware result. ``return_weights`` still
    exposes the (near-zero) per-modality weights so an upstream gate can act on them.
    """
    mods, arr = _stack(probs)                                   # (M, B, C)
    M, B, _C = arr.shape

    def _factor(d, m, *, neutral, unknown_penalty):
        # dict not supplied at all ⇒ neutral (that signal simply isn't being gated on).
        # dict supplied but THIS modality absent ⇒ UNKNOWN: under strict_unknown, treat as the
        # fail-CLOSED penalty (unknown quality ≠ perfect); otherwise keep the legacy neutral default.
        if not d:
            return neutral
        if m in d:
            return float(d[m])
        return unknown_penalty if strict_unknown else neutral

    conf = np.stack([normalized_entropy_confidence(arr[i]) for i in range(M)], axis=0)  # (M, B)
    gate = np.empty((M, B), dtype=np.float64)
    for i, m in enumerate(mods):
        # availability/quality/freshness: neutral 1.0, unknown-penalty 0.0 (fail closed → zero the gate).
        # ood: neutral 0.0 (in-distribution), unknown-penalty 1.0 (treat as fully OOD ⇒ (1-1)=0).
        static = (_factor(availability, m, neutral=1.0, unknown_penalty=0.0)
                  * _factor(quality, m, neutral=1.0, unknown_penalty=0.0)
                  * _factor(freshness, m, neutral=1.0, unknown_penalty=0.0)
                  * (1.0 - _factor(ood, m, neutral=0.0, unknown_penalty=1.0)))
        gate[i] = np.clip(static, 0.0, 1.0) * conf[i]

    denom = gate.sum(axis=0)                                    # (B,)
    collapsed = denom <= 1e-12
    safe_denom = np.where(collapsed, 1.0, denom)               # avoid divide-by-zero
    fused = (gate[:, :, None] * arr).sum(axis=0) / safe_denom[:, None]
    fused[collapsed] = np.nan                                  # abstain, do NOT fall back to a mean

    if return_weights:
        # mean gate per modality across the batch, normalized → a per-modality contribution weight
        mean_gate = gate.mean(axis=1)
        wsum = mean_gate.sum() or 1.0
        weights = {m: float(mean_gate[i] / wsum) for i, m in enumerate(mods)}
        return fused, weights
    return fused


def gated_fusion(probs: Dict[str, np.ndarray],
                 quality: Optional[Dict[str, float]] = None,
                 freshness: Optional[Dict[str, float]] = None,
                 availability: Optional[Dict[str, float]] = None,
                 ood: Optional[Dict[str, float]] = None,
                 strict_unknown: bool = False) -> GatedFusionResult:
    """Quality-aware gated fusion that abstains when all gates collapse (spec §11). Returns a typed
    :class:`GatedFusionResult`: samples whose every modality gate is ~0 are marked ``abstained`` and
    their fused row is NaN; ``all_abstained`` is True when no sample could be fused.

    ``strict_unknown`` (recommended for any safety-critical deployment) fails CLOSED: when a reliability
    map is supplied but omits a modality, that modality's quality/freshness/availability is treated as
    unknown-and-therefore-unusable (gate 0) rather than perfect — unknown quality ≠ current ≠ present."""
    fused, weights = quality_gated(probs, quality=quality, freshness=freshness,
                                   availability=availability, ood=ood, return_weights=True,
                                   strict_unknown=strict_unknown)
    abstained = np.isnan(fused).any(axis=1)                    # (B,)
    all_abstained = bool(abstained.all())
    reason = "all_modality_gates_zero" if abstained.any() else None
    return GatedFusionResult(
        fused=None if all_abstained else fused, weights=weights,
        abstained=abstained, all_abstained=all_abstained, abstain_reason=reason)


AGGREGATORS = {
    "weighted_late": weighted_late,
    "ensemble_avg": ensemble_avg,
    "confidence_weighted": confidence_weighted,
    "quality_gated": quality_gated,
}
