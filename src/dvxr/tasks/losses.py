"""dvxr.tasks.losses — the unified multi-task objective (ARCHITECTURE §A6).

    L_total = Σ_t λ_t L_task_t + λ_vq Σ L_vq + λ_rec Σ L_recon + λ_alg L_align

with class-weighted cross-entropy, Huber forecasting, InfoNCE cross-modal alignment,
and an optional Kendall & Gal uncertainty-weighting path (learned σ_t). torch is
imported lazily so importing this module without torch does not fail at import time.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def class_weighted_ce(logits, targets):
    """Cross-entropy with inverse-frequency class weights. targets==-1 -> ignored."""
    import torch
    import torch.nn.functional as F
    valid = targets >= 0
    if int(valid.sum()) == 0:
        return logits.sum() * 0.0
    logit_v, tgt_v = logits[valid], targets[valid]
    classes, counts = torch.unique(tgt_v, return_counts=True)
    w = torch.ones(logits.shape[1])
    n = int(tgt_v.numel())
    for c, cnt in zip(classes.tolist(), counts.tolist()):
        w[c] = n / (len(classes) * cnt)
    return F.cross_entropy(logit_v, tgt_v, weight=w)


def huber_forecast(pred, target, mask=None):
    import torch.nn.functional as F
    if mask is not None:
        if int(mask.sum()) == 0:
            return pred.sum() * 0.0
        pred, target = pred[mask], target[mask]
    return F.huber_loss(pred, target)


def mse_recon(recon: Dict[str, "object"], features: Dict[str, "object"]):
    import torch
    import torch.nn.functional as F
    keys = [m for m in recon if m in features]
    if not keys:
        return torch.tensor(0.0)
    return torch.stack([F.mse_loss(recon[m], features[m]) for m in keys]).mean()


def info_nce(z_dict: Dict[str, "object"], temperature: float = 0.1):
    """Cross-modal InfoNCE: same-sample latents of different modalities are positives."""
    import torch
    import torch.nn.functional as F
    mods = list(z_dict.keys())
    if len(mods) < 2:
        return torch.tensor(0.0)
    zs = [F.normalize(z_dict[m], dim=1) for m in mods]
    B = zs[0].shape[0]
    labels = torch.arange(B)
    total, count = 0.0, 0
    for i in range(len(mods)):
        for j in range(len(mods)):
            if i == j:
                continue
            sim = zs[i] @ zs[j].t() / max(temperature, 1e-6)
            total = total + F.cross_entropy(sim, labels)
            count += 1
    return total / max(count, 1)


def build_uncertainty_weighting(task_names: List[str]):
    """Kendall & Gal learned per-task σ_t (ARCHITECTURE §A6). torch nn.Module."""
    import torch
    from torch import nn

    class UncertaintyWeighting(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_sigma = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(())) for t in task_names})

        def combine(self, losses: Dict[str, "object"]):
            total = 0.0
            for t, loss in losses.items():
                s = self.log_sigma[t]
                total = total + 0.5 * torch.exp(-2.0 * s) * loss + s
            return total

        def sigmas(self) -> Dict[str, float]:
            return {t: float(torch.exp(self.log_sigma[t].detach()))
                    for t in self.log_sigma}

    return UncertaintyWeighting()


def total_loss(task_losses: Dict[str, "object"], vq_loss, recon_loss, align_loss,
               config, uw=None):
    """Combine all terms with the config's relative weights (or learned σ_t if uw)."""
    if uw is not None:
        task_term = uw.combine(task_losses)
    else:
        task_term = sum(config.lambda_task * loss for loss in task_losses.values())
    return (task_term
            + config.lambda_vq * vq_loss
            + config.lambda_recon * recon_loss
            + config.lambda_align * align_loss)
