"""dvxr.tasks.train — joint multi-task training loop (ARCHITECTURE §A7).

AdamW + linear-warmup→cosine-decay + grad-clip(1.0) (+ optional weight EMA), jointly
optimizing encoders + codebooks + fusion + heads. Logs per-term losses (and learned
σ_t when uncertainty weighting is on) to outputs/train_log.csv. Personalization
(per_subject_normalize + PersonalizedCalibrator) supports population + per-subject
held-out reporting.
"""
from __future__ import annotations

import math
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

from dvxr.calibration import expected_calibration_error
from dvxr.personalization import PersonalizedCalibrator
from dvxr.tasks.losses import (
    build_uncertainty_weighting,
    class_weighted_ce,
    huber_forecast,
    info_nce,
    mse_recon,
    total_loss,
)


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def train_multitask(
    model,
    features: Dict[str, "object"],
    labels: Dict[str, "object"],
    forecast_target=None,
    config=None,
    log_path: str | pathlib.Path = "outputs/train_log.csv",
    uncertainty_weighting: bool = False,
    ema_decay: Optional[float] = None,
) -> dict:
    """Train ``model`` in place; returns history + learned σ_t (if enabled)."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    cfg = config
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    forecast_key = "glucose"
    uw = None
    params = list(model.parameters())
    if uncertainty_weighting:
        uw = build_uncertainty_weighting(list(labels.keys()) + [forecast_key])
        params = params + list(uw.parameters())

    opt = torch.optim.AdamW(
        params, lr=cfg.lr_fusion, weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2))
    total_steps = cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} \
        if ema_decay else None

    history = []
    model.train()
    for epoch in range(cfg.epochs):
        for g in opt.param_groups:
            g["lr"] = cfg.lr_fusion * _cosine_warmup(epoch, warmup, total_steps)

        out = model(features)
        task_losses = {t: class_weighted_ce(out["logits"][t], labels[t])
                       for t in labels}
        if forecast_target is not None:
            task_losses[forecast_key] = huber_forecast(out["forecast"], forecast_target)
        else:
            task_losses[forecast_key] = out["forecast"].sum() * 0.0

        recon = mse_recon(out["recon"], features)
        align = info_nce(out["z"], cfg.align_temperature)
        loss = total_loss(task_losses, out["vq_loss"], recon, align, cfg, uw)

        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        if ema is not None:
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)

        row = {"epoch": epoch, "total": float(loss.detach()),
               "vq": float(out["vq_loss"].detach()), "recon": float(recon.detach()),
               "align": float(align.detach()), "lr": opt.param_groups[0]["lr"]}
        for t, l in task_losses.items():
            row[f"task_{t}"] = float(l.detach())
        if uw is not None:
            for t, s in uw.sigmas().items():
                row[f"sigma_{t}"] = s
        history.append(row)

    model.eval()
    if ema is not None:
        model.load_state_dict(ema)

    out_path = pathlib.Path(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_path, index=False)

    return {"model": model, "history": history, "log_path": str(out_path),
            "sigmas": uw.sigmas() if uw is not None else None}


def population_and_personalized_metrics(
    subject_ids, probabilities, truths, split_frac: float = 0.5,
) -> dict:
    """Subject-agnostic vs per-subject calibration ECE on a held-out split (§A7)."""
    sids = np.asarray(subject_ids, dtype=str)
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(truths, dtype=int)
    n = len(p)
    k = max(1, int(n * split_frac))
    tr, te = slice(0, k), slice(k, n)

    pop_ece = float(expected_calibration_error(y[te], p[te]))
    cal = PersonalizedCalibrator()
    cal.fit(sids[tr], p[tr], y[tr])
    p_pers = cal.predict(sids[te], p[te])
    pers_ece = float(expected_calibration_error(y[te], p_pers))
    return {"population_ece": pop_ece, "personalized_ece": pers_ece,
            "personalized_probs": p_pers}
