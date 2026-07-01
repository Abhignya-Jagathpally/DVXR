"""dvxr.tasks.model — MultiTaskModel: learnable per-modality encoders -> VQ codebooks
-> fusion -> joint latent h -> multi-task heads, with per-modality decoders for the
masked-reconstruction term. Jointly trainable end-to-end (ARCHITECTURE §A5-A7).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from dvxr.encoders.base import _torch_available


def build_multitask_model(config, input_dims: Dict[str, int],
                          classification_tasks: Optional[List[str]] = None,
                          modalities: Optional[List[str]] = None):
    """Construct a MultiTaskModel (torch required).

    input_dims: per-modality feature width (the encoder input size).
    """
    if not _torch_available():
        raise RuntimeError("MultiTaskModel requires torch")
    import torch
    from torch import nn

    from dvxr.fusion.model import build_cacmf_model
    from dvxr.tasks.heads import CLASSIFICATION_TASKS, build_task_module

    mods = list(modalities or input_dims.keys())
    cls_tasks = list(classification_tasks or CLASSIFICATION_TASKS)

    class MultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(config.seed)
            self.modalities = mods
            self.classification_tasks = cls_tasks
            self.encoders = nn.ModuleDict(
                {m: nn.Linear(input_dims[m], config.d) for m in mods})
            self.decoders = nn.ModuleDict(
                {m: nn.Linear(config.d, input_dims[m]) for m in mods})
            self.cacmf = build_cacmf_model(config, mods)
            self.heads = build_task_module(config, config.d_f, cls_tasks)

        def encode(self, features: Dict[str, "torch.Tensor"]):
            return {m: self.encoders[m](features[m]) for m in features}

        def forward(self, features: Dict[str, "torch.Tensor"], use_codebook: bool = True):
            z = self.encode(features)
            fo = self.cacmf.fuse(z, use_codebook=use_codebook)
            logits, yhat = self.heads(fo.h)
            q = self.cacmf._last_q
            recon = {m: self.decoders[m](q[m]) for m in q}
            return {
                "z": z, "h": fo.h, "logits": logits, "forecast": yhat,
                "recon": recon, "vq_loss": self.cacmf.vq_loss(), "fusion": fo,
            }

        def probabilities(self, features, use_codebook: bool = True):
            out = self.forward(features, use_codebook=use_codebook)
            return {t: torch.softmax(out["logits"][t], dim=1) for t in out["logits"]}

    return MultiTaskModel()
