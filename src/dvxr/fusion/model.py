"""dvxr.fusion.model — CACMFModel wires per-modality latents -> VQ codebooks ->
chosen fusion strategy -> joint latent h, and exports latents/attention/weights (§A5).

torch is imported lazily via ``build_cacmf_model``; ``dvxr.fusion.aggregate`` stays
importable without torch.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dvxr.encoders.base import _torch_available


@dataclass(frozen=True)
class FusionResult:
    """A request-local fusion result (spec §11 concurrency correction).

    Everything a caller needs from one ``fuse`` call, returned in the call's own return value so
    concurrent/interleaved requests on a shared model instance never read each other's state via the
    mutable ``_last*`` accessors. ``h`` is the joint latent; ``attention`` / ``weights`` are this
    call's explanation; ``codes`` are the per-modality VQ indices; ``vq_loss`` is a float.
    """
    h: object
    attention: Optional[Dict] = None
    weights: Optional[Dict] = None
    codes: Dict = field(default_factory=dict)
    present: Optional[Dict] = None
    vq_loss: float = 0.0


def build_cacmf_model(config, modalities: Optional[List[str]] = None):
    """Construct a CACMFModel (torch required)."""
    if not _torch_available():
        raise RuntimeError("CACMFModel requires torch")
    import torch
    from torch import nn

    from dvxr.config import MODALITIES
    from dvxr.encoders.codebook import get_vector_quantizer_class
    from dvxr.fusion.strategies import get_fusion_strategy

    VQ = get_vector_quantizer_class()
    mods = list(modalities or MODALITIES)

    class CACMFModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(config.seed)
            self.config = config
            self.modalities = mods
            self.fusion = get_fusion_strategy(config.fusion_strategy, config, mods)
            self.codebooks = nn.ModuleDict({
                m: VQ(num_codes=config.codebook_size, dim=config.d,
                      beta=config.commitment_beta, gumbel=config.gumbel,
                      temperature=config.temperature) for m in mods})
            self._last = None
            self._last_codes: Dict[str, object] = {}
            self._last_z: Dict[str, object] = {}
            self._last_q: Dict[str, object] = {}
            self._last_vqloss = None

        def quantize(self, latents: Dict[str, "torch.Tensor"]):
            q, codes, losses = {}, {}, []
            for m, z in latents.items():
                out = self.codebooks[m](z, training=self.training)
                q[m] = out.quantized
                codes[m] = out.indices
                losses.append(out.loss)
            vq_loss = torch.stack(losses).sum() if losses else torch.tensor(0.0)
            return q, codes, vq_loss

        def fuse(self, latents: Dict[str, "torch.Tensor"], use_codebook: bool = True):
            self._last_z = dict(latents)
            if use_codebook:
                q, codes, vq_loss = self.quantize(latents)
            else:
                q, codes, vq_loss = dict(latents), {}, torch.tensor(0.0)
            self._last_q, self._last_codes, self._last_vqloss = q, codes, vq_loss
            self._last = self.fusion(q)
            return self._last

        def fuse_result(self, latents: Dict[str, "torch.Tensor"], use_codebook: bool = True) -> "FusionResult":
            """Request-local fusion: compute and return a self-contained FusionResult WITHOUT reading
            or depending on the mutable ``_last*`` accessors, so concurrent callers can't clobber each
            other's attention/weights. (``fuse`` still updates ``_last*`` for legacy callers.)"""
            if use_codebook:
                q, codes, vq_loss = self.quantize(latents)
            else:
                q, codes, vq_loss = dict(latents), {}, torch.tensor(0.0)
            fo = self.fusion(q)
            return FusionResult(
                h=fo.h,
                attention=getattr(fo, "attention", None),
                weights=getattr(fo, "weights", None),
                present=getattr(fo, "present", None),
                codes=codes,
                vq_loss=float(vq_loss.detach()) if hasattr(vq_loss, "detach") else float(vq_loss))

        def forward(self, latents, use_codebook: bool = True):
            return self.fuse(latents, use_codebook)

        def attention_weights(self):
            return self._last.attention if self._last is not None else None

        def fusion_weights(self):
            return self._last.weights if self._last is not None else None

        def vq_loss(self):
            return self._last_vqloss

        def export_latents(self, latents, out_dir="outputs", index=None,
                           use_codebook: bool = True):
            out = pathlib.Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                fo = self.fuse(latents, use_codebook=use_codebook)
            n = fo.h.shape[0]
            idx = list(index) if index is not None else list(range(n))
            paths = {}

            np.save(out / "latent_joint_h.npy", fo.h.detach().cpu().numpy())
            paths["h"] = out / "latent_joint_h.npy"

            if self._last_codes:
                codes = pd.DataFrame(
                    {m: self._last_codes[m].detach().cpu().numpy()
                     for m in self._last_codes}, index=idx)
                codes.to_csv(out / "codebook_indices.csv")
                paths["codes"] = out / "codebook_indices.csv"

            if fo.attention is not None:
                att = pd.DataFrame(
                    {m: fo.attention[m].detach().cpu().numpy() for m in fo.attention},
                    index=idx)
                att.to_csv(out / "fusion_attention.csv")
                paths["attention"] = out / "fusion_attention.csv"

            if fo.weights is not None:
                w = pd.DataFrame(
                    {m: [float(fo.weights[m])] for m in fo.weights})
                w.to_csv(out / "fusion_weights.csv", index=False)
                paths["weights"] = out / "fusion_weights.csv"

            return paths

    return CACMFModel()
