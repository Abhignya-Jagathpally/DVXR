"""dvxr.fusion.strategies — the five fusion strategies (ARCHITECTURE §A4).

early | intermediate | late_weighted | attention | cross_modal.

Each is a torch ``nn.Module`` taking ``latents: Dict[modality -> Tensor(B, d)]`` for
an ARBITRARY present-modality subset and returning a ``FusionOutput`` with the joint
latent ``h (B, d_f)``. Absent modalities are represented by a learned per-modality
"absent" token (explicit, never a silent data imputation) and masked out of attention.
Attention (``α_m``) and late-fusion weights (``w_m``) are exported for explainability.

torch is imported lazily (via a builder) so importing this module without torch —
and ``dvxr.fusion.aggregate`` — still works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dvxr.encoders.base import _torch_available


@dataclass
class FusionOutput:
    h: "object"                                   # (B, d_f) joint latent
    present: List[str]
    attention: Optional[Dict[str, "object"]] = None   # α_m per present modality (B,)
    weights: Optional[Dict[str, "object"]] = None     # w_m per present modality (scalar)


def _build(config, modalities: List[str]):
    import torch
    from torch import nn

    d, d_f = config.d, config.d_f
    torch.manual_seed(config.seed)

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = list(modalities)
            self.d, self.d_f = d, d_f
            # learned "absent" token per modality (explicit missing-modality marker)
            self.absent = nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(d)) for m in self.modalities})

        def _stack(self, latents: Dict[str, "torch.Tensor"]):
            present = [m for m in self.modalities if m in latents]
            if not present:
                raise ValueError("fusion: no present modalities")
            B = latents[present[0]].shape[0]
            xs, mask = [], []
            for m in self.modalities:
                if m in latents:
                    xs.append(latents[m])
                    mask.append(True)
                else:
                    xs.append(self.absent[m].unsqueeze(0).expand(B, -1))
                    mask.append(False)
            x = torch.stack(xs, dim=1)                       # (B, M, d)
            present_mask = torch.tensor(mask, dtype=torch.bool)  # (M,)
            return x, present_mask, present, B

    class EarlyFusion(_Base):
        def __init__(self):
            super().__init__()
            M = len(self.modalities)
            self.mlp = nn.Sequential(
                nn.Linear(M * d, d_f), nn.ReLU(), nn.Linear(d_f, d_f))

        def forward(self, latents):
            x, _mask, present, B = self._stack(latents)
            h = self.mlp(x.reshape(B, -1))
            return FusionOutput(h=h, present=present)

    class IntermediateFusion(_Base):
        def __init__(self):
            super().__init__()
            M = len(self.modalities)
            self.proj = nn.ModuleDict(
                {m: nn.Linear(d, d) for m in self.modalities})
            self.mlp = nn.Sequential(
                nn.Linear(M * d, d_f), nn.ReLU(), nn.Linear(d_f, d_f))

        def forward(self, latents):
            present = [m for m in self.modalities if m in latents]
            B = latents[present[0]].shape[0]
            parts = []
            for m in self.modalities:
                if m in latents:
                    parts.append(self.proj[m](latents[m]))
                else:
                    parts.append(self.absent[m].unsqueeze(0).expand(B, -1))
            h = self.mlp(torch.cat(parts, dim=1))
            return FusionOutput(h=h, present=present)

    class LateWeightedFusion(_Base):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.zeros(len(self.modalities)))
            self.proj = nn.ModuleDict(
                {m: nn.Linear(d, d_f) for m in self.modalities})

        def forward(self, latents):
            present = [m for m in self.modalities if m in latents]
            idx = [self.modalities.index(m) for m in present]
            w = torch.softmax(self.theta[idx], dim=0)          # (P,) sums to 1
            h = sum(w[i] * self.proj[m](latents[m]) for i, m in enumerate(present))
            weights = {m: w[i] for i, m in enumerate(present)}
            return FusionOutput(h=h, present=present, weights=weights)

    class AttentionFusion(_Base):
        def __init__(self):
            super().__init__()
            self.type_emb = nn.Embedding(len(self.modalities), d)
            self.W = nn.Linear(d, d)
            self.v = nn.Linear(d, 1, bias=False)
            self.out = nn.Linear(d, d_f)

        def forward(self, latents):
            present = [m for m in self.modalities if m in latents]
            idx = [self.modalities.index(m) for m in present]
            toks = torch.stack([
                latents[m] + self.type_emb(torch.tensor(self.modalities.index(m)))
                for m in present], dim=1)                      # (B, P, d)
            scores = self.v(torch.tanh(self.W(toks))).squeeze(-1)  # (B, P)
            alpha = torch.softmax(scores, dim=1)               # (B, P) sums to 1
            h = self.out((alpha.unsqueeze(-1) * toks).sum(dim=1))
            attention = {m: alpha[:, i] for i, m in enumerate(present)}
            return FusionOutput(h=h, present=present, attention=attention)

    class CrossModalFusion(_Base):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.ModuleDict(
                {m: nn.Linear(d, d_f) for m in self.modalities})
            self.type_emb = nn.Embedding(len(self.modalities) + 1, d_f)  # +1 CLS
            self.cls = nn.Parameter(torch.zeros(1, 1, d_f))
            layer = nn.TransformerEncoderLayer(
                d_model=d_f, nhead=config.n_heads,
                dim_feedforward=d_f * 2, dropout=config.dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_fusion_layers)
            self.out = nn.Linear(d_f, d_f)
            # attention-pool head for explainability (α over present tokens)
            self.Wp = nn.Linear(d_f, d_f)
            self.vp = nn.Linear(d_f, 1, bias=False)

        def forward(self, latents):
            x, present_mask, present, B = self._stack(latents)   # x (B, M, d)
            M = len(self.modalities)
            toks = torch.stack([
                self.in_proj[m](x[:, j, :]) for j, m in enumerate(self.modalities)],
                dim=1)                                           # (B, M, d_f)
            type_ids = torch.arange(1, M + 1)
            toks = toks + self.type_emb(type_ids).unsqueeze(0)
            cls = self.cls.expand(B, -1, -1) + self.type_emb(torch.tensor([0]))
            seq = torch.cat([cls, toks], dim=1)                  # (B, M+1, d_f)
            # key padding: True = ignore. CLS never ignored; absent modalities ignored.
            pad = torch.cat([torch.zeros(1, dtype=torch.bool), ~present_mask])
            pad = pad.unsqueeze(0).expand(B, -1)                 # (B, M+1)
            enc = self.encoder(seq, src_key_padding_mask=pad)
            h = self.out(enc[:, 0])                              # CLS readout
            # attention-pool α over PRESENT modality tokens (explainability export)
            tok_out = enc[:, 1:]                                 # (B, M, d_f)
            scores = self.vp(torch.tanh(self.Wp(tok_out))).squeeze(-1)  # (B, M)
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~present_mask.unsqueeze(0), neg)
            alpha = torch.softmax(scores, dim=1)                 # (B, M)
            attention = {m: alpha[:, self.modalities.index(m)] for m in present}
            return FusionOutput(h=h, present=present, attention=attention)

    return {
        "early": EarlyFusion,
        "intermediate": IntermediateFusion,
        "late_weighted": LateWeightedFusion,
        "attention": AttentionFusion,
        "cross_modal": CrossModalFusion,
    }


def get_fusion_strategy(name: str, config, modalities: Optional[List[str]] = None):
    """Instantiate a fusion strategy by name (torch required)."""
    if not _torch_available():
        raise RuntimeError("fusion strategies require torch")
    from dvxr.config import MODALITIES
    classes = _build(config, list(modalities or MODALITIES))
    if name not in classes:
        raise ValueError(f"unknown fusion_strategy {name!r}; choose from {list(classes)}")
    return classes[name]()
