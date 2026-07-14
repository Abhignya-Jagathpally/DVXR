"""dvxr.encoders.labram_real — a REAL pretrained LaBraM EEG foundation model, loaded from
`braindecode/labram-pretrained` safetensors via a **vendored forward** (no braindecode import).

Why vendored: `braindecode` will not import under the pinned torch 2.12 (no compatible
torchaudio) — but LaBraM's weights are plain safetensors and its architecture is a standard
ViT-style encoder we reimplement here and load by key name (`strict=True`, all 221 keys
consumed). This is the real EEG foundation model the POW names, finally runnable in this env.

Architecture (LaBraM-base, verified against the state dict + the BSD-3 braindecode reference —
see docs/SLICE_B_LABRAM.md): embed_dim=200, 12 blocks, 10 heads (per-head QK LayerNorm),
MLP 800, LayerScale; TemporalConv patch embedding; **channel-major** token layout
[CLS, ch0·p0, ch0·p1, …, ch1·p0, …]; per-channel position embedding (indexed by canonical
channel), per-patch temporal embedding (index 0 reserved). Frozen representation = the final
normed CLS token (200-d).

Honesty: correctness is validated (shape asserts, strict state-dict load, non-degeneracy) —
see tests/test_labram_real.py. A wrong token layout would make a broken forward masquerade as
"real LaBraM"; that is exactly what this project does not do, so the forward mirrors the
reference layout exactly and any deviation should fail the strict load or the tests.
"""
from __future__ import annotations

import json
from typing import List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ torch modules
def _build_modules():
    """Return the LaBraM nn.Module classes (import torch lazily so the package import
    stays torch-optional, matching the rest of dvxr)."""
    import torch
    from torch import nn

    class _TemporalConv(nn.Module):
        """conv1 (1->8,(1,15),stride(1,8),pad(0,7)) -> GN(4) -> GELU, then two
        (8->8,(1,3),pad(0,1)) -> GN(4) -> GELU. Input (B,C,npat,200) -> (B,C*npat,200)."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, (1, 15), stride=(1, 8), padding=(0, 7))
            self.norm1 = nn.GroupNorm(4, 8)
            self.conv2 = nn.Conv2d(8, 8, (1, 3), padding=(0, 1))
            self.norm2 = nn.GroupNorm(4, 8)
            self.conv3 = nn.Conv2d(8, 8, (1, 3), padding=(0, 1))
            self.norm3 = nn.GroupNorm(4, 8)
            self.act = nn.GELU()

        def forward(self, x):                        # x: (B, C, NP, 200)
            b, c, npat, s = x.shape
            x = x.reshape(b, 1, c * npat, s)         # "B chs npat spatch -> B () (chs npat) spatch"
            x = self.act(self.norm1(self.conv1(x)))  # conv -> norm -> act (reference order)
            x = self.act(self.norm2(self.conv2(x)))
            x = self.act(self.norm3(self.conv3(x)))  # (B, 8, C*NP, 25)
            bb, cc, na, t = x.shape
            return x.permute(0, 2, 3, 1).reshape(bb, na, t * cc)   # "B C NA T -> B NA (T C)"

    class _PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_conv = _TemporalConv()

        def forward(self, x):
            return self.temporal_conv(x)

    class _Attn(nn.Module):
        def __init__(self, dim=200, heads=10):
            super().__init__()
            self.heads = heads
            self.hd = dim // heads
            self.scale = self.hd ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_norm = nn.LayerNorm(self.hd)
            self.k_norm = nn.LayerNorm(self.hd)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            b, n, d = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(b, n, d)
            return self.proj(x)

    class _Block(nn.Module):
        def __init__(self, dim=200, heads=10, mlp=800):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = _Attn(dim, heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))
            self.gamma_1 = nn.Parameter(torch.ones(dim))
            self.gamma_2 = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
            return x

    class LaBraM(nn.Module):
        def __init__(self, dim=200, depth=12, heads=10, mlp=800,
                     max_chans=128, max_patches=16):
            super().__init__()
            self.dim = dim
            self.patch_embed = _PatchEmbed()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.position_embedding = nn.Parameter(torch.zeros(1, max_chans + 1, dim))
            self.temporal_embedding = nn.Parameter(torch.zeros(1, max_patches, dim))
            self.blocks = nn.ModuleList([_Block(dim, heads, mlp) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x, input_chans):
            """x: (B, C, NP, 200); input_chans: LongTensor (1+C,) canonical indices with CLS=0.
            Returns the final normed CLS token (B, dim)."""
            b, c, npat, _ = x.shape
            x = self.patch_embed(x)                                  # (B, C*NP, dim)
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)                           # (B, 1+C*NP, dim)

            # per-channel position embedding, expanded across patches (channel-major)
            pos_used = self.position_embedding[:, input_chans]       # (1, 1+C, dim)
            pos = pos_used[:, 1:, :].unsqueeze(2).expand(b, -1, npat, -1).flatten(1, 2)
            pos = torch.cat([pos_used[:, 0:1, :].expand(b, -1, -1), pos], dim=1)
            x = x + pos

            # per-patch temporal embedding (index 0 reserved), tiled across channels
            te = self.temporal_embedding[:, 1:npat + 1, :]           # (1, NP, dim)
            te = te.unsqueeze(1).expand(b, c, npat, -1).reshape(b, c * npat, -1)
            x = torch.cat([x[:, :1, :], x[:, 1:, :] + te], dim=1)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 0]                                           # CLS token (B, dim)

    return LaBraM


# ------------------------------------------------------------------ loader / API
LABRAM_REPO = "braindecode/labram-pretrained"


def _canonical_channel_index(config: dict) -> dict:
    """Map UPPER(channel name) -> canonical index, from the repo config's chs_info order
    (== LABRAM_CHANNEL_ORDER; the position_embedding has one row per channel + CLS)."""
    chs = config.get("chs_info", [])
    return {c["ch_name"].upper(): i for i, c in enumerate(chs)}


class LaBraMEncoder:
    """Frozen real-LaBraM feature extractor. `from_pretrained()` downloads + loads the
    safetensors (strict), `embed()` turns raw EEG windows into (N, 200) CLS embeddings."""

    def __init__(self, model, canonical_index: dict):
        self.model = model
        self.canonical_index = canonical_index

    @classmethod
    def from_pretrained(cls, repo: str = LABRAM_REPO, allow_download: bool = True):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg_path = hf_hub_download(repo, "config.json")
        with open(cfg_path) as fh:
            config = json.load(fh)
        weights_path = hf_hub_download(repo, "model.safetensors")
        sd = load_file(weights_path)

        LaBraM = _build_modules()
        model = LaBraM()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # strict-equivalent guard: every checkpoint key must be consumed, none missing
        if missing or unexpected:
            raise RuntimeError(
                f"LaBraM state-dict mismatch — missing={list(missing)[:5]} "
                f"unexpected={list(unexpected)[:5]} (vendored forward is out of sync with the "
                f"checkpoint; do NOT use its embeddings until this is zero)")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, _canonical_channel_index(config))

    def input_chans(self, ch_names: Sequence[str]):
        """Canonical indices [CLS=0] + [i+1 for each known channel]. Returns (indices, keep_mask)
        where keep_mask selects the input channels that exist in LaBraM's vocabulary."""
        import torch
        keep, idx = [], [0]
        for j, n in enumerate(ch_names):
            ci = self.canonical_index.get(str(n).upper())
            if ci is not None:
                keep.append(j)
                idx.append(ci + 1)
        return torch.tensor(idx, dtype=torch.long), keep

    def embed(self, eeg: np.ndarray, ch_names: Sequence[str], patch_size: int = 200,
              batch_size: int = 64) -> np.ndarray:
        """eeg: (N, C, L) at 200 Hz (µV/100-scaled). Returns (N, 200) frozen CLS embeddings.
        Channels absent from LaBraM's vocab are dropped; L is trimmed to a whole # of patches."""
        import torch

        eeg = np.asarray(eeg, dtype=np.float32)
        n, c, ell = eeg.shape
        idx, keep = self.input_chans(ch_names)
        if not keep:
            raise RuntimeError("no input channel matched LaBraM's canonical vocabulary")
        npat = ell // patch_size
        if npat < 1:
            raise RuntimeError(f"window length {ell} < patch_size {patch_size}; can't feed LaBraM")
        eeg = eeg[:, keep, :npat * patch_size].reshape(n, len(keep), npat, patch_size)
        out = np.zeros((n, self.model.dim), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb = torch.tensor(eeg[s:s + batch_size])
                out[s:s + batch_size] = self.model(xb, idx).numpy()
        return out


def labram_available() -> bool:
    """True when transformers/torch/safetensors + the weights are reachable (cache or download)."""
    import importlib.util
    for m in ("torch", "safetensors", "huggingface_hub"):
        if importlib.util.find_spec(m) is None:
            return False
    return True
