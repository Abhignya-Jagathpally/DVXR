"""dvxr.encoders.codebook — vector-quantization codebook (ARCHITECTURE §A3).

``VectorQuantizer`` is a torch ``nn.Module`` implementing nearest-neighbour
codebook lookup with a straight-through estimator, commitment loss, EMA codebook
updates, dead-code reinitialization, batch perplexity, and an optional
Gumbel-softmax soft-assignment path.

``VQBiosignalEncoder`` extends ``NeuralBiosignalEncoder``: the continuous latent
``z`` passes through the quantizer, and the masked-feature reconstruction is
predicted from the *quantized* vector. The public API of the parent
(``fit_transform`` / ``transform`` / ``save`` / ``from_pretrained`` /
``gradient_saliency``) is preserved; new methods expose code indices, quantized
vectors, and codebook usage.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from dvxr.neural_encoders import (
    NeuralBiosignalEncoder,
    _embedding_frame,
    _require_torch,
)


@dataclass
class VQOutput:
    """Result of one quantizer forward pass."""
    quantized: "object"     # straight-through quantized tensor, shape like z
    indices: "object"       # LongTensor of code indices, shape (N,)
    loss: "object"          # scalar tensor: codebook + beta*commitment
    perplexity: "object"    # scalar tensor in (1, K]
    onehot: "object"        # (N, K) assignment matrix


def _build_vector_quantizer(cfg):
    """Return the VectorQuantizer *class* (torch imported lazily)."""
    torch = _require_torch()
    nn = torch.nn

    class VectorQuantizer(nn.Module):
        def __init__(
            self,
            num_codes: int = 512,
            dim: int = 64,
            beta: float = 0.25,
            decay: float = 0.99,
            eps: float = 1e-5,
            dead_threshold: float = 1.0,
            gumbel: bool = False,
            temperature: float = 1.0,
        ):
            super().__init__()
            self.num_codes = num_codes
            self.dim = dim
            self.beta = beta
            self.decay = decay
            self.eps = eps
            self.dead_threshold = dead_threshold
            self.gumbel = gumbel
            self.temperature = temperature

            codebook = torch.randn(num_codes, dim) * (1.0 / max(dim, 1) ** 0.5)
            # Codebook is a BUFFER (updated by EMA, not by autograd).
            self.register_buffer("codebook", codebook)
            self.register_buffer("cluster_size", torch.ones(num_codes))
            self.register_buffer("embed_avg", codebook.clone())

        def forward(self, z, training: Optional[bool] = None) -> VQOutput:
            if training is None:
                training = self.training
            orig_shape = z.shape
            flat = z.reshape(-1, self.dim)                      # (N, d)

            # squared L2 distance to every code
            dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1)
            )                                                   # (N, K)
            indices = dist.argmin(1)                            # (N,)
            onehot = nn.functional.one_hot(indices, self.num_codes).type_as(flat)
            quantized_hard = self.codebook[indices]             # (N, d)

            # NOTE (m1): codebook_loss is DIAGNOSTIC-ONLY. The codebook is an EMA buffer
            # (updated in _ema_update, not by autograd) and quantized_hard indexes it, so
            # this term carries no gradient to any learnable parameter (flat is detached
            # on its side too). Only `commitment` trains the encoder. It is kept for
            # monitoring the codebook-fit magnitude, not as an optimisation objective.
            codebook_loss = (flat.detach() - quantized_hard).pow(2).mean()   # diagnostic
            commitment = (flat - quantized_hard.detach()).pow(2).mean()      # trains encoder
            loss = codebook_loss + self.beta * commitment

            if self.gumbel and training:
                # differentiable soft assignment (alternative path)
                logits = -dist / max(self.temperature, 1e-6)
                soft = nn.functional.softmax(logits, dim=1)
                quantized = soft @ self.codebook
            else:
                # straight-through estimator: grad flows to z unchanged
                quantized = flat + (quantized_hard - flat).detach()

            # perplexity of batch usage
            avg = onehot.mean(0)
            perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

            if training:
                self._ema_update(flat.detach(), onehot.detach())

            return VQOutput(
                quantized=quantized.reshape(orig_shape),
                indices=indices,
                loss=loss,
                perplexity=perplexity,
                onehot=onehot,
            )

        @torch.no_grad()
        def _ema_update(self, flat, onehot) -> None:
            n_i = onehot.sum(0)                                 # (K,)
            self.cluster_size.mul_(self.decay).add_(n_i, alpha=1 - self.decay)
            embed_sum = onehot.t() @ flat                       # (K, d)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            )
            self.codebook.copy_(self.embed_avg / cluster_size.unsqueeze(1))

            # dead-code reinitialization: replace unused codes with live encodings
            dead = self.cluster_size < self.dead_threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0 and flat.shape[0] > 0:
                pick = torch.randint(0, flat.shape[0], (n_dead,))
                self.codebook[dead] = flat[pick]
                self.embed_avg[dead] = flat[pick]
                self.cluster_size[dead] = 1.0

    return VectorQuantizer


def get_vector_quantizer_class():
    """Public accessor for the VectorQuantizer nn.Module class (torch required)."""
    return _build_vector_quantizer(None)


def _build_module(cfg):
    torch = _require_torch()
    nn = torch.nn
    VectorQuantizer = _build_vector_quantizer(cfg)

    class _VQEncoderModule(nn.Module):
        """input_proj/transformer/head names match the parent so the parent's
        ``gradient_saliency`` works unchanged; a decoder reconstructs features
        from the quantized latent."""

        def __init__(self, n_features, embedding_dim, hidden_dim, n_layers,
                     n_heads, num_codes, beta, gumbel, temperature, decay):
            super().__init__()
            self.input_proj = nn.Linear(1, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
                dropout=0.0, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(hidden_dim, embedding_dim)
            self.quantizer = VectorQuantizer(
                num_codes=num_codes, dim=embedding_dim, beta=beta,
                decay=decay, gumbel=gumbel, temperature=temperature)
            self.decoder = nn.Linear(embedding_dim, n_features)

        # aliases so the parent's gradient_saliency (which calls eval_mode/
        # train_mode on the holder) works against this real nn.Module.
        def eval_mode(self):
            self.eval()

        def train_mode(self):
            self.train()

        def encode(self, x):
            """x: (B, F) -> continuous latent z: (B, d)."""
            tokens = self.input_proj(x.unsqueeze(-1))
            enc = self.transformer(tokens)
            return self.head(enc.mean(dim=1))

        def forward(self, x, mask=None):
            x_in = x
            if mask is not None:
                x_in = x.clone()
                x_in[mask] = 0.0
            z = self.encode(x_in)                              # (B, d)
            vq = self.quantizer(z, training=self.training)
            recon = self.decoder(vq.quantized)                 # (B, F)
            return z, vq, recon

    return _VQEncoderModule


class VQBiosignalEncoder(NeuralBiosignalEncoder):
    """NeuralBiosignalEncoder + vector-quantization codebook (§A3)."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        epochs: int = 30,
        seed: int = 7,
        codebook_size: int = 512,
        commitment_beta: float = 0.25,
        gumbel: bool = False,
        temperature: float = 1.0,
        ema_decay: float = 0.99,
        mask_ratio: float = 0.3,
    ):
        super().__init__(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            n_layers=n_layers, n_heads=n_heads, epochs=epochs, seed=seed)
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
        self.gumbel = gumbel
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self._last_perplexity: Optional[float] = None

    # ---------------- fitting ----------------
    def fit_transform(self, frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        torch = _require_torch()
        self._columns = list(columns)
        matrix = frame[columns].to_numpy(dtype=np.float32)
        self._mean = matrix.mean(axis=0)
        self._std = np.where(matrix.std(axis=0) == 0.0, 1.0, matrix.std(axis=0))
        matrix = (matrix - self._mean) / self._std

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Module = _build_module(None)
        self._model = Module(
            n_features=len(columns), embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim, n_layers=self.n_layers,
            n_heads=self.n_heads, num_codes=self.codebook_size,
            beta=self.commitment_beta, gumbel=self.gumbel,
            temperature=self.temperature, decay=self.ema_decay)

        self._train(matrix, torch)
        embeddings = self._encode_matrix(matrix, torch)
        return _embedding_frame(embeddings, frame.index)

    def _train(self, matrix: np.ndarray, torch) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._model.train()
        optim = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        x_all = torch.tensor(matrix, dtype=torch.float32)
        n, f = x_all.shape
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        self._loss_history: List[float] = []
        for _ in range(self.epochs):
            mask = torch.rand(n, f, generator=gen) < self.mask_ratio
            _z, vq, recon = self._model(x_all, mask)
            if mask.any():
                recon_loss = ((recon[mask] - x_all[mask]) ** 2).mean()
            else:
                recon_loss = ((recon - x_all) ** 2).mean()
            loss = recon_loss + vq.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            self._loss_history.append(float(loss.detach()))
            self._last_perplexity = float(vq.perplexity.detach())
        self._model.eval()

    def _encode_matrix(self, matrix: np.ndarray, torch) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(matrix, dtype=torch.float32)
            return self._model.encode(x).numpy()

    # ---------------- quantization outputs ----------------
    def quantize(self, frame: pd.DataFrame):
        """Return (indices DataFrame, quantized-vector DataFrame) for each row."""
        if self._model is None:
            raise RuntimeError("VQBiosignalEncoder must be fitted before quantize().")
        torch = _require_torch()
        aligned = frame.reindex(columns=self._columns, fill_value=0.0)
        matrix = (aligned.to_numpy(dtype=np.float32) - self._mean) / self._std
        self._model.eval()
        with torch.no_grad():
            z = self._model.encode(torch.tensor(matrix, dtype=torch.float32))
            vq = self._model.quantizer(z, training=False)
        idx = pd.DataFrame({"code_index": vq.indices.numpy()}, index=frame.index)
        quant = pd.DataFrame(
            vq.quantized.numpy(), index=frame.index,
            columns=[f"q_{i:02d}" for i in range(self.embedding_dim)])
        return idx, quant

    def codebook_usage(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-code histogram + frequency over the rows of *frame*."""
        idx, _ = self.quantize(frame)
        counts = idx["code_index"].value_counts().sort_index()
        total = int(counts.sum())
        return pd.DataFrame({
            "code_index": counts.index.astype(int),
            "count": counts.values.astype(int),
            "frequency": counts.values / max(total, 1),
        }).reset_index(drop=True)

    def perplexity(self, frame: Optional[pd.DataFrame] = None) -> float:
        """Batch perplexity: last-training value, or recomputed on *frame*."""
        if frame is None:
            if self._last_perplexity is None:
                raise RuntimeError("No perplexity available; fit or pass a frame.")
            return self._last_perplexity
        usage = self.codebook_usage(frame)
        p = usage["frequency"].to_numpy()
        return float(np.exp(-(p * np.log(p + 1e-10)).sum()))

    def export(self, frame: pd.DataFrame, out_dir: str | pathlib.Path = "outputs"):
        """Write outputs/codebook_usage.csv and outputs/latent_quantized.npy."""
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        usage = self.codebook_usage(frame)
        usage.to_csv(out / "codebook_usage.csv", index=False)
        _idx, quant = self.quantize(frame)
        np.save(out / "latent_quantized.npy", quant.to_numpy())
        return out / "codebook_usage.csv", out / "latent_quantized.npy"

    # ---------------- persistence ----------------
    def save(self, path: str | pathlib.Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted VQBiosignalEncoder.")
        torch = _require_torch()
        torch.save({
            "state_dict": self._model.state_dict(),
            "columns": self._columns, "mean": self._mean, "std": self._std,
            "hparams": {
                "embedding_dim": self.embedding_dim, "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers, "n_heads": self.n_heads,
                "epochs": self.epochs, "seed": self.seed,
                "codebook_size": self.codebook_size,
                "commitment_beta": self.commitment_beta, "gumbel": self.gumbel,
                "temperature": self.temperature, "ema_decay": self.ema_decay,
                "mask_ratio": self.mask_ratio,
            },
        }, str(path))

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path, **kwargs) -> "VQBiosignalEncoder":
        torch = _require_torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        enc = cls(**{**payload["hparams"], **kwargs})
        enc._columns = payload["columns"]
        enc._mean, enc._std = payload["mean"], payload["std"]
        torch.manual_seed(enc.seed)
        Module = _build_module(None)
        enc._model = Module(
            n_features=len(enc._columns), embedding_dim=enc.embedding_dim,
            hidden_dim=enc.hidden_dim, n_layers=enc.n_layers, n_heads=enc.n_heads,
            num_codes=enc.codebook_size, beta=enc.commitment_beta,
            gumbel=enc.gumbel, temperature=enc.temperature, decay=enc.ema_decay)
        enc._model.load_state_dict(payload["state_dict"])
        enc._model.eval()
        return enc
