from __future__ import annotations

"""BIOT-style self-supervised transformer encoder for biosignal feature windows.

This module provides ``NeuralBiosignalEncoder``, a drop-in replacement for
``FeatureEncoder`` that learns representations via a masked-feature
reconstruction objective and returns fixed-size embeddings as a DataFrame with
columns ``embed_00 ... embed_{embedding_dim-1}``.
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required by NeuralBiosignalEncoder but could not be imported. "
            "Install it with: pip install torch"
        ) from exc
    return torch


def _embedding_frame(encoded: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        encoded,
        index=index,
        columns=[f"embed_{i:02d}" for i in range(encoded.shape[1])],
    )


class _BIOTStyleEncoder:
    """Internal PyTorch module holder (built lazily to avoid import at module level)."""

    def __init__(
        self,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
    ):
        torch = _require_torch()
        nn = torch.nn

        # Project each scalar feature to a token of size hidden_dim.
        # We treat each feature as an independent "channel" (BIOT paradigm).
        self.input_proj = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Mean-pool then project to embedding_dim
        self.head = nn.Linear(hidden_dim, embedding_dim)

        # Reconstruction head for masked-feature pretraining
        self.recon_head = nn.Linear(hidden_dim, 1)

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Collect all modules for parameter iteration
        self._modules_list = [
            self.input_proj,
            self.transformer,
            self.head,
            self.recon_head,
        ]

    def parameters(self):
        for m in self._modules_list:
            yield from m.parameters()

    def encode(self, x):
        """x: (batch, n_features) float tensor -> (batch, embedding_dim)."""
        tokens = x.unsqueeze(-1)                          # (B, F, 1)
        tokens = self.input_proj(tokens)                   # (B, F, hidden_dim)
        enc = self.transformer(tokens)                     # (B, F, hidden_dim)
        pooled = enc.mean(dim=1)                           # (B, hidden_dim)
        return self.head(pooled)                           # (B, embedding_dim)

    def reconstruct_masked(self, x, mask):
        """
        x    : (B, F) normalized input
        mask : (B, F) bool tensor, True = masked (to be reconstructed)
        returns (B, F) reconstruction for masked positions
        """
        x_masked = x.clone()
        x_masked[mask] = 0.0                               # zero-out masked tokens
        tokens = x_masked.unsqueeze(-1)                    # (B, F, 1)
        tokens = self.input_proj(tokens)                   # (B, F, hidden_dim)
        enc = self.transformer(tokens)                     # (B, F, hidden_dim)
        recon = self.recon_head(enc).squeeze(-1)           # (B, F)
        return recon

    def train_mode(self):
        for m in self._modules_list:
            m.train()

    def eval_mode(self):
        for m in self._modules_list:
            m.eval()

    def state_dict(self) -> dict:
        sd = {}
        names = ["input_proj", "transformer", "head", "recon_head"]
        for name, m in zip(names, self._modules_list):
            for k, v in m.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    def load_state_dict(self, sd: dict) -> None:
        names = ["input_proj", "transformer", "head", "recon_head"]
        for name, m in zip(names, self._modules_list):
            prefix = f"{name}."
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            m.load_state_dict(sub)


class NeuralBiosignalEncoder:
    """BIOT-style transformer encoder for wearable biosignal feature windows.

    Implements the same API as ``FeatureEncoder``:

    - ``fit_transform(frame, columns) -> pd.DataFrame``
    - ``transform(frame) -> pd.DataFrame``

    Additionally provides:

    - ``save(path)`` / ``from_pretrained(cls, path)``
    - ``gradient_saliency(frame, columns) -> pd.DataFrame``

    Parameters
    ----------
    embedding_dim : int
        Size of the output embedding vector per window.
    hidden_dim : int
        Internal transformer token/hidden size.
    n_layers : int
        Number of TransformerEncoder layers.
    n_heads : int
        Number of attention heads (``hidden_dim`` must be divisible by this).
    epochs : int
        Number of self-supervised pretraining epochs.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        epochs: int = 30,
        seed: int = 7,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.epochs = epochs
        self.seed = seed

        self._columns: list[str] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._model: Optional[_BIOTStyleEncoder] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Fit the encoder on *frame[columns]* and return embeddings.

        Parameters
        ----------
        frame : pd.DataFrame
            Input window-level feature DataFrame.
        columns : list[str]
            Feature column names to encode.

        Returns
        -------
        pd.DataFrame
            Embedding DataFrame with columns ``embed_00 ... embed_{embedding_dim-1}``
            and the same index as *frame*.
        """
        torch = _require_torch()

        self._columns = list(columns)
        matrix = frame[columns].to_numpy(dtype=np.float32)

        # Fit per-feature standardisation
        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)
        self._std = np.where(self._std == 0.0, 1.0, self._std)  # avoid div/0
        matrix = (matrix - self._mean) / self._std

        n_features = len(columns)
        # Seed BEFORE module construction so weight initialisation is deterministic.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._model = _BIOTStyleEncoder(
            n_features=n_features,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        )

        self._train(matrix, torch)

        # Encode all rows
        embeddings = self._encode_matrix(matrix, torch)
        return _embedding_frame(embeddings, frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode *frame* using the already-fitted normalization and model.

        Parameters
        ----------
        frame : pd.DataFrame
            Input window-level feature DataFrame. Must contain the same columns
            as those used during ``fit_transform``.

        Returns
        -------
        pd.DataFrame
            Embedding DataFrame with columns ``embed_00 ... embed_{embedding_dim-1}``
            and the same index as *frame*.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if self._model is None:
            raise RuntimeError(
                "NeuralBiosignalEncoder must be fitted via fit_transform before calling transform."
            )
        torch = _require_torch()

        aligned = frame.reindex(columns=self._columns, fill_value=0.0)
        matrix = aligned.to_numpy(dtype=np.float32)
        matrix = (matrix - self._mean) / self._std

        embeddings = self._encode_matrix(matrix, torch)
        return _embedding_frame(embeddings, frame.index)

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the encoder to *path* (torch .pt file).

        Parameters
        ----------
        path : str or Path
            Destination file path (typically ``*.pt``).
        """
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted NeuralBiosignalEncoder.")
        torch = _require_torch()
        payload = {
            "state_dict": self._model.state_dict(),
            "columns": self._columns,
            "mean": self._mean,
            "std": self._std,
            "hparams": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "epochs": self.epochs,
                "seed": self.seed,
            },
        }
        torch.save(payload, str(path))

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path, **kwargs) -> "NeuralBiosignalEncoder":
        """Load a previously saved encoder.

        Parameters
        ----------
        path : str or Path
            Path to a file created by :meth:`save`.
        **kwargs
            Any keyword overrides for hyperparameters (rarely needed).

        Returns
        -------
        NeuralBiosignalEncoder
            Fitted encoder ready for ``transform`` and ``gradient_saliency``.
        """
        torch = _require_torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        hparams = {**payload["hparams"], **kwargs}
        enc = cls(**hparams)
        enc._columns = payload["columns"]
        enc._mean = payload["mean"]
        enc._std = payload["std"]
        n_features = len(enc._columns)
        enc._model = _BIOTStyleEncoder(
            n_features=n_features,
            embedding_dim=enc.embedding_dim,
            hidden_dim=enc.hidden_dim,
            n_layers=enc.n_layers,
            n_heads=enc.n_heads,
        )
        enc._model.load_state_dict(payload["state_dict"])
        return enc

    def gradient_saliency(
        self,
        frame: pd.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Per-feature attribution via gradient of the embedding L2-norm.

        For each input row the saliency of feature *j* is
        ``|d ||embed||_2 / d x_j|``.

        Parameters
        ----------
        frame : pd.DataFrame
            Input window-level feature DataFrame.
        columns : list[str], optional
            Feature columns to use. Defaults to the columns seen during
            ``fit_transform``.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(frame), len(columns))`` with original feature names as
            columns and the same index as *frame*.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if self._model is None:
            raise RuntimeError(
                "NeuralBiosignalEncoder must be fitted via fit_transform before calling gradient_saliency."
            )
        torch = _require_torch()

        if columns is None:
            columns = self._columns

        aligned = frame.reindex(columns=columns, fill_value=0.0)
        matrix = aligned.to_numpy(dtype=np.float32)

        # Normalise using stored scaler, mapping only the requested columns
        col_idx = [self._columns.index(c) for c in columns]
        mean = self._mean[col_idx]
        std = self._std[col_idx]
        matrix = (matrix - mean) / std

        x = torch.tensor(matrix, dtype=torch.float32, requires_grad=True)

        self._model.eval_mode()
        tokens = x.unsqueeze(-1)                           # (B, F, 1)
        tokens_proj = self._model.input_proj(tokens)       # (B, F, hidden)
        enc_out = self._model.transformer(tokens_proj)     # (B, F, hidden)
        pooled = enc_out.mean(dim=1)                       # (B, hidden)
        embed = self._model.head(pooled)                   # (B, embedding_dim)
        norm = embed.norm(dim=1)                           # (B,)
        norm.sum().backward()

        saliency = x.grad.detach().abs().numpy()           # (B, F)
        return pd.DataFrame(saliency, index=frame.index, columns=columns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(self, matrix: np.ndarray, torch) -> None:
        """Self-supervised masked-feature reconstruction training."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        optim = torch.optim.Adam(list(self._model.parameters()), lr=1e-3)
        x_all = torch.tensor(matrix, dtype=torch.float32)
        n, f = x_all.shape
        mask_ratio = 0.3

        self._model.train_mode()
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        for _ in range(self.epochs):
            # Random mask for all rows
            mask = torch.rand(n, f, generator=rng) < mask_ratio  # (N, F) bool

            recon = self._model.reconstruct_masked(x_all, mask)  # (N, F)
            # Only compute loss on masked positions
            loss = ((recon[mask] - x_all[mask]) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        self._model.eval_mode()

    def _encode_matrix(self, matrix: np.ndarray, torch) -> np.ndarray:
        """Encode a normalised numpy matrix to embeddings."""
        self._model.eval_mode()
        with torch.no_grad():
            x = torch.tensor(matrix, dtype=torch.float32)
            embed = self._model.encode(x)
            return embed.numpy()
