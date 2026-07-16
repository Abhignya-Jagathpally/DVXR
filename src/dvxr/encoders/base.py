"""dvxr.encoders.base — the uniform per-modality encoder interface (ARCHITECTURE §A2).

Every adapter exposes ``fit_transform`` / ``transform`` / ``save`` / ``from_pretrained``
(the ``EncoderProtocol``) and produces a fixed-width latent of ``config.d`` columns
``z_00 .. z_{d-1}``. Each adapter tries its real-weight PRIMARY encoder (capability
checked: import-guarded + weight-guarded, and only when ``config.use_real_weights``),
logs which encoder actually ran (``.used_encoder``), and degrades to an always-runnable
FALLBACK with no network and no GPU when the primary is unavailable.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from dvxr.config import CACMFConfig, DEFAULTS

logger = logging.getLogger("dvxr.encoders")


def z_frame(arr, index, d: int) -> pd.DataFrame:
    """Coerce an (n, k) array to exactly ``d`` columns z_00..z_{d-1} (pad/truncate)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, k = arr.shape
    if k < d:
        arr = np.hstack([arr, np.zeros((n, d - k), dtype=np.float32)])
    elif k > d:
        arr = arr[:, :d]
    return pd.DataFrame(arr, index=index, columns=[f"z_{i:02d}" for i in range(d)])


@runtime_checkable
class EncoderProtocol(Protocol):
    used_encoder: str

    def fit_transform(self, frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame: ...
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...
    def save(self, path) -> None: ...
    @classmethod
    def from_pretrained(cls, path) -> "EncoderProtocol": ...


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Backends — each has .name, .fit_transform(frame, cols)->ndarray, .transform, save/load
# --------------------------------------------------------------------------- #

class _PCABackend:
    """Always-runnable PCA baseline (sklearn; no torch, no network)."""
    def __init__(self, d: int, tag: str = "pca_feature"):
        from dvxr.encoders.baseline import FeatureEncoder
        self.name = tag
        self.enc = FeatureEncoder(max_components=d)

    def fit_transform(self, frame, columns):
        return self.enc.fit_transform(frame, columns).to_numpy(dtype=np.float32)

    def transform(self, frame):
        return self.enc.transform(frame).to_numpy(dtype=np.float32)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(self.enc, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        obj = cls(d)
        with open(path, "rb") as fh:
            obj.enc = pickle.load(fh)
        return obj


class _VQBackend:
    """band-power/feature -> VQBiosignalEncoder (torch). §A2 EEG/biosignal fallback."""
    def __init__(self, d: int, epochs: int, K: int, seed: int, tag: str = "vq_biosignal"):
        from dvxr.encoders.codebook import VQBiosignalEncoder
        self.name = tag
        self.enc = VQBiosignalEncoder(
            embedding_dim=d, hidden_dim=max(16, d), n_layers=2, n_heads=4,
            epochs=epochs, codebook_size=K, seed=seed)

    def fit_transform(self, frame, columns):
        return self.enc.fit_transform(frame, columns).to_numpy(dtype=np.float32)

    def transform(self, frame):
        return self.enc.transform(frame).to_numpy(dtype=np.float32)

    def save(self, path):
        self.enc.save(path)

    @classmethod
    def load(cls, path, d):
        from dvxr.encoders.codebook import VQBiosignalEncoder
        obj = cls.__new__(cls)
        obj.name = "vq_biosignal"
        obj.enc = VQBiosignalEncoder.from_pretrained(path)
        return obj


class _CGMSummaryBackend:
    """CGM dynamics summary (mean, CV, MAGE, time-in-range, slope) padded to d.

    Stands in for the conformalized Ridge/state-space fallback's latent summary (§A2).
    """
    def __init__(self, d: int, tag: str = "cgm_summary"):
        from sklearn.preprocessing import StandardScaler
        self.name = tag
        self.d = d
        self.scaler = StandardScaler()
        self.columns: List[str] = []

    @staticmethod
    def _summary(values: np.ndarray) -> np.ndarray:
        # values: (n, F) per-row glucose-like feature series
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        cv = std / (np.abs(mean) + 1e-9)
        diffs = np.abs(np.diff(values, axis=1))
        mage = diffs.mean(axis=1) if diffs.shape[1] else np.zeros_like(mean)
        # time-in-range proxy on the raw feature scale (70..180 if glucose-scaled)
        tir = ((values >= 70) & (values <= 180)).mean(axis=1)
        x = np.arange(values.shape[1], dtype=np.float32)
        xc = x - x.mean()
        denom = (xc ** 2).sum() + 1e-9
        slope = ((values - mean[:, None]) * xc).sum(axis=1) / denom
        return np.column_stack([mean, cv, mage, tir, slope]).astype(np.float32)

    def fit_transform(self, frame, columns):
        self.columns = list(columns)
        s = self._summary(frame[columns].to_numpy(dtype=np.float32))
        return self.scaler.fit_transform(s)

    def transform(self, frame):
        aligned = frame.reindex(columns=self.columns, fill_value=0.0)
        s = self._summary(aligned.to_numpy(dtype=np.float32))
        return self.scaler.transform(s)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"scaler": self.scaler, "columns": self.columns}, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        obj = cls(d)
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        obj.scaler, obj.columns = st["scaler"], st["columns"]
        return obj


class _MomentBackend:
    """REAL weights: MOMENT-1 time-series foundation model (momentfm). CPU-runnable.

    Rows' feature vectors are treated as length-512 univariate series (pad/truncate);
    the model's embedding is projected to d via PCA. Downloads on first use.
    """
    def __init__(self, model_id: str, d: int):
        import torch
        from momentfm import MOMENTPipeline
        self.name = f"moment:{model_id}"
        self.d = d
        self.torch = torch
        self.model = MOMENTPipeline.from_pretrained(
            model_id, model_kwargs={"task_name": "embedding"})
        self.model.init()
        self.model.eval()
        self._proj = None
        self.columns: List[str] = []

    def _embed(self, frame, columns) -> np.ndarray:
        X = frame[columns].to_numpy(dtype=np.float32)
        n, f = X.shape
        L = 512
        series = np.zeros((n, 1, L), dtype=np.float32)
        mask = np.zeros((n, L), dtype=np.float32)
        w = min(f, L)
        series[:, 0, :w] = X[:, :w]
        mask[:, :w] = 1.0
        with self.torch.no_grad():
            out = self.model(
                x_enc=self.torch.tensor(series),
                input_mask=self.torch.tensor(mask))
        return np.asarray(out.embeddings.detach().cpu().numpy(), dtype=np.float32)

    def fit_transform(self, frame, columns):
        from sklearn.decomposition import PCA
        self.columns = list(columns)
        emb = self._embed(frame, columns)
        k = min(self.d, emb.shape[1], max(1, emb.shape[0] - 1))
        self._proj = PCA(n_components=k, random_state=7).fit(emb)
        return self._proj.transform(emb).astype(np.float32)

    def transform(self, frame):
        emb = self._embed(frame, self.columns)
        return self._proj.transform(emb).astype(np.float32)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"proj": self._proj, "columns": self.columns,
                         "model_id": self.name.split(":", 1)[1]}, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        obj = cls(st["model_id"], d)
        obj._proj, obj.columns = st["proj"], st["columns"]
        return obj


class _HFEmbedBackend:
    """REAL weights: a HuggingFace encoder (e.g. Bio_ClinicalBERT / Geneformer) that
    embeds a per-row pseudo-text built from the modality's channel names+values.
    Projected to d via PCA. Downloads on first use.
    """
    def __init__(self, model_id: str, d: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = f"hf:{model_id}"
        self.d = d
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self._proj = None
        self.columns: List[str] = []

    def _texts(self, frame, columns) -> List[str]:
        vals = frame[columns]
        out = []
        for _, row in vals.iterrows():
            out.append(" ".join(f"{c} {row[c]:.2f}" for c in columns))
        return out

    def _embed(self, frame, columns) -> np.ndarray:
        texts = self._texts(frame, columns)
        embs = []
        bs = 16
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
                out = self.model(**enc)
                cls = out.last_hidden_state[:, 0, :]
                embs.append(cls.detach().cpu().numpy())
        return np.vstack(embs).astype(np.float32)

    def fit_transform(self, frame, columns):
        from sklearn.decomposition import PCA
        self.columns = list(columns)
        emb = self._embed(frame, columns)
        k = min(self.d, emb.shape[1], max(1, emb.shape[0] - 1))
        self._proj = PCA(n_components=k, random_state=7).fit(emb)
        return self._proj.transform(emb).astype(np.float32)

    def transform(self, frame):
        emb = self._embed(frame, self.columns)
        return self._proj.transform(emb).astype(np.float32)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"proj": self._proj, "columns": self.columns,
                         "model_id": self.name.split(":", 1)[1]}, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        obj = cls(st["model_id"], d)
        obj._proj, obj.columns = st["proj"], st["columns"]
        return obj


class _ClinicalNotesBackend:
    """REAL weights: a clinical transformer (Bio_ClinicalBERT) over UNSTRUCTURED note
    text. Unlike ``_HFEmbedBackend`` — which synthesizes pseudo-text from numeric
    column name/value pairs — this consumes an actual free-text notes column.

    Clinical reports routinely exceed BERT's 512-token limit, so each note is tokenized
    into <=512-token windows (capped at ``max_chunks``) and the per-window [CLS] vectors
    are MEAN-POOLED into one note embedding, then projected to ``d`` via PCA. Loads from
    the HF cache (offline-friendly: transformers honors HF_HUB_OFFLINE natively).
    """
    def __init__(self, model_id: str, d: int, max_chunks: int = 4, chunk_len: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = f"clinicalnotes:{model_id}"
        self.model_id = model_id
        self.d = d
        self.max_chunks = max_chunks
        self.chunk_len = chunk_len
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self._proj = None
        self.columns: List[str] = []

    @staticmethod
    def _text_column(columns: List[str]) -> str:
        for cand in ("note_text", "text", "note", "notes"):
            if cand in columns:
                return cand
        return columns[0]

    def _texts(self, frame, columns) -> List[str]:
        col = self._text_column(list(columns))
        return frame[col].astype(str).tolist()

    def _embed_one(self, text: str) -> np.ndarray:
        # tokenize the whole note (no special tokens), then slice into <=chunk_len windows
        ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        cls_id, sep_id = self.tok.cls_token_id, self.tok.sep_token_id
        body = self.chunk_len - 2
        windows = [ids[i:i + body] for i in range(0, max(len(ids), 1), body)][: self.max_chunks]
        input_ids, masks = [], []
        for w in windows:
            seq = [cls_id] + w + [sep_id]
            pad = self.chunk_len - len(seq)
            masks.append([1] * len(seq) + [0] * pad)
            input_ids.append(seq + [self.tok.pad_token_id] * pad)
        ii = self.torch.tensor(input_ids)
        am = self.torch.tensor(masks)
        with self.torch.no_grad():
            out = self.model(input_ids=ii, attention_mask=am)
        cls = out.last_hidden_state[:, 0, :]              # (n_windows, hidden)
        return cls.mean(dim=0).detach().cpu().numpy()     # mean-pool windows

    def _embed(self, frame, columns) -> np.ndarray:
        texts = self._texts(frame, columns)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)

    def fit_transform(self, frame, columns):
        from sklearn.decomposition import PCA
        self.columns = list(columns)
        emb = self._embed(frame, columns)
        k = min(self.d, emb.shape[1], max(1, emb.shape[0] - 1))
        self._proj = PCA(n_components=k, random_state=7).fit(emb)
        return self._proj.transform(emb).astype(np.float32)

    def transform(self, frame):
        emb = self._embed(frame, self.columns)
        return self._proj.transform(emb).astype(np.float32)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"proj": self._proj, "columns": self.columns,
                         "model_id": self.model_id}, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        obj = cls(st["model_id"], d)
        obj._proj, obj.columns = st["proj"], st["columns"]
        return obj


class _TfidfSvdBackend:
    """Always-runnable notes FLOOR (sklearn TF-IDF + TruncatedSVD; no torch/transformers,
    no network). Fit on TRAIN only via ``fit_transform``/``transform`` (leak-free per fold).
    """
    def __init__(self, d: int, tag: str = "tfidf_svd"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.name = tag
        self.d = d
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english")
        self.svd = TruncatedSVD(n_components=d, random_state=7)
        self.columns: List[str] = []

    def _texts(self, frame, columns):
        col = _ClinicalNotesBackend._text_column(list(columns))
        return frame[col].astype(str).tolist()

    def fit_transform(self, frame, columns):
        self.columns = list(columns)
        X = self.vec.fit_transform(self._texts(frame, columns))
        k = min(self.d, X.shape[1] - 1, max(1, X.shape[0] - 1))
        self.svd.set_params(n_components=k)
        return self.svd.fit_transform(X).astype(np.float32)

    def transform(self, frame):
        X = self.vec.transform(self._texts(frame, self.columns))
        return self.svd.transform(X).astype(np.float32)

    # convenience for a one-shot embed (fits on the given frame); callers that need a
    # leak-free split should use fit_transform/transform per fold instead.
    def _embed(self, frame, columns):
        return self.fit_transform(frame, columns)

    def save(self, path):
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"vec": self.vec, "svd": self.svd, "columns": self.columns}, fh)

    @classmethod
    def load(cls, path, d):
        import pickle
        obj = cls(d)
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        obj.vec, obj.svd, obj.columns = st["vec"], st["svd"], st["columns"]
        return obj


def clinical_notes_available() -> bool:
    """True when a real clinical-notes transformer can be constructed (torch+transformers)."""
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


_BACKEND_LOADERS = {
    "_PCABackend": _PCABackend, "_VQBackend": _VQBackend,
    "_CGMSummaryBackend": _CGMSummaryBackend,
    "_MomentBackend": _MomentBackend, "_HFEmbedBackend": _HFEmbedBackend,
    "_ClinicalNotesBackend": _ClinicalNotesBackend, "_TfidfSvdBackend": _TfidfSvdBackend,
}


def make_primary_backend(modality: str, cfg: CACMFConfig):
    """Try to construct the REAL-weight primary backend for a modality.

    Returns the backend or None (logged) when weights/deps are unavailable. Only
    called when ``cfg.use_real_weights`` is True, so the offline test path never
    touches the network.
    """
    fm = cfg.foundation_model(modality)
    if fm is None:
        return None
    loader = fm.primary_loader
    try:
        if loader == "momentfm":
            return _MomentBackend(fm.primary_id, cfg.d)
        if loader == "transformers":
            return _HFEmbedBackend(fm.primary_id, cfg.d)
        if loader == "clinical_notes":
            import os
            return _ClinicalNotesBackend(
                os.environ.get("DVXR_EHR_NOTES_MODEL", fm.primary_id), cfg.d)
        if loader in ("braindecode", "chronos", "repo", "local"):
            # These need extra packages/checkpoints not guaranteed here; try fallback id.
            if fm.fallback_loader == "transformers":
                return _HFEmbedBackend(fm.fallback_id, cfg.d)
            if fm.fallback_loader == "momentfm":
                return _MomentBackend(fm.fallback_id, cfg.d)
            logger.info("[%s] primary loader %r needs extra deps/weights; "
                        "using bundled fallback.", modality, loader)
            return None
    except Exception as exc:  # pragma: no cover - network/dep dependent
        logger.warning("[%s] real-weight primary (%s) unavailable: %s; "
                       "falling back.", modality, fm.primary_id, exc)
        return None
    return None


# --------------------------------------------------------------------------- #
# Adapter base
# --------------------------------------------------------------------------- #

class BaseAdapter:
    modality = "base"

    def __init__(self, config: CACMFConfig = DEFAULTS):
        self.config = config
        self.d = config.d
        self.used_encoder = "unset"
        self._backend = None
        self._columns: List[str] = []

    # each adapter overrides this with its always-runnable fallback backend
    def _make_fallback(self):
        if _torch_available():
            return _VQBackend(self.d, epochs=min(self.config.epochs, 20),
                              K=self.config.codebook_size, seed=self.config.seed)
        return _PCABackend(self.d)

    def fit_transform(self, frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        self._columns = list(columns)
        backend = None
        if self.config.use_real_weights:
            backend = make_primary_backend(self.modality, self.config)
        if backend is None:
            backend = self._make_fallback()
        self._backend = backend
        self.used_encoder = backend.name
        logger.info("[%s] encoder = %s", self.modality, backend.name)
        arr = backend.fit_transform(frame, columns)
        return z_frame(arr, frame.index, self.d)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._backend is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before transform.")
        return z_frame(self._backend.transform(frame), frame.index, self.d)

    def save(self, path) -> None:
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(json.dumps({
            "modality": self.modality, "d": self.d,
            "used_encoder": self.used_encoder, "columns": self._columns,
            "backend_cls": type(self._backend).__name__,
        }))
        self._backend.save(str(p / "backend.bin"))

    @classmethod
    def from_pretrained(cls, path, config: Optional[CACMFConfig] = None):
        p = pathlib.Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj = cls(config or DEFAULTS.with_(d=meta["d"]))
        obj._columns = meta["columns"]
        obj.used_encoder = meta["used_encoder"]
        backend_cls = _BACKEND_LOADERS[meta["backend_cls"]]
        obj._backend = backend_cls.load(str(p / "backend.bin"), meta["d"])
        return obj


class ModalityEncoderRegistry:
    """Selects and caches the adapter for each modality from config."""
    def __init__(self, config: CACMFConfig = DEFAULTS):
        self.config = config
        self._cache: dict = {}

    def get(self, modality: str) -> BaseAdapter:
        if modality not in self._cache:
            from dvxr.encoders import ADAPTERS
            if modality not in ADAPTERS:
                raise KeyError(f"no adapter registered for modality {modality!r}")
            self._cache[modality] = ADAPTERS[modality](self.config)
        return self._cache[modality]
