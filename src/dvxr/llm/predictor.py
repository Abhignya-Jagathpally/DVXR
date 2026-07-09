"""dvxr.llm.predictor — the LLM in the PREDICTIVE path (Option 3: soft-prompt fusion).

Each modality's VQ codebook tokens are projected into a frozen causal-LM's embedding
space and prepended as soft-prompt tokens; the LLM reads them jointly with a short
grounded text prompt, and its pooled hidden state is the multimodal representation a
calibrated head predicts from. This makes CACMF's VQ codebook a *cross-modal tokenizer*
that lets a frozen LLM read BCI + wearable + CGM + EHR together (NeuroLM-style).

Design for this repo (offline/CPU/deterministic):
  * Frozen LLM, default CPU-runnable ``Qwen/Qwen2.5-0.5B-Instruct``; ``DVXR_LLM_PREDICTOR``
    env overrides (e.g. ``Qwen/Qwen2.5-7B-Instruct`` on a GPU — auto-detected).
  * Per-modality soft tokens from ``VQBiosignalEncoder.quantize`` (discrete codes) → a
    seeded linear projection into the LLM hidden dim. A missing modality uses a learned
    absent token → **interoperability** (arbitrary modality subsets at test time).
  * The frozen LLM forward is run once per window and cached (label-free, no leakage), then
    the shared bench head trains on train indices only — same contract as the SOTA opponent.
    A trainable-projection + LoRA variant (the full Option 3) is gated behind ``trainable``
    / a GPU and is documented but not run on CPU here.
  * Prediction and explanation stay separate: the head originates the calibrated number;
    ``llm/insight.py`` only narrates it.

Everything is import-guarded: no transformers/torch or no local weights → callers skip.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
SOFT_PROMPT_PREFIX = "Assess the patient's clinical risk from these fused physiological signals:"


def resolve_model_id() -> str:
    return os.environ.get("DVXR_LLM_PREDICTOR", DEFAULT_LLM)


def _seeded_matrix(rows: int, cols: int, seed: int, tag: str) -> np.ndarray:
    """Deterministic small-scale projection matrix (frozen; the head does the learning)."""
    h = int(hashlib.sha256(f"{tag}:{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    return rng.normal(0.0, 1.0 / np.sqrt(cols), size=(rows, cols)).astype(np.float32)


@dataclass
class SoftPromptReader:
    """Frozen causal LM that reads per-modality soft tokens + a text prompt."""

    model_id: str
    d_code: int = 24
    seed: int = 7
    _model: object = field(default=None, repr=False)
    _tok: object = field(default=None, repr=False)
    hidden: int = 0
    device: str = "cpu"
    _proj: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _absent: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def load(self) -> "SoftPromptReader":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, output_hidden_states=True)
        self._model.eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)  # frozen
        self.hidden = int(self._model.config.hidden_size)
        return self

    def _project(self, modality: str, quant: np.ndarray) -> np.ndarray:
        if modality not in self._proj:
            self._proj[modality] = _seeded_matrix(self.hidden, quant.shape[1], self.seed, f"proj:{modality}")
            self._absent[modality] = _seeded_matrix(1, self.hidden, self.seed, f"absent:{modality}")[0]
        return quant @ self._proj[modality].T  # (N, hidden)

    def encode(self, quant_by_mod: Dict[str, np.ndarray], present: Dict[str, bool],
               batch_size: int = 16) -> np.ndarray:
        """Return the frozen LLM's pooled hidden state per row given per-modality soft tokens.

        ``quant_by_mod[m]`` is (N, d_code) quantized vectors; ``present[m]`` False → the
        learned absent token is used for every row of that modality (missing-modality path).
        """
        import torch

        mods = list(quant_by_mod.keys())
        n = len(next(iter(quant_by_mod.values())))
        # soft tokens per modality → (N, n_mod, hidden)
        soft = np.zeros((n, len(mods), self.hidden), dtype=np.float32)
        for j, m in enumerate(mods):
            if present.get(m, True):
                soft[:, j, :] = self._project(m, quant_by_mod[m])
            else:
                self._project(m, quant_by_mod[m])  # ensure absent token exists
                soft[:, j, :] = self._absent[m]

        # text prompt embeddings (shared across rows)
        embed_layer = self._model.get_input_embeddings()
        ids = self._tok(SOFT_PROMPT_PREFIX, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            prompt_emb = embed_layer(ids)[0]  # (T, hidden)

        out = np.zeros((n, self.hidden), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = end - start
            st = torch.tensor(soft[start:end], dtype=prompt_emb.dtype, device=self.device)  # (b, n_mod, h)
            pe = prompt_emb.unsqueeze(0).expand(b, -1, -1)  # (b, T, h)
            inp = torch.cat([st, pe], dim=1)  # soft tokens first, then prompt
            attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
            with torch.no_grad():
                res = self._model(inputs_embeds=inp, attention_mask=attn)
                last = res.hidden_states[-1]  # (b, L, h)
            out[start:end] = last.mean(dim=1).float().cpu().numpy()  # mean-pool
        return out


# ---- module-level cache: one loaded reader per model id (loads are expensive) ----
_READERS: Dict[str, SoftPromptReader] = {}


def get_reader(d_code: int = 24, seed: int = 7) -> SoftPromptReader:
    mid = resolve_model_id()
    key = f"{mid}:{d_code}:{seed}"
    if key not in _READERS:
        _READERS[key] = SoftPromptReader(mid, d_code=d_code, seed=seed).load()
    return _READERS[key]


def _modality_quant(task, seed: int, d_code: int) -> Dict[str, np.ndarray]:
    """Per-modality VQ quantized vectors for every window (unsupervised, label-free)."""
    from dvxr.encoders.codebook import VQBiosignalEncoder

    out: Dict[str, np.ndarray] = {}
    for m in task.modalities:
        X = task.features[m]
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        enc = VQBiosignalEncoder(embedding_dim=d_code, hidden_dim=32, n_layers=1,
                                 n_heads=2, epochs=8, codebook_size=64, seed=seed)
        enc.fit_transform(df, cols)
        _, quant = enc.quantize(df)
        qcols = [c for c in quant.columns if c.startswith("q_")]
        out[m] = quant[qcols].to_numpy(dtype=np.float32)
    return out


def llm_window_embeddings(task, seed: int = 7, d_code: int = 24,
                          drop: Optional[List[str]] = None) -> np.ndarray:
    """Frozen-LLM pooled embedding per window (cached). ``drop`` lists modalities to
    treat as MISSING (absent token) — used for missing-modality robustness."""
    drop = drop or []
    cache_key = "_llm_emb" if not drop else "_llm_emb_drop:" + ",".join(sorted(drop))
    if cache_key in task.extra:
        return task.extra[cache_key]
    quant = _modality_quant(task, seed, d_code)
    present = {m: (m not in drop) for m in task.modalities}
    reader = get_reader(d_code=d_code, seed=seed)
    emb = reader.encode(quant, present)
    task.extra[cache_key] = emb
    task.extra["_llm_backend"] = reader.model_id
    return emb


def rep_llm(task, tr, te, seed: int = 7):
    """Bench representation: frozen soft-prompted LLM pooled hidden state → shared head.
    Returns (X_train, X_test). Registered as ``rep:llm`` (a 'proposed' config)."""
    emb = llm_window_embeddings(task, seed=seed)
    return emb[tr], emb[te]


def modality_attribution(task, seed: int = 7, d_code: int = 24) -> Dict[str, float]:
    """Interpretability: how much each modality moves the LLM representation — mean L2
    shift of the pooled embedding when that modality is replaced by its absent token."""
    base = llm_window_embeddings(task, seed=seed, d_code=d_code)
    scores: Dict[str, float] = {}
    for m in task.modalities:
        dropped = llm_window_embeddings(task, seed=seed, d_code=d_code, drop=[m])
        scores[m] = float(np.linalg.norm(base - dropped, axis=1).mean())
    total = sum(scores.values()) or 1.0
    return {m: v / total for m, v in scores.items()}


def missing_modality_robustness(task, seed: int = 7, n_repeats: int = 2,
                                n_folds: int = 3, d_code: int = 24) -> List[dict]:
    """Interoperability headline: train the shared head ONCE on full-modality LLM
    embeddings, then test with each modality individually dropped (absent token) at
    TEST time only. The graceful-degradation regime a single-modality model can't do —
    it reports how much test error rises per missing modality, under held-out-subject CV.

    Returns one row per (drop-set) with mean CV error + degradation vs the full model.
    Classification only (uses 1-AUROC via the bench error metric)."""
    from dvxr.bench.baselines import error_metric
    from dvxr.bench.protocol import repeated_group_folds
    from dvxr.bench.representations import _fit_head

    if task.kind != "classification":
        return []
    full = llm_window_embeddings(task, seed=seed, d_code=d_code)
    drop_embs = {m: llm_window_embeddings(task, seed=seed, d_code=d_code, drop=[m])
                 for m in task.modalities}
    folds = repeated_group_folds(task.subject_ids, n_repeats, n_folds, seed)

    def _cv_err(test_emb):
        errs = []
        for tr, te in folds:
            # head is trained on FULL-modality embeddings; only the test view changes
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(full[tr])
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=seed).fit(sc.transform(full[tr]), task.y[tr])
            classes = list(clf.classes_)
            proba = clf.predict_proba(sc.transform(test_emb[te]))
            pos = classes.index(1) if 1 in classes else len(classes) - 1
            errs.append(error_metric(task, task.y[te], proba[:, pos]))
        return float(np.nanmean(errs))

    base_err = _cv_err(full)
    rows = [{"dropped": "none", "err": base_err, "degradation": 0.0}]
    for m in task.modalities:
        e = _cv_err(drop_embs[m])
        rows.append({"dropped": m, "err": e, "degradation": e - base_err})
    return rows
