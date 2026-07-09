"""dvxr.bench.raw_seq — the raw-signal lever: a small multimodal 1D-CNN over RAW windows,
vs the summary-stat GBM floor. This is the honest path to beating SOTA that the tiny
summary-stat tasks could not offer: on Sleep-EDF the raw waveform carries structure
(spindles, K-complexes, REM eye movements) that per-window summary statistics discard, so a
convolutional sequence encoder can beat a tuned GBM on the same subjects.

The CNN reads ``task.extra["raw"][m]`` (per-modality (N, C*L) downsampled raw windows), runs
a per-modality Conv1d stack, concatenates the modality embeddings, and a linear head predicts
— trained end-to-end on TRAIN indices only (no leakage), scored on held-out subjects. Compared
head-to-head against the same GBM/linear floor the rest of the bench uses. torch-guarded.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from dvxr.bench.baselines import error_metric, pred_classical, pred_xgboost
from dvxr.bench.protocol import relativity, repeated_group_folds
from dvxr.bench.tasks import BenchTask


def _raw_channels(task: BenchTask, m: str) -> int:
    """#channels packed into the flat raw vector for modality m (eeg=2, others=1)."""
    return 2 if m == "eeg" else 1


def _build_rawcnn(task: BenchTask, n_classes: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mods = task.modalities

    class RawCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = mods
            self.branches = nn.ModuleDict()
            emb = 0
            for m in mods:
                c = _raw_channels(task, m)
                self.branches[m] = nn.Sequential(
                    nn.Conv1d(c, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                    nn.BatchNorm1d(16),
                    nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.AdaptiveAvgPool1d(1))
                emb += 32
            self.head = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(64, n_classes))

        def forward(self, raw: Dict[str, "torch.Tensor"]):
            zs = []
            for m in self.mods:
                x = raw[m]                       # (B, C, L)
                zs.append(self.branches[m](x).squeeze(-1))
            return self.head(torch.cat(zs, dim=1))

    return RawCNN()


def _raw_tensors(task: BenchTask):
    import torch
    out = {}
    for m in task.modalities:
        arr = task.extra["raw"][m]               # (N, C*L)
        c = _raw_channels(task, m)
        n, w = arr.shape
        out[m] = torch.tensor(arr.reshape(n, c, w // c), dtype=torch.float32)
    return out


def pred_rawcnn(task: BenchTask, tr, te, seed: int = 7, epochs: int = 25,
                lr: float = 1e-3, batch_size: int = 128) -> np.ndarray:
    """Train the raw-signal multimodal CNN on TRAIN; return test-fold P(class=1)."""
    import torch
    from torch import nn

    y = np.asarray(task.y, dtype=int)
    classes = sorted(np.unique(y[tr]).tolist())
    if len(classes) < 2:
        return np.full(len(te), float(classes[0]))
    raw = _raw_tensors(task)
    # per-modality standardisation on train only (raw µV scales differ across modalities)
    stats = {m: (raw[m][tr].mean(), raw[m][tr].std().clamp_min(1e-6)) for m in task.modalities}
    raw = {m: (raw[m] - mu) / sd for m, (mu, sd) in stats.items()}

    cls_index = {c: i for i, c in enumerate(classes)}
    ytr = torch.tensor([cls_index[int(v)] for v in y[tr]])
    counts = np.array([(y[tr] == c).sum() for c in classes], dtype=float)
    w = torch.tensor(counts.sum() / (len(classes) * np.clip(counts, 1, None)), dtype=torch.float32)

    model = _build_rawcnn(task, len(classes), seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=w)
    tr = np.asarray(tr)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(tr), batch_size):
            idx = tr[order[s:s + batch_size]]
            batch = {m: raw[m][idx] for m in task.modalities}
            logits = model(batch)
            loss = lossf(logits, ytr[order[s:s + batch_size]])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    te = np.asarray(te)
    pos = cls_index.get(1, len(classes) - 1)
    out = np.zeros(len(te))
    with torch.no_grad():
        for s in range(0, len(te), batch_size):
            idx = te[s:s + batch_size]
            batch = {m: raw[m][idx] for m in task.modalities}
            out[s:s + batch_size] = torch.softmax(model(batch), dim=1)[:, pos].numpy()
    return out


def _importable(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def sleep_win_benchmark(task: BenchTask, seed: int = 7, n_repeats: int = 2,
                        n_folds: int = 5, epochs: int = 25) -> dict:
    """Head-to-head under held-out-subject CV: the raw-signal CNN (proposed deep) vs the
    summary-stat floor (xgboost if available, else HistGBM). Returns per-model mean 1-AUROC,
    plus the RER + bootstrap CI of the proposed model vs the floor. A win = RER>0, CI low >0."""
    folds = repeated_group_folds(task.subject_ids, n_repeats, n_folds, seed)
    floor_name = "xgboost" if _importable("xgboost") else "classical_gbm"
    floor_fn = pred_xgboost if floor_name == "xgboost" else pred_classical

    cnn_err: List[float] = []
    floor_err: List[float] = []
    for tr, te in folds:
        yte = task.y[te]
        cnn_err.append(error_metric(task, yte, pred_rawcnn(task, tr, te, seed=seed, epochs=epochs)))
        floor_err.append(error_metric(task, yte, floor_fn(task, tr, te, seed=seed)))
    rel = relativity(task.name, task.metric, floor_name, cnn_err, floor_err, seed=seed)
    win = bool(rel.rer_pct > 0 and rel.rer_ci[0] > 0)
    return {"task": task.name, "target": task.extra.get("target", ""),
            "floor": floor_name, "rawcnn_err": float(np.nanmean(cnn_err)),
            "floor_err": float(np.nanmean(floor_err)), "rer_pct": rel.rer_pct,
            "rer_ci": list(rel.rer_ci), "p_wilcoxon": rel.p_wilcoxon,
            "n_folds": rel.n_folds, "win": win}
