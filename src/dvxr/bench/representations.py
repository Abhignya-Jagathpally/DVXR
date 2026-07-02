"""dvxr.bench.representations — {raw, pca, neural, vq, fused} -> one shared head.

This is the fix for the "encoder never feeds the head" blocker: every
representation is a swappable feature producer, fit on TRAIN ONLY, and scored by
the SAME shared head (logistic for classification, ridge for regression). The
"fused" representation is the CACMF encoder+VQ+cross-modal-fusion trained on the
train fold; its joint latent h is then probed by the same head. So all
configurations are compared on equal footing and the proposed model is actually
evaluated.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from dvxr.bench.tasks import BenchTask


# --------------------------------------------------------------- shared head
def _fit_head(kind: str, Xtr, ytr, Xte, seed: int = 7):
    """One shared head for every representation. Returns test predictions."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "classification":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=seed).fit(Xtr, ytr)
        classes = clf.classes_
        proba = clf.predict_proba(Xte)
        if len(classes) == 1:                       # degenerate train fold
            out = np.full(len(Xte), float(classes[0]))
            return out
        pos = list(classes).index(1) if 1 in classes else 1
        return proba[:, pos]
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=seed).fit(Xtr, ytr).predict(Xte)


def _concat(task: BenchTask) -> np.ndarray:
    return np.hstack([task.features[m] for m in task.modalities])


def _as_frame(arr: np.ndarray) -> Tuple[pd.DataFrame, list]:
    cols = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols), cols


# ------------------------------------------------------- representation makers
# Each returns (X_train, X_test), fit on train indices only.
def rep_raw(task, tr, te, seed=7):
    X = _concat(task)
    return X[tr], X[te]


def rep_pca(task, tr, te, seed=7):
    from dvxr.encoders.baseline import FeatureEncoder
    X = _concat(task)
    df, cols = _as_frame(X)
    enc = FeatureEncoder(max_components=min(24, X.shape[1]))
    emb_tr = enc.fit_transform(df.iloc[tr].reset_index(drop=True), cols).to_numpy()
    emb_te = enc.transform(df.iloc[te].reset_index(drop=True)).to_numpy()
    return emb_tr, emb_te


def rep_neural(task, tr, te, seed=7):
    from dvxr.neural_encoders import NeuralBiosignalEncoder
    X = _concat(task)
    df, cols = _as_frame(X)
    enc = NeuralBiosignalEncoder(embedding_dim=16, hidden_dim=32, n_layers=1,
                                 n_heads=2, epochs=12, seed=seed)
    emb_tr = enc.fit_transform(df.iloc[tr].reset_index(drop=True), cols).to_numpy()
    emb_te = enc.transform(df.iloc[te].reset_index(drop=True)).to_numpy()
    return emb_tr, emb_te


def rep_vq(task, tr, te, seed=7):
    from dvxr.encoders.codebook import VQBiosignalEncoder
    X = _concat(task)
    df, cols = _as_frame(X)
    enc = VQBiosignalEncoder(embedding_dim=16, hidden_dim=32, n_layers=1,
                             n_heads=2, epochs=10, codebook_size=64, seed=seed)
    emb_tr = enc.fit_transform(df.iloc[tr].reset_index(drop=True), cols).to_numpy()
    emb_te = enc.transform(df.iloc[te].reset_index(drop=True)).to_numpy()
    return emb_tr, emb_te


def _train_fused(task, tr, seed=7, epochs=25):
    """Train the CACMF encoder+VQ+cross-modal fusion+head end-to-end on train fold.

    Returns (model, feats_all_tensors). Standardisation is fit on train only.
    """
    import torch

    from dvxr.config import DEFAULTS
    from dvxr.tasks.model import build_multitask_model
    from dvxr.tasks.train import train_multitask
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    mods = task.modalities
    scalers = {m: StandardScaler().fit(task.features[m][tr]) for m in mods}
    feats_all = {m: scalers[m].transform(task.features[m]) for m in mods}
    input_dims = {m: feats_all[m].shape[1] for m in mods}

    cfg = DEFAULTS.with_(d=24, d_f=48, n_heads=3, n_fusion_layers=2,
                         codebook_size=128, epochs=epochs,
                         fusion_strategy="cross_modal", seed=seed)
    is_cls = task.kind == "classification"
    model = build_multitask_model(cfg, input_dims,
                                  classification_tasks=[task.name] if is_cls else [])
    ftr = {m: torch.tensor(feats_all[m][tr], dtype=torch.float32) for m in mods}
    labels, forecast = {}, None
    y_mu, y_sd = 0.0, 1.0
    if is_cls:
        labels = {task.name: torch.tensor(task.y[tr]).long()}
    else:
        # standardise the regression target so the linear forecast head can reach
        # its scale (raw mg/dL otherwise leaves the head badly mis-scaled).
        y_mu = float(np.mean(task.y[tr]))
        y_sd = float(np.std(task.y[tr])) or 1.0
        forecast = torch.tensor((task.y[tr] - y_mu) / y_sd, dtype=torch.float32)
    train_multitask(model, ftr, labels, forecast_target=forecast, config=cfg,
                    log_path="outputs/_bench_train_log.csv")
    model.eval()
    f_all = {m: torch.tensor(feats_all[m], dtype=torch.float32) for m in mods}
    return model, f_all, (y_mu, y_sd)


def rep_fused(task, tr, te, seed=7):
    """CACMF fused joint latent h as a swappable representation (probe with shared head).

    The proposed model: the trained encoder+VQ+cross-modal fusion produces h, which
    the shared head is trained on (so the encoder genuinely feeds the head).
    """
    import torch
    model, f_all, _ = _train_fused(task, tr, seed=seed)
    with torch.no_grad():
        h = model(f_all)["h"].numpy()
    return h[tr], h[te]


def pred_fused_e2e(task, tr, te, seed=7):
    """Transparency config: CACMF predicting through its OWN end-to-end head."""
    import torch
    model, f_all, (y_mu, y_sd) = _train_fused(task, tr, seed=seed)
    with torch.no_grad():
        if task.kind == "classification":
            pred = model.probabilities(f_all)[task.name][:, 1].numpy()
        else:
            pred = model(f_all)["forecast"].numpy() * y_sd + y_mu
    return pred[te]


REPRESENTATIONS: Dict[str, Callable] = {
    "raw": rep_raw, "pca": rep_pca, "neural": rep_neural,
    "vq": rep_vq, "fused": rep_fused,
}


def evaluate_representation(task: BenchTask, name: str, tr, te, seed: int = 7):
    """Fit representation on train, shared head on train, return test predictions."""
    Xtr, Xte = REPRESENTATIONS[name](task, tr, te, seed=seed)
    return _fit_head(task.kind, Xtr, task.y[tr], Xte, seed=seed)
