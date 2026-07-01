"""dvxr.explain.attention_maps — export fusion attention α_m and late-fusion
weights w_m as a tidy CSV (ARCHITECTURE §A5/§A7, Stage 7).
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def attention_table(fusion_output, index=None) -> pd.DataFrame:
    """Tidy long table: one row per (sample, modality) with attention + weight."""
    fo = fusion_output
    n = int(fo.h.shape[0])
    idx = list(index) if index is not None else list(range(n))
    att = fo.attention or {}
    wts = fo.weights or {}
    rows = []
    for m in fo.present:
        a = _to_np(att.get(m))
        w = fo.weights.get(m) if fo.weights else None
        if w is not None:
            w_val = float(w.detach()) if hasattr(w, "detach") else float(w)
        else:
            w_val = np.nan
        for i, s in enumerate(idx):
            rows.append({
                "sample": s, "modality": m,
                "attention": float(a[i]) if a is not None else np.nan,
                "weight": w_val,
            })
    return pd.DataFrame(rows)


def export_attention(fusion_output, out_path: str | pathlib.Path = "outputs/fusion_attention.csv",
                     index=None) -> pd.DataFrame:
    df = attention_table(fusion_output, index=index)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
