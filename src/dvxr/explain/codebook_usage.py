"""dvxr.explain.codebook_usage — per-modality codebook histogram, perplexity, and
top code indices associated with each positive task label (ARCHITECTURE §A7, Stage 7).

The label association is a simple, auditable frequency lift — NOT a causal claim.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _as_1d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def codebook_histogram(indices: Dict[str, "object"]) -> pd.DataFrame:
    """Per-modality per-code counts. Each modality's counts sum to n_samples."""
    rows = []
    for m, idx in indices.items():
        vals = _as_1d(idx)
        u, c = np.unique(vals, return_counts=True)
        for code, cnt in zip(u.tolist(), c.tolist()):
            rows.append({"modality": m, "code_index": int(code), "count": int(cnt)})
    return pd.DataFrame(rows, columns=["modality", "code_index", "count"])


def _perplexity(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def codebook_perplexity(indices: Dict[str, "object"]) -> Dict[str, float]:
    out = {}
    for m, idx in indices.items():
        _u, c = np.unique(_as_1d(idx), return_counts=True)
        out[m] = _perplexity(c.astype(float))
    return out


def top_codes_per_label(indices, labels, top_n: int = 5) -> pd.DataFrame:
    """For one modality's per-sample code indices + binary labels, rank codes by
    their frequency lift in the positive class: lift = P(code|pos) / P(code).
    """
    codes = _as_1d(indices)
    y = _as_1d(labels).astype(int)
    n = len(codes)
    pos = codes[y == 1]
    rows = []
    for code in np.unique(codes):
        overall = float((codes == code).mean())
        pos_frac = float((pos == code).mean()) if len(pos) else 0.0
        lift = pos_frac / overall if overall > 0 else 0.0
        rows.append({"code_index": int(code),
                     "pos_count": int((pos == code).sum()),
                     "overall_freq": overall, "lift": lift})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["pos_count", "lift"], ascending=False).head(top_n).reset_index(drop=True)
