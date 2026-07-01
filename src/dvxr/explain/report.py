"""dvxr.explain.report — one-call explanation bundle (ARCHITECTURE §A5/§A7, Stage 7).

``explain_prediction`` returns four blocks for a single prediction — physiological
biomarkers, neural saliency, modality attention, and active codebook entries — and
writes a human-readable ``outputs/explanation_example.md``. Reuses the existing
``biomarkers.physiological_biomarkers`` and ``neural_biomarker_saliency``.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import pandas as pd

from dvxr.biomarkers import neural_biomarker_saliency, physiological_biomarkers
from dvxr.explain.attention_maps import attention_table
from dvxr.explain.codebook_usage import codebook_histogram, codebook_perplexity


def _df_to_md(df: Optional[pd.DataFrame], max_rows: int = 12) -> str:
    if df is None or len(df) == 0:
        return "_none_\n"
    try:
        return df.head(max_rows).to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + df.head(max_rows).to_string(index=False) + "\n```\n"


def explain_prediction(
    events: Optional[pd.DataFrame] = None,
    cacmf_model=None,
    latents: Optional[Dict[str, "object"]] = None,
    feature_frame: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    out_path: str | pathlib.Path = "outputs/explanation_example.md",
    top_n: int = 10,
) -> dict:
    """Assemble the four explanation blocks (whichever inputs are provided)."""
    blocks: dict = {"physiological_biomarkers": None, "neural_saliency": None,
                    "modality_attention": None, "active_codes": None,
                    "codebook_perplexity": None}

    if events is not None:
        blocks["physiological_biomarkers"] = physiological_biomarkers(events)

    if feature_frame is not None and feature_columns:
        blocks["neural_saliency"] = neural_biomarker_saliency(
            feature_frame, feature_columns, top_n=top_n)

    if cacmf_model is not None and latents is not None:
        fo = cacmf_model.fuse(latents)
        blocks["modality_attention"] = attention_table(fo)
        codes = {m: cacmf_model._last_codes[m] for m in cacmf_model._last_codes}
        if codes:
            blocks["active_codes"] = codebook_histogram(codes)
            blocks["codebook_perplexity"] = codebook_perplexity(codes)

    _write_markdown(blocks, pathlib.Path(out_path))
    return blocks


def _write_markdown(blocks: dict, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CACMF Prediction Explanation\n",
             "One prediction explained across four auditable views. Attention/codes "
             "are associations, not causal claims.\n"]
    lines.append("\n## 1. Physiological biomarkers\n")
    lines.append(_df_to_md(blocks["physiological_biomarkers"]))
    lines.append("\n## 2. Neural saliency (top features)\n")
    lines.append(_df_to_md(blocks["neural_saliency"]))
    lines.append("\n## 3. Modality attention / fusion weights\n")
    lines.append(_df_to_md(blocks["modality_attention"]))
    lines.append("\n## 4. Active codebook entries\n")
    lines.append(_df_to_md(blocks["active_codes"]))
    if blocks.get("codebook_perplexity"):
        pp = ", ".join(f"{m}={v:.2f}" for m, v in blocks["codebook_perplexity"].items())
        lines.append(f"\nCodebook perplexity: {pp}\n")
    out.write_text("\n".join(lines))
