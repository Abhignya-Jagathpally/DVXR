"""dvxr.eval.paper — auto-fill IEEE paper tables from outputs/ (ARCHITECTURE Goal 4).

Every number in a generated .tex table traces to a file under outputs/. Missing
source files are skipped (recorded in the manifest), never fabricated. No LaTeX
install is required to generate the tables.
"""
from __future__ import annotations

import math
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _esc(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _fmt(v, fmt: str = "%.3f") -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "--"
        return fmt % v
    return _esc(str(v))


def df_to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{_esc(caption)}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _select(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(3)
    return out


def build_paper_tables(outputs_dir: str | pathlib.Path = "outputs",
                       tables_dir: str | pathlib.Path = "paper/tables") -> Dict[str, dict]:
    """Read outputs/* and emit paper/tables/*.tex. Returns a source-traceable manifest."""
    out = pathlib.Path(outputs_dir)
    tdir = pathlib.Path(tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, dict] = {}

    def emit(name: str, df: pd.DataFrame, caption: str, label: str, source: str):
        path = tdir / f"{name}.tex"
        path.write_text(df_to_booktabs(df, caption, label))
        manifest[name] = {"path": str(path), "source": source, "rows": len(df)}

    # 1. ablation (fused vs single vs aggregation)
    abl = out / "ablation_table.csv"
    if abl.exists():
        df = pd.read_csv(abl)
        emit("ablation",
             _select(df, ["task", "config_type", "config_name", "auroc",
                          "f1", "accuracy", "ece", "mae", "coverage"]),
             "CACMF ablation: single-modality vs fusion vs aggregation "
             "(subject-held-out).", "tab:ablation", str(abl))

    # 2. clinical task metrics (from the goal1 pipeline)
    clin = out / "clinical_task_metrics.csv"
    if clin.exists():
        df = pd.read_csv(clin)
        emit("clinical_metrics", _select(df, list(df.columns)[:8]),
             "Clinical task metrics (documented proxies where noted).",
             "tab:clinical", str(clin))

    # 3. codebook perplexity
    cb = out / "codebook_usage.csv"
    if cb.exists():
        df = pd.read_csv(cb)
        if "frequency" in df.columns:
            f = df["frequency"].to_numpy(dtype=float)
            ppl = float(np.exp(-(f * np.log(f + 1e-12)).sum()))
            summ = pd.DataFrame([{"codes_used": int((df["count"] > 0).sum()),
                                  "perplexity": round(ppl, 3)}])
            emit("codebook", summ,
                 "VQ codebook usage (perplexity; higher = more codes used).",
                 "tab:codebook", str(cb))

    (tdir / "MANIFEST.txt").write_text(
        "\n".join(f"{k}.tex  <-  {v['source']}  ({v['rows']} rows)"
                  for k, v in manifest.items()) + "\n")
    return manifest
