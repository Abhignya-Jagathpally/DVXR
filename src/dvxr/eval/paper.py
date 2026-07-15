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


def _ci(lo_hi) -> str:
    if not lo_hi:
        return ""
    return f" [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def build_product_tables(tables_dir: str | pathlib.Path = "paper/tables",
                         screener_root: str | pathlib.Path = "outputs/product/screeners"
                         ) -> Dict[str, dict]:
    """Emit the *honest product* tables (screening results, fusion contribution, external SOTA,
    clinical utility) from `dvxr.serve.evidence` + the saved screener manifests. Every number traces
    to `evidence.verify_against_scoreboards()` (window-level, scoreboard-pinned), the screeners'
    subject-held-out aggregation, or the persisted decision curve. Nothing is fabricated.
    """
    import json
    from dvxr.serve.evidence import (PRODUCT_CLAIMS, OUR_METRICS, EXTERNAL_SOTA, METHOD_CLAIMS,
                                     verify_against_scoreboards)
    tdir = pathlib.Path(tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(screener_root)
    manifest: Dict[str, dict] = {}

    # a drift guard: refuse to emit if any headline number no longer traces to its scoreboard
    problems = verify_against_scoreboards()
    if problems:
        raise ValueError("scoreboard drift — refusing to emit paper tables: " + "; ".join(problems))

    def ece_of(task: str):
        man = root / task / "manifest.json"
        if man.exists():
            return json.loads(man.read_text())["heldout"].get("ece")
        return None

    def dca_of(task: str):
        man = root / task / "manifest.json"
        if man.exists():
            return json.loads(man.read_text())["heldout"].get("decision_curve")
        return None

    def emit(name, df, caption, label, source):
        (tdir / f"{name}.tex").write_text(df_to_booktabs(df, caption, label))
        manifest[name] = {"path": str(tdir / f"{name}.tex"), "source": source, "rows": len(df)}

    # 1. headline screening results — both AUROC granularities + calibration
    rows = []
    for c in PRODUCT_CLAIMS:
        om = OUR_METRICS.get(c.task, {})
        subj = om.get("subject_auroc")
        rows.append({
            "Task": c.label, "Encoder": c.encoder,
            "AUROC (window)": f"{c.auroc:.3f}{_ci(c.auroc_ci)}",
            "AUROC (subject)": (f"{subj:.3f}{_ci(om.get('subject_ci'))}" if subj is not None
                                else "within-subject: n/a"),
            "ECE": (f"{ece_of(c.task):.3f}" if ece_of(c.task) is not None else "--"),
            "Protocol / N": (f"{om['protocol']}, n={om['n_subjects']}" if om.get("protocol")
                             else "subject-held-out CV"),
        })
    emit("product_headline", pd.DataFrame(rows),
         "DVXR Screen headline results under subject-held-out cross-validation. Window-level AUROC "
         "traces to the committed scoreboard; subject-level AUROC aggregates each held-out subject's "
         "windows (reported only for single-class-per-subject tasks). Research-grade screening, not "
         "diagnosis.", "tab:headline", "dvxr.serve.evidence.PRODUCT_CLAIMS+OUR_METRICS")

    # 2. fusion contribution — the winning method vs the proposal's own learned CACMF fusion
    frows = []
    for c in PRODUCT_CLAIMS:
        comp = c.comparators
        cacmf = comp.get("learned CACMF fusion")
        winner = max(comp.items(), key=lambda kv: kv[1])
        frows.append({
            "Task": c.task,
            "Best method (ours)": f"{winner[0]} = {winner[1]:.3f}",
            "Learned CACMF": (f"{cacmf:.3f}" if cacmf is not None else "--"),
            "Delta AUROC": (f"+{winner[1]-cacmf:.3f}" if cacmf is not None else "--"),
        })
    emit("fusion_contribution", pd.DataFrame(frows),
         "Reliability-gated do-no-harm late fusion vs the proposal's own learned cross-modal CACMF "
         "fusion. The learned fusion loses on every task; do-no-harm gating beats it on 4 of 6 tasks "
         "in the full library (a nuanced positive, not a clean sweep -- see text).",
         "tab:fusion", "dvxr.serve.evidence.PRODUCT_CLAIMS.comparators+METHOD_CLAIMS")

    # 3. DVXR vs published SOTA — protocol-labeled, DOI-carried
    xrows = []
    for task, ext in EXTERNAL_SOTA.items():
        om = OUR_METRICS.get(task, {})
        ours = f"win {om.get('window_auroc')}" + (
            f" / subj {om['subject_auroc']}" if om.get("subject_auroc") is not None else "")
        for e in ext:
            val = "--" if e.value != e.value else f"{e.value:.3f}"
            xrows.append({
                "Cohort": task, "DVXR (AUROC)": ours,
                "Published": e.method, "Score": f"{val} {e.metric}",
                "Protocol": e.protocol, "DOI": e.doi,
            })
    emit("external_sota", pd.DataFrame(xrows),
         "DVXR (subject-held-out AUROC) beside published results on the same/comparable cohorts, "
         "each labeled with its protocol and DOI. Cross-subject (LOSO / subject-independent) is the "
         "honest bar; segment-level numbers carry subject leakage and are shown for context only, "
         "never as a head-to-head win across mismatched protocols.",
         "tab:sota", "dvxr.serve.evidence.EXTERNAL_SOTA")

    # 4. clinical utility — decision-curve net benefit (bootstrap-gated)
    urows = []
    for task in ("mumtaz_depression", "wesad_stress", "eegmat_workload"):
        dca = dca_of(task)
        if not dca or not dca.get("summary"):
            continue
        s = dca["summary"]
        band = (f"{int(s['useful_band'][0]*100)}--{int(s['useful_band'][1]*100)}%"
                if s.get("useful") else "none (not stable)")
        urows.append({
            "Task": task, "Level": dca.get("level", "window"),
            "Useful band": band,
            "Peak net benefit": (f"{s.get('best_gain'):.3f}" if s.get("useful") else "--"),
            "Bootstrap 95% LB": f"{s.get('best_gain_lo'):.3f}",
        })
    if urows:
        emit("clinical_utility", pd.DataFrame(urows),
             "Clinical utility by decision-curve analysis (Vickers and Elkin, 2006): the decision-"
             "threshold band over which the screener's net benefit exceeds both treat-all and treat-"
             "none, the peak added net benefit, and its one-sided 95% bootstrap lower bound. The "
             "useful band is bootstrap-gated so a chance advantage reads as not useful.",
             "tab:utility", "outputs/product/screeners/*/manifest.json (decision_curve)")

    # method-contribution provenance line (for the manifest, not a table)
    (tdir / "PRODUCT_MANIFEST.txt").write_text(
        "\n".join(f"{k}.tex  <-  {v['source']}  ({v['rows']} rows)" for k, v in manifest.items())
        + "\n\nmethod claim: " + METHOD_CLAIMS[0]["claim"] + "\n")
    return manifest
