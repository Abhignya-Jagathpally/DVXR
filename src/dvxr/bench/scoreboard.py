"""dvxr.bench.scoreboard — the relativity table + honest Markdown report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from dvxr.bench.run import TaskResult


def scoreboard_dataframe(results: List[TaskResult], target_pct: float = 50.0) -> pd.DataFrame:
    return pd.DataFrame([r.relativity.as_row(target_pct) for r in results])


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_scoreboard(results: List[TaskResult], out_dir: str = "outputs",
                     ablation_by_task: Dict[str, dict] | None = None,
                     target_pct: float = 50.0, meta: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = scoreboard_dataframe(results, target_pct)
    csv_path = out / "benchmark_scoreboard.csv"
    df.to_csv(csv_path, index=False)

    # M4: which tasks are actually multimodal
    MODALITY = {"stress": "MULTIMODAL (4 peripheral-physiology streams, one wearable)",
                "glucose": "single-modality (CGM only)",
                "mortality": "single-modality (EHR only)"}

    lines: List[str] = ["# CACMF relativity scoreboard — real labels, held-out subjects\n"]
    if meta:
        lines.append("**Run params:** " + ", ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
    if results:
        lines.append(f"**Protocol:** {results[0].protocol}\n")
    lines.append(
        "Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable "
        "representation into a shared head. Baseline = the single strongest NON-fused "
        "opponent on the same folds (trivial floor, classical GBM, best single "
        "modality, or a real pretrained SOTA encoder — unstable configs excluded). "
        "Error metric per task; RER = (base_err - prop_err)/base_err. No configuration "
        "is assumed to win.\n")
    lines.append("\n**Modality labeling (M4):** " + "; ".join(
        f"{k} = {v}" for k, v in MODALITY.items()) + ". Multimodal-fusion conclusions "
        "rest on the **stress** task; no dataset co-registers EEG+CGM+EHR per subject.\n")

    # headline scoreboard
    try:
        lines.append(df.to_markdown(index=False))
    except Exception:
        lines.append("```\n" + df.to_string(index=False) + "\n```")

    # honest verdict per task
    lines.append("\n## Verdict\n")
    for r in results:
        rel = r.relativity
        met = rel.meets_target(target_pct)
        verdict = ("MEETS" if met else "does NOT meet") + f" the >={int(target_pct)}% RER bar"
        mod = MODALITY.get(r.task, "")
        lines.append(
            f"- **{r.task}** ({r.metric}, {mod}): fused {_fmt(rel.prop_err)} vs "
            f"{r.best_baseline} {_fmt(rel.base_err)} -> RER {rel.rer_pct:.1f}% "
            f"(95% CI {rel.rer_ci[0]:.1f}..{rel.rer_ci[1]:.1f}, "
            f"Wilcoxon p={_fmt(rel.p_wilcoxon,4)}, Holm p={_fmt(rel.p_holm,4)}) "
            f"-> **{verdict}.**")

    # M2: stability / failures
    any_fail = any(r.failures for r in results) or any(r.unstable for r in results)
    lines.append("\n## Stability (M2)\n")
    if not any_fail:
        lines.append("- No config/fold failures; no unstable configs.")
    for r in results:
        if r.failures or r.unstable:
            lines.append(f"- **{r.task}**: failures by config = {r.failures or '{}'}; "
                         f"unstable (NaN >20% folds) = {r.unstable or '[]'}")

    # per-config CV means
    lines.append("\n## Per-configuration CV error (lower is better)\n")
    for r in results:
        means = r.config_means()
        tbl = pd.DataFrame(
            sorted(means.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else 9e9)),
            columns=["config", f"{r.metric}"])
        tbl[r.metric] = tbl[r.metric].round(4)
        lines.append(f"\n### {r.task}"
                     + (f"  (SOTA backend: {r.backend_note})" if r.backend_note else ""))
        try:
            lines.append(tbl.to_markdown(index=False))
        except Exception:
            lines.append("```\n" + tbl.to_string(index=False) + "\n```")

    # true modality ablation
    if ablation_by_task:
        lines.append("\n## True modality ablation (retrain without the modality)\n")
        for task, table in ablation_by_task.items():
            if not table:
                continue
            lines.append(f"\n### {task}  (contribution = error increase when dropped)")
            try:
                lines.append(pd.DataFrame(table).to_markdown(index=False))
            except Exception:
                lines.append("```\n" + pd.DataFrame(table).to_string(index=False) + "\n```")

    (out / "benchmark_scoreboard.md").write_text("\n".join(lines) + "\n")
    return {"csv": str(csv_path), "md": str(out / "benchmark_scoreboard.md"),
            "n_tasks": len(results)}
