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

    lines: List[str] = ["# CACMF relativity scoreboard — real labels, held-out subjects\n"]
    if meta:
        lines.append("**Protocol:** " + ", ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
    lines.append(
        "Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable "
        "representation into a shared head. Baseline = the single strongest NON-fused "
        "opponent on the same folds (trivial floor, classical GBM, best single "
        "modality, or a real pretrained SOTA encoder). Error metric per task; "
        "RER = (base_err - prop_err)/base_err. No configuration is assumed to win.\n")

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
        lines.append(
            f"- **{r.task}** ({r.metric}): fused {_fmt(rel.prop_err)} vs "
            f"{r.best_baseline} {_fmt(rel.base_err)} -> RER {rel.rer_pct:.1f}% "
            f"(95% CI {rel.rer_ci[0]:.1f}..{rel.rer_ci[1]:.1f}, "
            f"Wilcoxon p={_fmt(rel.p_wilcoxon,4)}, Holm p={_fmt(rel.p_holm,4)}) "
            f"-> **{verdict}.**")

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
