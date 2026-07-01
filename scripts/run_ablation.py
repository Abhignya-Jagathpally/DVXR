#!/usr/bin/env python3
"""run_ablation.py — Goal-3 fused-vs-single ablation.

Runs on synthetic fixtures by default (offline, CPU, deterministic); wire real data
by passing a dataset dict to run_ablation. Writes outputs/ablation_table.csv and a
Markdown summary. Honest: no configuration is assumed to win.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import DEFAULTS  # noqa: E402
from dvxr.eval.ablation import ablation_summary, make_synthetic_dataset, run_ablation  # noqa: E402


def main() -> int:
    cfg = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1, codebook_size=32)
    dataset = make_synthetic_dataset(n_subjects=14, per_subject=12, seed=0)
    df = run_ablation(dataset, config=cfg, test_frac=0.3, seed=7)

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_table.csv", index=False)
    (out / "ablation_summary.md").write_text(ablation_summary(df))

    print(f"[ablation] {len(df)} rows across {df['task'].nunique()} task(s)")
    print(f"[ablation] configs/task: "
          f"{df.groupby('task')['config_name'].nunique().to_dict()}")
    print(f"[ablation] wrote {out / 'ablation_table.csv'} + ablation_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
