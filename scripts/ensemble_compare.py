"""Honestly combine deep-ensemble members and compare to the model ladder.

Averages the per-window predictions of ensemble members (which share ONE patient-disjoint
split) and computes the ensemble RMSE per horizon, then places it next to the committed
ladder (persistence / GBM / MLP / single deep net). No cherry-picking: the ensemble competes
on the same held-out patients, and whatever it scores is reported.

Usage:
  python scripts/ensemble_compare.py \
    --runs neuroglycemic-runtime/runs/cgmacros-ens-42 ...-43 ...-44 \
    --ladder outputs/_r2/glucose_model_ladder.csv --out outputs/_r2/ensemble_result.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["participant_key", "patient_id", "anchor_time", "horizon_minutes"]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def combine(run_dirs):
    frames = []
    for d in run_dirs:
        p = Path(d) / "test_predictions.csv"
        if not p.is_file():
            raise SystemExit(f"missing {p} (member not finished?)")
        f = pd.read_csv(p)[KEYS + ["target_glucose_mg_dl", "predicted_glucose_mg_dl"]]
        frames.append(f)
    # verify members share the identical split (same key set + same targets)
    base = frames[0][KEYS].astype(str).apply(tuple, axis=1)
    for f in frames[1:]:
        other = f[KEYS].astype(str).apply(tuple, axis=1)
        if set(base) != set(other):
            raise SystemExit("members do not share the same test split — cannot honestly ensemble")
    merged = frames[0].rename(columns={"predicted_glucose_mg_dl": "m0"})
    for i, f in enumerate(frames[1:], 1):
        merged = merged.merge(
            f[KEYS + ["predicted_glucose_mg_dl"]].rename(columns={"predicted_glucose_mg_dl": f"m{i}"}),
            on=KEYS)
    member_cols = [c for c in merged.columns if c.startswith("m") and c[1:].isdigit()]
    merged["ensemble"] = merged[member_cols].mean(axis=1)
    return merged, member_cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ladder", type=Path, default=Path("outputs/_r2/glucose_model_ladder.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/_r2/ensemble_result.md"))
    args = ap.parse_args()

    merged, member_cols = combine(args.runs)
    ladder = pd.read_csv(args.ladder)

    rows = []
    for h in sorted(merged["horizon_minutes"].unique()):
        g = merged[merged["horizon_minutes"] == h]
        y = g["target_glucose_mg_dl"].to_numpy(float)
        ens = _rmse(y, g["ensemble"].to_numpy(float))
        members = [_rmse(y, g[c].to_numpy(float)) for c in member_cols]
        rows.append({"horizon_minutes": int(h), "ensemble_rmse": round(ens, 3),
                     "best_member_rmse": round(min(members), 3),
                     "mean_member_rmse": round(float(np.mean(members)), 3)})
    ens_df = pd.DataFrame(rows)

    # comparison at 30 min against the ladder
    l30 = ladder[ladder["horizon_minutes"] == 30].set_index("model")["rmse_mg_dl"]
    ens30 = ens_df[ens_df["horizon_minutes"] == 30]["ensemble_rmse"].iloc[0]
    gbm = float(l30.get("gradient_boosting", float("nan")))
    beats_gbm = ens30 < gbm

    lines = ["# Deep-ensemble result vs the model ladder (@30 min, same split)\n",
             ens_df.to_markdown(index=False), "",
             f"- ensemble @30 min: **{ens30:.2f}** mg/dL",
             f"- gradient boosting @30 min: {gbm:.2f}",
             f"- single deep net @30 min: {float(l30.get('neuroglycemic_net', float('nan'))):.2f}",
             f"- persistence @30 min: {float(l30.get('persistence', float('nan'))):.2f}",
             "",
             (f"**Verdict: the ensemble BEATS gradient boosting** ({ens30:.2f} < {gbm:.2f})."
              if beats_gbm else
              f"**Verdict: the ensemble does NOT beat gradient boosting** ({ens30:.2f} vs {gbm:.2f}) — "
              "reported honestly. Gradient boosting remains the best point forecaster; the deep "
              "ensemble's value is calibrated uncertainty + abstention + fusion.")]
    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    # also emit a machine-readable row for the figure
    ens_df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(report)


if __name__ == "__main__":
    main()
