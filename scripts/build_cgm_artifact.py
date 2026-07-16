#!/usr/bin/env python
"""Offline builder: fit the CGM-only artifacts and register them as the ACTIVE product models.

This is the deployment step that makes the Generate API executable (P0-1). It fits the single-modality
CGM excursion classifier and the CGM continuous forecaster on a real single cohort (CGMacros by default),
saves each as a portable artifact (``model.joblib`` + sha256-verified ``manifest.json``), and registers a
pointer + hash in a local model registry as the ACTIVE model for its report type. The API then LOADS
these at request time — it never fits.

Honesty: this only ever fits single-modality, single-cohort CGM models. There is NO fused/EEG artifact —
the fused ``stress_glucose_risk`` report has no committed model and abstains by construction.

The real model binaries are written under ``artifacts/`` (gitignored); a clean checkout has no artifact,
so the API fails closed (abstains) until this builder runs. Deterministic given a fixed seed.

Usage:
    python scripts/build_cgm_artifact.py [--data data/real/cgmacros] \
        [--artifact-root artifacts] [--registry-db artifacts/registry.db] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.eval.glucose_ablation import load_cgmacros  # noqa: E402
from dvxr.prediction.forecast_service import CgmOnlyGlucoseForecastService  # noqa: E402
from dvxr.prediction.registry import FORECAST_REGISTRY_NAME, RISK_REGISTRY_NAME  # noqa: E402
from dvxr.prediction.service import CgmOnlyExcursionService  # noqa: E402
from dvxr.storage.local import open_local_stores  # noqa: E402
from dvxr.targets import ExcursionThresholds, build_excursion_labels  # noqa: E402


def _manifest_sha(path: Path) -> str:
    return json.loads((path / "manifest.json").read_text())["artifact_sha256"]


def _thinned_anchors(cgm, history_minutes, anchor_stride, max_anchors_per_subject):
    """Thin the anchor grid (as run_glucose_ablation does) so training on a DENSE real cohort is
    tractable — every-timestamp anchoring would create millions of near-duplicate examples."""
    anchors = []
    for _sid, g in cgm.groupby("subject_id"):
        t = pd.to_datetime(g["timestamp"]).sort_values()
        anchors += list(t.iloc[history_minutes // 5::anchor_stride])[:max_anchors_per_subject]
    return sorted(set(anchors))


def build(data_root: str, artifact_root: str, registry_db: str, seed: int = 7,
          history_minutes: int = 240, anchor_stride: int = 8,
          max_anchors_per_subject: int = 60) -> dict:
    cgm = load_cgmacros(data_root)
    if cgm.empty:
        raise SystemExit(f"no CGMacros data found under {data_root!r} — cannot build a real artifact")
    thr = ExcursionThresholds(history_minutes=history_minutes)
    art = Path(artifact_root)
    registry = open_local_stores(registry_db).models
    anchors = _thinned_anchors(cgm, history_minutes, anchor_stride, max_anchors_per_subject)

    # --- CGM excursion classifier -> cgm_glucose_risk ---
    examples = build_excursion_labels(cgm, thresholds=thr, anchors=anchors, subject_col="subject_id")
    risk = CgmOnlyExcursionService.fit(cgm, examples, thresholds=thr, seed=seed)
    risk_dir = art / "cgm_only" / "excursion"
    risk.save(risk_dir)
    risk_rel = str(risk_dir.relative_to(art))
    registry.register(RISK_REGISTRY_NAME, risk.model_version,
                      {"kind": "cgm_excursion", "artifact_path": risk_rel,
                       "artifact_sha256": _manifest_sha(risk_dir), "model_version": risk.model_version,
                       "modality_scope": risk.modality_scope,
                       "horizons_fitted": sorted(int(h) for h in risk._models),
                       "skipped_horizons": list(risk.skipped_horizons)},
                      active=True)

    # --- CGM continuous forecaster -> cgm_glucose_forecast ---
    forecast = CgmOnlyGlucoseForecastService.fit(cgm, thresholds=thr, anchors=anchors, seed=seed)
    fc_dir = art / "cgm_only" / "forecast"
    forecast.save(fc_dir)
    fc_rel = str(fc_dir.relative_to(art))
    registry.register(FORECAST_REGISTRY_NAME, forecast.model_version,
                      {"kind": "cgm_forecast", "artifact_path": fc_rel,
                       "artifact_sha256": _manifest_sha(fc_dir), "model_version": forecast.model_version,
                       "modality_scope": forecast.modality_scope,
                       "coverage_target": forecast.coverage_target,
                       "baseline_report": forecast.baseline_report},
                      active=True)

    return {
        "n_subjects": int(cgm["subject_id"].nunique()),
        "registry_db": registry_db, "artifact_root": artifact_root,
        "cgm_glucose_risk": {"version": risk.model_version, "path": risk_rel,
                             "horizons": sorted(int(h) for h in risk._models),
                             "skipped": list(risk.skipped_horizons)},
        "cgm_glucose_forecast": {"version": forecast.model_version, "path": fc_rel,
                                 "coverage_target": forecast.coverage_target,
                                 "baseline_report": forecast.baseline_report},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/real/cgmacros")
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--registry-db", default="artifacts/registry.db")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--history-minutes", type=int, default=240)
    args = ap.parse_args()
    Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
    summary = build(args.data, args.artifact_root, args.registry_db, seed=args.seed,
                    history_minutes=args.history_minutes)
    print(json.dumps(summary, indent=2))
    print("\nRegistered ACTIVE artifacts. The fused stress_glucose_risk report has NO artifact and "
          "abstains by construction.")


if __name__ == "__main__":
    main()
