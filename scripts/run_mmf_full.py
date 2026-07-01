#!/usr/bin/env python3
"""run_mmf_full.py — one-command CACMF pipeline entry point.

Stage 1 provides the scaffold + ``--profile``. Later stages wire in VQ encoders,
fusion, multi-task training, real-time streaming, explainability, LLM insight,
ablation, and paper tables. Everything must run offline on CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvxr.config import CACMFConfig, DEFAULTS  # noqa: E402


def cmd_profile(args: argparse.Namespace) -> int:
    from dvxr.ingest.profile import profile_data_dir

    report = profile_data_dir(
        path=args.data, report_path=args.report, strict=args.strict)
    cov = report.modality_coverage()
    print(f"[profile] scanned {len(report.files)} files under {args.data!r}")
    print(f"[profile] modality coverage: {cov}")
    if report.unmapped:
        print(f"[profile] {len(report.unmapped)} UNMAPPED files "
              f"(see {args.report}); re-run with --strict to fail on them.")
    print(f"[profile] report written -> {args.report}")
    return 0


def cmd_realtime(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from dvxr.realtime.monitor import stream_fused_predictions
    from dvxr.schemas import REQUIRED_EVENT_COLUMNS

    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rng = np.random.default_rng(7)

    def row(sec, modality, channel, value):
        return {"subject_id": "demo", "session_id": "s1",
                "timestamp_utc": base + pd.Timedelta(seconds=sec),
                "source_system": "demo", "device": "demo", "modality": modality,
                "channel": channel, "value": float(value), "unit": "u",
                "sampling_rate_hz": 1.0, "quality_flag": "ok",
                "label_name": "", "label_value": ""}

    rows = []
    for sec in range(0, 600, 5):
        rows.append(row(sec, "eda", "eda", 2.0 + rng.normal(0, 0.3) + sec / 1200.0))
        rows.append(row(sec, "eeg", "eeg", rng.normal(0, 1)))
    for sec in range(0, 600, 60):
        rows.append(row(sec, "cgm", "glucose", 150 + 40 * np.sin(sec / 120.0)))
    events = pd.DataFrame(rows)[REQUIRED_EVENT_COLUMNS]

    df = stream_fused_predictions(events, step_seconds=30, window_seconds=30)
    out = Path(args.realtime_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    fired = df["interventions"].astype(str).str.len().gt(0).sum()
    print(f"[realtime] streamed {len(df)} steps; {fired} step(s) fired interventions")
    print(f"[realtime] stream written -> {out}")
    return 0


def cmd_insight(args: argparse.Namespace) -> int:
    from dvxr.llm.insight import write_insight_report

    # a grounded demo bundle (numbers only; the layer never invents values)
    bundle = {
        "tasks": {
            "stress_detection": {"probability": 0.72, "band": "elevated"},
            "cognitive_workload": {"probability": 0.58, "band": "watch"},
        },
        "glucose": {"now": 168.0, "forecast": 183.0, "lower": 168.0, "upper": 198.0},
        "biomarkers": {"hrv_rmssd": 28.4, "eda_tonic_mean": 3.10},
        "top_modality": "wearable_phys",
        "interventions": ["Glucose is elevated — hydrate and recheck soon.",
                          "Stress is elevated — try a short paced-breathing break."],
    }
    res = write_insight_report(bundle, out_path=args.insight_out)
    print(f"[insight] backend = {res['backend']}")
    print(f"[insight] report written -> {res['path']}")
    return 0


def _synthetic_events():
    import numpy as np
    import pandas as pd

    from dvxr.schemas import REQUIRED_EVENT_COLUMNS
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rng = np.random.default_rng(7)

    def row(sec, modality, channel, value):
        return {"subject_id": "demo", "session_id": "s1",
                "timestamp_utc": base + pd.Timedelta(seconds=sec),
                "source_system": "demo", "device": "demo", "modality": modality,
                "channel": channel, "value": float(value), "unit": "u",
                "sampling_rate_hz": 1.0, "quality_flag": "ok",
                "label_name": "", "label_value": ""}

    rows = []
    for sec in range(0, 600, 5):
        rows.append(row(sec, "eda", "eda", 2.0 + rng.normal(0, 0.3) + sec / 1200.0))
        rows.append(row(sec, "eeg", "eeg", rng.normal(0, 1)))
    for sec in range(0, 600, 60):
        rows.append(row(sec, "cgm", "glucose", 150 + 40 * np.sin(sec / 120.0)))
    return pd.DataFrame(rows)[REQUIRED_EVENT_COLUMNS]


def cmd_full(args: argparse.Namespace) -> int:
    """Run the entire CACMF pipeline end-to-end on synthetic fixtures (offline/CPU)."""
    import numpy as np
    import pandas as pd
    import torch

    from dvxr.config import FUSION_STRATEGIES
    from dvxr.encoders.codebook import VQBiosignalEncoder
    from dvxr.eval.ablation import ablation_summary, make_synthetic_dataset, run_ablation
    from dvxr.eval.paper import build_paper_tables
    from dvxr.explain.attention_maps import export_attention
    from dvxr.explain.report import explain_prediction
    from dvxr.fusion.strategies import get_fusion_strategy
    from dvxr.ingest.profile import profile_data_dir
    from dvxr.llm.insight import write_insight_report
    from dvxr.realtime.monitor import stream_fused_predictions
    from dvxr.tasks.model import build_multitask_model
    from dvxr.tasks.train import train_multitask

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULTS.with_(d=8, d_f=16, n_heads=2, n_fusion_layers=1,
                         codebook_size=32, epochs=15, seed=7)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    steps = []

    # 1. profile local data (skip gracefully if absent)
    try:
        rep = profile_data_dir("data", str(out / "data_schema_report.md"))
        steps.append(f"profile: {len(rep.files)} files, coverage {rep.modality_coverage()}")
    except Exception as exc:  # pragma: no cover
        steps.append(f"profile: skipped ({exc})")

    # 2. per-modality VQ codebook encoder + usage export
    feat_cols = [f"f{i}" for i in range(10)]
    feat_frame = pd.DataFrame(
        np.random.RandomState(0).randn(48, 10), columns=feat_cols)
    vq = VQBiosignalEncoder(embedding_dim=cfg.d, hidden_dim=16, n_layers=1,
                            n_heads=2, epochs=8, codebook_size=cfg.codebook_size, seed=7)
    vq.fit_transform(feat_frame, feat_cols)
    vq.export(feat_frame, out)                      # codebook_usage.csv + latent_quantized.npy
    steps.append(f"vq codebook: perplexity {vq.perplexity(feat_frame):.2f}")

    # 3. dataset + per-modality latents
    ds = make_synthetic_dataset(n_subjects=14, per_subject=10, seed=0)
    input_dims = {m: ds["features"][m].shape[1] for m in ds["features"]}
    model = build_multitask_model(cfg, input_dims)
    model.eval()
    feats_t = {m: torch.tensor(ds["features"][m], dtype=torch.float32) for m in input_dims}
    with torch.no_grad():
        z = model.encode(feats_t)

    # 4. all five fusion strategies + latent/attention export
    shapes = {}
    for s in FUSION_STRATEGIES:
        fo = get_fusion_strategy(s, cfg, list(input_dims.keys()))(z)
        shapes[s] = tuple(fo.h.shape)
    export_attention(get_fusion_strategy("attention", cfg, list(input_dims.keys()))(z),
                     out / "fusion_attention.csv")
    model.cacmf.export_latents(z, out_dir=str(out))
    steps.append(f"fusion strategies: {shapes}")

    # 5. joint multi-task training
    labels = {"stress_detection": torch.tensor(ds["tasks"]["stress_detection"]["y"]).long()}
    target = torch.tensor(ds["tasks"]["glucose"]["y"], dtype=torch.float32)
    res = train_multitask(model, feats_t, labels, forecast_target=target, config=cfg,
                          log_path=str(out / "train_log.csv"))
    steps.append(f"train: final total loss {res['history'][-1]['total']:.4f}")

    # 6. real-time fused stream + interventions
    events = _synthetic_events()
    rt = stream_fused_predictions(events, step_seconds=30, window_seconds=30)
    rt.to_csv(out / "realtime_fused_stream.csv", index=False)
    steps.append(f"realtime: {len(rt)} steps, "
                 f"{rt['interventions'].astype(str).str.len().gt(0).sum()} with interventions")

    # 7. explainability bundle
    explain_prediction(events=events, cacmf_model=model.cacmf, latents=z,
                       feature_frame=feat_frame, feature_columns=feat_cols,
                       out_path=str(out / "explanation_example.md"))
    steps.append("explain: explanation_example.md")

    # 8. offline LLM insight
    bundle = {
        "tasks": {"stress_detection": {"probability": 0.72, "band": "elevated"}},
        "glucose": {"now": 168.0, "forecast": 183.0, "lower": 168.0, "upper": 198.0},
        "biomarkers": {"hrv_rmssd": 28.4},
        "top_modality": "wearable_phys",
        "interventions": ["Glucose is elevated — hydrate and recheck soon."],
    }
    ins = write_insight_report(bundle, out_path=str(out / "insight_example.md"))
    steps.append(f"insight: backend {ins['backend']}")

    # 9. ablation
    abl = run_ablation(ds, config=cfg, test_frac=0.3, seed=7)
    abl.to_csv(out / "ablation_table.csv", index=False)
    (out / "ablation_summary.md").write_text(ablation_summary(abl))
    steps.append(f"ablation: {len(abl)} rows across {abl['task'].nunique()} tasks")

    # 10. paper tables
    manifest = build_paper_tables(out, ROOT / "paper" / "tables")
    steps.append(f"paper tables: {sorted(manifest)}")

    print("\n=== CACMF full pipeline (offline / CPU / deterministic) ===")
    for s in steps:
        print(f"  - {s}")
    print(f"artifacts -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CACMF full pipeline runner")
    parser.add_argument("--config", type=str, default=None,
                        help="path to a CACMFConfig YAML/JSON (defaults to built-ins)")
    parser.add_argument("--profile", action="store_true",
                        help="profile the local data/ directory and write the schema report")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--report", type=str, default="outputs/data_schema_report.md")
    parser.add_argument("--strict", action="store_true",
                        help="fail loudly if any data file cannot be mapped")
    parser.add_argument("--realtime", action="store_true",
                        help="replay a synthetic multimodal fixture through the fused monitor")
    parser.add_argument("--realtime-out", type=str,
                        default="outputs/realtime_fused_stream.csv")
    parser.add_argument("--insight", action="store_true",
                        help="generate a grounded LLM insight (offline fallback if no API key)")
    parser.add_argument("--insight-out", type=str,
                        default="outputs/insight_example.md")
    parser.add_argument("--full", action="store_true",
                        help="run the entire CACMF pipeline end-to-end (default)")
    args = parser.parse_args(argv)

    cfg: CACMFConfig = CACMFConfig.load(args.config) if args.config else DEFAULTS
    print(f"[run_mmf_full] CACMF config: fusion={cfg.fusion_strategy} "
          f"aggregation={cfg.aggregation} d={cfg.d} K={cfg.codebook_size} "
          f"real_weights={cfg.use_real_weights} seed={cfg.seed}")

    if args.profile:
        return cmd_profile(args)
    if args.realtime:
        return cmd_realtime(args)
    if args.insight:
        return cmd_insight(args)
    # default: run the whole pipeline end-to-end
    return cmd_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
