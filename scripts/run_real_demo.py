"""Run the pipeline on real, credential-free public data.

Wearable stress classification uses the PhysioNet Non-EEG dataset (real wrist
physiology with relax/physical/emotional/cognitive-stress phases). EHR ingestion
uses the open MIMIC-IV clinical demo. Download both first with:

    python3 scripts/fetch_data.py all-free --subjects 5
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.encoders import FeatureEncoder
from goal1_pipeline.explain import top_linear_contributors
from goal1_pipeline.features import build_glucose_forecast_table, build_stress_windows, feature_columns
from goal1_pipeline.loaders import load_mimic_demo_ehr, load_noneeg_dataset, load_shanghai_cgm_dataset
from goal1_pipeline.models import train_glucose_forecaster, train_stress_classifier
from goal1_pipeline.schemas import summarize_events
from goal1_pipeline.streaming import predict_latest_stress


def main() -> None:
    real_dir = ROOT / "data" / "real"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    noneeg_dir = real_dir / "noneeg"
    mimic_dir = real_dir / "mimic_demo" / "hosp"
    cgm_dir = real_dir / "shanghai_cgm"
    if not noneeg_dir.exists() or not mimic_dir.exists() or not cgm_dir.exists():
        sys.exit("Real data not found. Run: python3 scripts/fetch_data.py all-free --subjects 20")

    subjects = len({p.name.split("_")[0] for p in noneeg_dir.glob("Subject*_AccTempEDA.hea")})
    print(f"== Real wearable stress data (PhysioNet Non-EEG, {subjects} subjects) ==")
    events = load_noneeg_dataset(noneeg_dir, subjects=subjects)
    summary = summarize_events(events)
    print(f"Ingested events: rows={summary.rows}, subjects={summary.subjects}, modalities={', '.join(summary.modalities)}")
    print(f"Stress labels present: {', '.join(summary.label_values)}")
    events.to_csv(output_dir / "real_noneeg_events.csv", index=False)

    windows = build_stress_windows(events, window_seconds=30, step_seconds=15)
    windows.to_csv(output_dir / "real_stress_windows.csv", index=False)
    stress_fraction = (windows["target"] == "stress").mean()
    print(f"Stress windows: {len(windows)} rows, {len(feature_columns(windows))} features, {stress_fraction:.1%} stress")

    encoder = FeatureEncoder(max_components=16)
    encoder.fit_transform(windows, feature_columns(windows))

    model = train_stress_classifier(windows)
    model.predictions.to_csv(output_dir / "real_stress_predictions.csv", index=False)
    print(
        "Stress model (subject-held-out test): "
        f"accuracy={model.metrics['accuracy']:.3f}, f1={model.metrics['f1']:.3f}, "
        f"auroc={model.metrics['auroc']:.3f}, brier={model.metrics['brier']:.3f}, "
        f"ece={model.metrics['ece']:.3f}, n_test={int(model.metrics['test_windows'])}"
    )
    print("Top stress contributors (real data):")
    print(top_linear_contributors(model, top_n=8).to_string(index=False))

    latest = predict_latest_stress(events, model, window_seconds=30)
    print(f"Latest streaming-style prediction: {latest['predicted_label']} (p={latest['stress_probability']:.3f})")

    print("\n== Real CGM / diabetes data (Shanghai T1DM/T2DM) ==")
    cgm_events = load_shanghai_cgm_dataset(cgm_dir)
    cgm_summary = summarize_events(cgm_events)
    cgm_events.to_csv(output_dir / "real_cgm_events.csv", index=False)
    print(
        f"Ingested CGM readings: rows={cgm_summary.rows}, patients={cgm_summary.subjects}, "
        f"sessions={cgm_summary.sessions}, glucose {cgm_events['value'].min():.0f}-{cgm_events['value'].max():.0f} mg/dL"
    )
    glucose_table = build_glucose_forecast_table(cgm_events, history_minutes=120, horizon_minutes=30)
    glucose_model = train_glucose_forecaster(glucose_table)
    glucose_model.predictions.to_csv(output_dir / "real_glucose_predictions.csv", index=False)
    print(
        "Glucose model (30-min horizon, patient-held-out test): "
        f"MAE={glucose_model.metrics['mae_mg_dl']:.2f} mg/dL, "
        f"RMSE={glucose_model.metrics['rmse_mg_dl']:.2f} mg/dL, "
        f"90% interval +/-{glucose_model.metrics['interval_radius_mg_dl']:.1f} mg/dL, "
        f"coverage={glucose_model.metrics['interval_coverage']:.3f}, "
        f"n_test={int(glucose_model.metrics['test_rows'])}"
    )

    print("\n== Real EHR ingestion (MIMIC-IV clinical demo) ==")
    ehr = load_mimic_demo_ehr(mimic_dir)
    ehr_summary = summarize_events(ehr)
    ehr.to_csv(output_dir / "real_mimic_ehr_events.csv", index=False)
    distinct_concepts = ehr["channel"].nunique()
    print(
        f"Ingested EHR events: rows={ehr_summary.rows}, patients={ehr_summary.subjects}, "
        f"distinct concepts={distinct_concepts}"
    )
    print("Sample EHR concepts:", ", ".join(sorted(ehr["channel"].unique())[:8]))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
