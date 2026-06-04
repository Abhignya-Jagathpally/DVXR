from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal1_pipeline.encoders import FeatureEncoder, recommendation_table
from goal1_pipeline.explain import top_linear_contributors
from goal1_pipeline.features import build_glucose_forecast_table, build_stress_windows, feature_columns
from goal1_pipeline.models import train_glucose_forecaster, train_stress_classifier
from goal1_pipeline.registry import dataset_choice_table, model_choice_table
from goal1_pipeline.reporting import write_model_card
from goal1_pipeline.sample_data import generate_public_like_events
from goal1_pipeline.schemas import summarize_events
from goal1_pipeline.streaming import predict_latest_stress


def main() -> None:
    data_path = ROOT / "data" / "sample" / "canonical_events.csv"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    events = generate_public_like_events(data_path, subjects=3, minutes=9, eeg_channels=6, eeg_rate_hz=16.0)
    summary = summarize_events(events)
    print(f"Loaded canonical events: rows={summary.rows}, subjects={summary.subjects}, sessions={summary.sessions}")
    print(f"Modalities: {', '.join(summary.modalities)}")

    recommendations = recommendation_table()
    recommendations.to_csv(output_dir / "encoder_recommendations.csv", index=False)
    model_choice_table().to_csv(output_dir / "model_choice_registry.csv", index=False)
    dataset_choice_table().to_csv(output_dir / "dataset_registry.csv", index=False)
    print("Saved encoder recommendations.")

    stress_windows = build_stress_windows(events, window_seconds=30, step_seconds=15)
    stress_windows.to_csv(output_dir / "stress_windows.csv", index=False)
    print(f"Built stress windows: {len(stress_windows)} rows, {len(feature_columns(stress_windows))} feature columns")

    encoder = FeatureEncoder(max_components=16)
    embeddings = encoder.fit_transform(stress_windows, feature_columns(stress_windows))
    embeddings.to_csv(output_dir / "stress_window_embeddings.csv", index=False)
    print(f"Generated local embeddings: shape={embeddings.shape}")

    stress_model = train_stress_classifier(stress_windows)
    stress_model.predictions.to_csv(output_dir / "stress_predictions.csv", index=False)
    print(
        "Stress model: "
        f"accuracy={stress_model.metrics['accuracy']:.3f}, "
        f"f1={stress_model.metrics['f1']:.3f}, "
        f"auroc={stress_model.metrics['auroc']:.3f}, "
        f"brier={stress_model.metrics['brier']:.3f}, "
        f"ece={stress_model.metrics['ece']:.3f}"
    )
    print("Top stress contributors:")
    print(top_linear_contributors(stress_model, top_n=8).to_string(index=False))

    glucose_table = build_glucose_forecast_table(events, history_minutes=30, horizon_minutes=30)
    glucose_table.to_csv(output_dir / "glucose_forecast_table.csv", index=False)
    glucose_model = train_glucose_forecaster(glucose_table)
    glucose_model.predictions.to_csv(output_dir / "glucose_predictions.csv", index=False)
    print(
        "Glucose model: "
        f"MAE={glucose_model.metrics['mae_mg_dl']:.2f} mg/dL, "
        f"RMSE={glucose_model.metrics['rmse_mg_dl']:.2f} mg/dL, "
        f"90% interval +/-{glucose_model.metrics['interval_radius_mg_dl']:.2f} mg/dL, "
        f"coverage={glucose_model.metrics['interval_coverage']:.3f}"
    )

    latest = predict_latest_stress(events, stress_model, window_seconds=30)
    print(
        "Latest streaming-style stress prediction: "
        f"{latest['predicted_label']} "
        f"(p={latest['stress_probability']:.3f})"
    )
    write_model_card(output_dir / "goal1_model_card.md", summary, stress_model, glucose_model)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()