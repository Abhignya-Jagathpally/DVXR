from __future__ import annotations

from pathlib import Path

from .schemas import DataSummary


def write_model_card(path: str | Path, summary: DataSummary, stress_model, glucose_model) -> None:
    path = Path(path)
    lines = [
        "# Goal 1 Model Card",
        "",
        "## Data",
        "",
        f"- Rows: {summary.rows}",
        f"- Subjects: {summary.subjects}",
        f"- Sessions: {summary.sessions}",
        f"- Modalities: {', '.join(summary.modalities)}",
        "",
        "This demo uses synthetic public-data-shaped fixtures. Treat its metrics as pipeline validation, not scientific evidence.",
        "",
        "## Stress Risk Model",
        "",
        "- Task: binary stress vs non-stress classification from wearable and EEG windows.",
        "- Split: subject-level train/calibration/test split.",
        "- Calibration: Platt calibration on a held-out calibration split.",
        f"- Accuracy: {stress_model.metrics['accuracy']:.3f}",
        f"- F1: {stress_model.metrics['f1']:.3f}",
        f"- AUROC: {stress_model.metrics['auroc']:.3f}",
        f"- Brier score: {stress_model.metrics['brier']:.3f}",
        f"- Expected calibration error: {stress_model.metrics['ece']:.3f}",
        "",
        "## Glucose Forecast Model",
        "",
        "- Task: short-horizon glucose forecasting from recent CGM features.",
        "- Uncertainty: split-conformal residual interval from held-out calibration data.",
        f"- MAE: {glucose_model.metrics['mae_mg_dl']:.2f} mg/dL",
        f"- RMSE: {glucose_model.metrics['rmse_mg_dl']:.2f} mg/dL",
        f"- 90% interval radius: {glucose_model.metrics['interval_radius_mg_dl']:.2f} mg/dL",
        f"- Interval coverage: {glucose_model.metrics['interval_coverage']:.3f}",
        "",
        "## Limitations",
        "",
        "- Synthetic fixtures are intentionally clean, so perfect stress scores are expected.",
        "- Real WESAD/DEAP/CGM/EHR results must use subject-held-out or time-forward splits.",
        "- Personalized risk must be claimed only after per-subject calibration improves held-out performance.",
        "- LLM or agent outputs should explain model results; they should not replace deterministic signal processing.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")