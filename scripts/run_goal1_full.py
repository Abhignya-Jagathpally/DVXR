"""End-to-end Goal 1 demonstration across all wired capabilities.

Exercises, on always-runnable synthetic fixtures, every piece added to close the
Goal 1 deliverable gaps:

  1. Multimodal ingestion (wearable/EEG/biosensor/CGM/EHR + multi-omics)
  2. Real device + VR/AR converters (Galea, EMOTIV, VR/AR)
  3. Neural foundation-model embeddings (torch BIOT-style encoder) vs PCA baseline
  4. The seven named clinical task heads
  5. Real-time stress + glucose monitoring (streaming)
  6. Explainable neural + physiological biomarkers
  7. Per-subject personalization

Run: python3 scripts/run_goal1_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from goal1_pipeline.biomarkers import neural_biomarker_saliency, physiological_biomarkers
from goal1_pipeline.clinical_tasks import clinical_tasks_table, derive_task_labels, train_clinical_task
from goal1_pipeline.encoders import FeatureEncoder
from goal1_pipeline.features import build_stress_windows, feature_columns
from goal1_pipeline.models import train_stress_classifier
from goal1_pipeline.omics import build_omics_features, generate_omics_like_table
from goal1_pipeline.personalization import per_subject_normalize
from goal1_pipeline.realtime import stream_predictions
from goal1_pipeline.sample_data import generate_public_like_events
from goal1_pipeline.schemas import summarize_events


def _hr(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multimodal ingestion ------------------------------------------------
    _hr("1. Multimodal ingestion (wearable/EEG/biosensor/CGM/EHR)")
    events = generate_public_like_events(output_dir / "goal1_full_events.csv", subjects=6, minutes=18)
    summary = summarize_events(events)
    print(f"Events: rows={summary.rows}, subjects={summary.subjects}, modalities={summary.modalities}")

    _hr("1b. Multi-omics ingestion (genomics/proteomics/metabolomics)")
    omics_events = generate_omics_like_table(output_dir / "goal1_omics_events.csv", subjects=8)
    omics_features = build_omics_features(omics_events)
    omics_features.to_csv(output_dir / "goal1_omics_features.csv", index=False)
    print(
        f"Omics events: rows={len(omics_events)}, subjects={omics_features.shape[0]}, "
        f"omic features/subject={omics_features.shape[1] - 3}"
    )

    # 2. Device + VR/AR converters ------------------------------------------
    _hr("2. Real device + VR/AR converters (demo mode)")
    import convert_emotiv_subject
    import convert_galea_subject
    import ingest_vr_session

    galea = convert_galea_subject.convert(None, output_dir / "device_galea.csv", demo=True)
    emotiv = convert_emotiv_subject.convert(None, output_dir / "device_emotiv.csv", demo=True, device="epocx")
    vr = ingest_vr_session.convert(None, output_dir / "device_vr.csv", demo=True)
    print(f"Galea: {len(galea)} rows {sorted(galea['modality'].unique())}")
    print(f"EMOTIV EPOC X: {len(emotiv)} rows {sorted(emotiv['modality'].unique())}")
    print(f"VR/AR: {len(vr)} rows {sorted(vr['modality'].unique())}")

    # 3. Neural foundation-model embeddings vs PCA --------------------------
    _hr("3. Neural foundation-model embeddings (torch) vs PCA baseline")
    stress_windows = build_stress_windows(events, window_seconds=30, step_seconds=15)
    cols = feature_columns(stress_windows)
    pca_embed = FeatureEncoder(max_components=16).fit_transform(stress_windows, cols)
    print(f"PCA baseline embeddings: shape={pca_embed.shape}")
    try:
        from goal1_pipeline.neural_encoders import NeuralBiosignalEncoder

        neural = NeuralBiosignalEncoder(embedding_dim=16, epochs=15, seed=7)
        neural_embed = neural.fit_transform(stress_windows, cols)
        neural.save(output_dir / "neural_encoder.pt")
        print(f"Neural (BIOT-style) embeddings: shape={neural_embed.shape} -> saved neural_encoder.pt")
    except RuntimeError as exc:  # torch missing
        print(f"Neural encoder unavailable ({exc}); PCA baseline used.")

    # 4. Seven clinical task heads ------------------------------------------
    _hr("4. Clinical task heads (7 Goal-1 tasks)")
    tasks_table = clinical_tasks_table()
    tasks_table.to_csv(output_dir / "clinical_tasks.csv", index=False)
    results = []
    for task in tasks_table["name"]:
        try:
            # smoke/demo run on synthetic fixtures — allow the fabrication helpers (never a
            # reported number); scientific runs use the default allow_synthetic=False gate.
            frame = derive_task_labels(events, task, allow_synthetic=True)
            trained = train_clinical_task(frame, task)
            m = trained.metrics
            results.append((task, m.get("accuracy"), m.get("auroc"), m.get("f1")))
            print(f"  {task:24s} acc={m.get('accuracy'):.3f} auroc={m.get('auroc'):.3f} f1={m.get('f1'):.3f}")
        except Exception as exc:  # robustness: report, don't abort the demo
            print(f"  {task:24s} skipped ({type(exc).__name__}: {exc})")
    pd.DataFrame(results, columns=["task", "accuracy", "auroc", "f1"]).to_csv(
        output_dir / "clinical_task_metrics.csv", index=False
    )

    # 5. Real-time stress + glucose monitoring ------------------------------
    _hr("5. Real-time stress + glucose monitoring (streaming)")
    stress_model = train_stress_classifier(stress_windows)
    stream = stream_predictions(events, stress_model, step_seconds=60, window_seconds=30)
    stream.to_csv(output_dir / "realtime_stream.csv", index=False)
    print(f"Streaming predictions: {len(stream)} time steps, columns={list(stream.columns)}")

    # 6. Explainable neural + physiological biomarkers ----------------------
    _hr("6. Explainable neural + physiological biomarkers")
    bio = physiological_biomarkers(events)
    bio.to_csv(output_dir / "physiological_biomarkers.csv", index=False)
    print(f"Physiological biomarkers: {bio.shape[0]} subject-sessions, markers={[c for c in bio.columns if c not in ('subject_id', 'session_id')]}")
    saliency = neural_biomarker_saliency(stress_windows, cols, top_n=8)
    saliency.to_csv(output_dir / "neural_biomarker_saliency.csv", index=False)
    method = saliency["method"].iloc[0] if "method" in saliency.columns and len(saliency) else "n/a"
    print(f"Top biomarker saliency ({method}):")
    print(saliency.head(8).to_string(index=False))

    # 7. Per-subject personalization ----------------------------------------
    _hr("7. Per-subject personalization")
    normalized = per_subject_normalize(stress_windows, cols)
    print(f"Per-subject normalized windows: shape={normalized[cols].shape} (features z-scored within each subject)")

    print(f"\nAll Goal 1 capabilities exercised. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
