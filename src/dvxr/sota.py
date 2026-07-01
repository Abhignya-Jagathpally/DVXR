from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SotaModel:
    task: str
    model: str
    rank: int
    selected_for_goal1: bool
    evidence_score: int
    goal1_fit_score: int
    integration_score: int
    calibration_score: int
    rationale: str
    limitation: str
    source: str

    @property
    def total_score(self) -> int:
        return self.evidence_score + self.goal1_fit_score + self.integration_score + self.calibration_score


SOTA_MODELS = [
    SotaModel(
        task="eeg_bci",
        model="EEG-X",
        rank=1,
        selected_for_goal1=True,
        evidence_score=5,
        goal1_fit_score=5,
        integration_score=3,
        calibration_score=3,
        rationale=(
            "Best fit for Galea/EMOTIV because it is designed for varying channel counts, "
            "channel combinations, recording lengths, and noisy cross-device EEG."
        ),
        limitation="Recent preprint; integration depends on released weights/code maturity.",
        source="https://arxiv.org/abs/2511.08861",
    ),
    SotaModel(
        task="eeg_bci",
        model="LaBraM",
        rank=2,
        selected_for_goal1=False,
        evidence_score=5,
        goal1_fit_score=4,
        integration_score=4,
        calibration_score=3,
        rationale=(
            "Strong BCI foundation model trained for generic EEG representations; good benchmark "
            "for DEAP/BCI transfer once weight integration is available."
        ),
        limitation="Less explicitly device-agnostic than EEG-X for Galea-vs-EMOTIV channel mismatch.",
        source="https://arxiv.org/abs/2405.18765",
    ),
    SotaModel(
        task="eeg_bci",
        model="EEGPT",
        rank=3,
        selected_for_goal1=False,
        evidence_score=4,
        goal1_fit_score=4,
        integration_score=4,
        calibration_score=3,
        rationale="Strong universal EEG representation model and useful comparison for DEAP arousal.",
        limitation="Selection is weaker than EEG-X for noisy device-agnostic deployment.",
        source="https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf",
    ),
    SotaModel(
        task="biosignal_wearable",
        model="BIOT",
        rank=1,
        selected_for_goal1=True,
        evidence_score=5,
        goal1_fit_score=5,
        integration_score=4,
        calibration_score=3,
        rationale=(
            "Best for heterogeneous wearable/BCI streams because it was built for mismatched channels, "
            "variable lengths, and missing biosignal values."
        ),
        limitation="Does not directly solve glucose forecasting or EHR timeline modeling.",
        source="https://papers.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html",
    ),
    SotaModel(
        task="biosignal_wearable",
        model="MOMENT",
        rank=2,
        selected_for_goal1=False,
        evidence_score=5,
        goal1_fit_score=4,
        integration_score=5,
        calibration_score=3,
        rationale="Best open general time-series foundation fallback across classification, forecasting, imputation, and anomaly detection.",
        limitation="Generic time-series model; less biosignal-specific than BIOT for EEG/EDA/PPG mixtures.",
        source="https://arxiv.org/abs/2402.03885",
    ),
    SotaModel(
        task="cgm_glucose",
        model="GluFormer",
        rank=1,
        selected_for_goal1=True,
        evidence_score=5,
        goal1_fit_score=5,
        integration_score=2,
        calibration_score=3,
        rationale=(
            "Transformer foundation model trained on large-scale CGM data; the strongest fit for "
            "glucose forecasting when weights/data access are available."
        ),
        limitation="Weights/data access are gated; not guaranteed reproducible without the source cohort.",
        source="https://www.nature.com/articles/s41586-025-09925-9",
    ),
    SotaModel(
        task="cgm_glucose",
        model="Conformalized Ridge baseline",
        rank=2,
        selected_for_goal1=True,
        evidence_score=3,
        goal1_fit_score=4,
        integration_score=5,
        calibration_score=5,
        rationale=(
            "Always-runnable forecasting baseline in this repo: split-conformal prediction intervals "
            "give distribution-free coverage when GluFormer access is unavailable."
        ),
        limitation="Linear baseline; lower ceiling than a CGM foundation model on long-horizon dynamics.",
        source="https://arxiv.org/abs/2107.07511",
    ),
    SotaModel(
        task="ehr_timeline",
        model="Med-BERT / BEHRT",
        rank=1,
        selected_for_goal1=True,
        evidence_score=5,
        goal1_fit_score=4,
        integration_score=4,
        calibration_score=3,
        rationale=(
            "Established transformer encoders for structured EHR code sequences; best fit for the "
            "MIMIC-IV demo concept timelines used in Goal 1."
        ),
        limitation="Structured-code focus; does not model free-text notes directly.",
        source="https://arxiv.org/abs/1904.05342",
    ),
    SotaModel(
        task="ehr_timeline",
        model="NYUTron / Foresight",
        rank=2,
        selected_for_goal1=False,
        evidence_score=4,
        goal1_fit_score=4,
        integration_score=3,
        calibration_score=3,
        rationale="Note/concept timeline models that add clinical-text reasoning over the structured baseline.",
        limitation="Heavier text pipeline; less aligned with the demo's code-only event stream.",
        source="https://www.nature.com/articles/s41586-023-06160-y",
    ),
    SotaModel(
        task="wearable_llm_insight",
        model="PHIA",
        rank=1,
        selected_for_goal1=True,
        evidence_score=4,
        goal1_fit_score=5,
        integration_score=4,
        calibration_score=2,
        rationale=(
            "LLM agent with code execution and retrieval over wearable time-series questions; best "
            "reference for turning deterministic model outputs into explainable insights."
        ),
        limitation="Agent layer should explain, not replace, the deterministic signal pipeline.",
        source="https://pmc.ncbi.nlm.nih.gov/articles/PMC12855967/",
    ),
    SotaModel(
        task="wearable_llm_insight",
        model="Health-LLM / HealthAlpaca",
        rank=2,
        selected_for_goal1=False,
        evidence_score=4,
        goal1_fit_score=4,
        integration_score=4,
        calibration_score=2,
        rationale="Instruction-tuned health prediction baseline for comparison against an agentic insight layer.",
        limitation="Prompted prediction is harder to calibrate and audit than the deterministic baselines.",
        source="https://arxiv.org/abs/2401.06866",
    ),
]


def sota_model_table() -> pd.DataFrame:
    """Return all benchmarked SOTA candidates with their composite scores."""
    rows = [{**model.__dict__, "total_score": model.total_score} for model in SOTA_MODELS]
    frame = pd.DataFrame(rows)
    return frame.sort_values(["task", "rank"]).reset_index(drop=True)


def selected_sota_table() -> pd.DataFrame:
    """Return only the models selected for the Goal 1 pipeline."""
    table = sota_model_table()
    return table[table["selected_for_goal1"]].reset_index(drop=True)


def write_sota_report(output_dir: str | Path) -> tuple[Path, Path]:
    """Write the full comparison and the selected-model report to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "sota_comparison.csv"
    selection_path = output_dir / "sota_selection.csv"
    sota_model_table().to_csv(comparison_path, index=False)
    selected_sota_table().to_csv(selection_path, index=False)
    return comparison_path, selection_path