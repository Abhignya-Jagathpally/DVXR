from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelChoice:
    modality: str
    selected: str
    rank: int
    evidence_basis: str
    why_for_goal1: str
    source: str


@dataclass(frozen=True)
class DatasetChoice:
    dataset: str
    status: str
    modalities: str
    goal1_use: str
    source: str
    note: str = ""


MODEL_CHOICES = [
    ModelChoice(
        modality="wearable_llm_agent",
        selected="PHIA",
        rank=1,
        evidence_basis="LLM agent with code execution and retrieval over wearable time-series questions.",
        why_for_goal1="Best reference for turning model outputs into explainable personal health insights after deterministic analytics run.",
        source="https://pmc.ncbi.nlm.nih.gov/articles/PMC12855967/",
    ),
    ModelChoice(
        modality="wearable_llm_prediction",
        selected="Health-LLM / HealthAlpaca",
        rank=1,
        evidence_basis="Wearable health-prediction benchmark across multiple LLMs and task types.",
        why_for_goal1="Best reference for LLM-conditioned wearable prediction; use after numerical encoders, not instead of them.",
        source="https://arxiv.org/abs/2401.06866",
    ),
    ModelChoice(
        modality="personal_health_llm",
        selected="PH-LLM",
        rank=1,
        evidence_basis="Fine-tuned Gemini model for personal-health reasoning over numerical wearable time-series.",
        why_for_goal1="Best reference for multimodal encoding of wearable summaries into an LLM interface.",
        source="https://arxiv.org/abs/2406.06474",
    ),
    ModelChoice(
        modality="eeg_bci",
        selected="EEG-X",
        rank=1,
        evidence_basis="Device-agnostic channel embeddings and noise-aware reconstruction for cross-device EEG.",
        why_for_goal1="Most relevant for Galea plus EMOTIV because channel layouts, devices, and noise profiles differ.",
        source="https://arxiv.org/abs/2511.08861",
    ),
    ModelChoice(
        modality="eeg_bci",
        selected="LaBraM",
        rank=2,
        evidence_basis="Large EEG foundation model for generic BCI representations.",
        why_for_goal1="Strong benchmark if EEG-X weights are unavailable or if large-scale EEG transfer is the experiment focus.",
        source="https://arxiv.org/abs/2405.18765",
    ),
    ModelChoice(
        modality="biosignal",
        selected="BIOT",
        rank=1,
        evidence_basis="Biosignal Transformer built for mismatched channels, variable lengths, and missing values.",
        why_for_goal1="Best fit for heterogeneous EEG/ECG/EDA/PPG/activity streams before multimodal fusion.",
        source="https://arxiv.org/abs/2305.10351",
    ),
    ModelChoice(
        modality="time_series",
        selected="MOMENT",
        rank=1,
        evidence_basis="Open time-series foundation model covering classification, forecasting, anomaly detection, and imputation.",
        why_for_goal1="Best general fallback for physiology channels when a biosignal-specific encoder is not available.",
        source="https://arxiv.org/abs/2402.03885",
    ),
    ModelChoice(
        modality="cgm",
        selected="GluFormer",
        rank=1,
        evidence_basis="CGM foundation model trained on over 10M glucose measurements and validated across external cohorts.",
        why_for_goal1="Best match for glucose instability and personalized diabetes-risk representation learning.",
        source="https://www.nature.com/articles/s41586-025-09925-9",
    ),
    ModelChoice(
        modality="cgm",
        selected="SSM-CGM",
        rank=2,
        evidence_basis="Interpretable Mamba/state-space CGM forecasting with activity signals and temporal attribution.",
        why_for_goal1="Best implementable direction for real-time glucose forecasting and explainable counterfactuals.",
        source="https://arxiv.org/abs/2510.04386",
    ),
    ModelChoice(
        modality="ehr_structured",
        selected="Med-BERT / BEHRT",
        rank=1,
        evidence_basis="Transformer pretraining over tokenized structured EHR visit sequences (diagnoses, codes, demographics).",
        why_for_goal1="Best fit for turning longitudinal coded EHR events into patient embeddings before multimodal fusion.",
        source="https://arxiv.org/abs/1904.05342",
    ),
    ModelChoice(
        modality="ehr_timeline",
        selected="NYUTron / Foresight",
        rank=2,
        evidence_basis="Clinical-text and concept-timeline language models for outcome and trajectory prediction.",
        why_for_goal1="Best reference when notes or free-text concept sequences are available alongside structured EHR.",
        source="https://www.nature.com/articles/s41586-023-06160-y",
    ),
]


DATASET_CHOICES = [
    DatasetChoice(
        dataset="Mumtaz-MDD (depression EEG)",
        status="public, download required",
        modalities="eeg",
        goal1_use="HEADLINE screening cohort: MDD-vs-healthy resting EEG; LaBraM depression screener "
                  "(window-level AUROC 0.961 / subject-level 0.986, subject-held-out CV).",
        source="figshare 4244171 (Mumtaz et al., 2016, CC BY 4.0)",
        note="19-ch 10-20 eyes-closed resting EEG; load with loaders.load_mumtaz_mdd_dataset. "
             "Comparatively separable cohort vs MODMA (SOTA LOSO ~65%, doi:10.1093/cercor/bhae505).",
    ),
    DatasetChoice(
        dataset="eegmat (PhysioNet Mental Arithmetic)",
        status="public",
        modalities="eeg, ecg",
        goal1_use="Cognitive-workload screening cohort (rest vs serial-subtraction); ECG autonomic "
                  "AUROC 0.74, LaBraM-EEG 0.663 (within-subject state task).",
        source="https://physionet.org/content/eegmat/1.0.0/ (Zyma et al., 2019, doi:10.13026/C2JQ1P)",
        note="19-ch EEG + ECG; load with loaders.load_eegmat_dataset.",
    ),
    DatasetChoice(
        dataset="WESAD",
        status="public, download required",
        modalities="ecg, eda, emg, resp, temp, ppg, motion",
        goal1_use="Primary wearable stress/affect source; chest + wrist signals feed the stress classifier.",
        source="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
        note="Convert with scripts/convert_wesad_subject.py after kagglehub download.",
    ),
    DatasetChoice(
        dataset="DEAP",
        status="public, request access",
        modalities="eeg, peripheral physiology",
        goal1_use="EEG + peripheral source for emotion/arousal labels; exercises the EEG band-power features.",
        source="https://www.eecs.qmul.ac.uk/mmv/datasets/deap/",
        note="Convert preprocessed_python .dat files with scripts/convert_deap_subject.py.",
    ),
    DatasetChoice(
        dataset="MIMIC-IV (demo)",
        status="public demo, credentialed for full",
        modalities="ehr",
        goal1_use="Structured EHR event tokenization for the EHR ingestion and timeline-modeling path.",
        source="https://physionet.org/content/mimic-iv-demo/",
        note="Demo subset is openly downloadable; full set needs PhysioNet credentialing.",
    ),
    DatasetChoice(
        dataset="Public CGM cohorts",
        status="public, download required",
        modalities="cgm",
        goal1_use="Continuous glucose traces for the glucose forecasting baseline and prediction intervals.",
        source="https://github.com/IrinaStatsLab/Awesome-CGM",
        note="Use any 5-minute-cadence CGM export mapped to modality=cgm, channel=glucose.",
    ),
    DatasetChoice(
        dataset="Galea / EMOTIV exports",
        status="device exports, user-provided",
        modalities="eeg, physiology",
        goal1_use="Real BCI/EEG streams; convert to canonical events before modeling.",
        source="https://galea.co / https://www.emotiv.com",
        note="No public dataset; map device exports into the canonical event schema.",
    ),
]


def model_choice_table() -> pd.DataFrame:
    """Auditable registry of model choices written to model_choice_registry.csv."""
    return pd.DataFrame([choice.__dict__ for choice in MODEL_CHOICES])


def dataset_choice_table() -> pd.DataFrame:
    """Auditable registry of dataset choices written to dataset_registry.csv."""
    return pd.DataFrame([choice.__dict__ for choice in DATASET_CHOICES])