from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EncoderRecommendation:
    modality: str
    primary: str
    fallback: str
    reason: str


RECOMMENDATIONS = [
    EncoderRecommendation(
        modality="eeg",
        primary="EEG-X",
        fallback="LaBraM or BENDR",
        reason="Device-agnostic EEG is the highest priority because Galea and EMOTIV differ in channels, noise, and sampling.",
    ),
    EncoderRecommendation(
        modality="wearable_physiology",
        primary="BIOT",
        fallback="MOMENT",
        reason="BIOT handles mismatched biosignal channels; MOMENT is a strong generic time-series fallback.",
    ),
    EncoderRecommendation(
        modality="cgm",
        primary="GluFormer",
        fallback="Ridge/SSM-CGM-style forecaster",
        reason="CGM has strong temporal structure and benefits from self-supervised glucose representations.",
    ),
    EncoderRecommendation(
        modality="ehr",
        primary="Med-BERT/BEHRT",
        fallback="Foresight-style timeline features",
        reason="Structured EHR is best treated as longitudinal patient event sequences before fusion.",
    ),
]


class FeatureEncoder:
    """Small local encoder used until foundation-model weights are wired in."""

    def __init__(self, max_components: int = 24):
        self.max_components = max_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.columns: list[str] = []

    def fit_transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.columns = columns
        matrix = self.scaler.fit_transform(frame[columns])
        components = min(self.max_components, matrix.shape[0], matrix.shape[1])
        self.pca = PCA(n_components=components, random_state=7)
        encoded = self.pca.fit_transform(matrix)
        return _embedding_frame(encoded, frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pca is None:
            raise RuntimeError("FeatureEncoder must be fitted before transform")
        aligned = frame.reindex(columns=self.columns, fill_value=0.0)
        matrix = self.scaler.transform(aligned)
        encoded = self.pca.transform(matrix)
        return _embedding_frame(encoded, frame.index)


def recommendation_table() -> pd.DataFrame:
    return pd.DataFrame([rec.__dict__ for rec in RECOMMENDATIONS])


def _embedding_frame(encoded: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(encoded, index=index, columns=[f"embed_{i:02d}" for i in range(encoded.shape[1])])
