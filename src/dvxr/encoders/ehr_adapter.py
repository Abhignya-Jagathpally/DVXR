"""EHRAdapter — structured EHR + notes. Primary: a CEHR-BERT-style local encoder on
MIMIC code timelines (no public weights) with Bio_ClinicalBERT for notes; fallback:
tokenized-code timeline features -> PCA projection.
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter, _PCABackend


class EHRAdapter(BaseAdapter):
    modality = "ehr"

    def _make_fallback(self):
        return _PCABackend(self.d, tag="ehr_code_timeline")
