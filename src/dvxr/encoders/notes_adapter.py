"""NotesEHRAdapter — UNSTRUCTURED clinical notes (free text). Primary: a real clinical
transformer (Bio_ClinicalBERT) chunk-pooled over the note text; fallback: a TF-IDF +
TruncatedSVD floor. This is the free-text counterpart to the structured ``EHRAdapter``
(which embeds pseudo-text over MIMIC code timelines).
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter, _TfidfSvdBackend


class NotesEHRAdapter(BaseAdapter):
    modality = "ehr_notes"

    def _make_fallback(self):
        return _TfidfSvdBackend(self.d)
