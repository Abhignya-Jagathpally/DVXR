"""OmicsAdapter — multi-omics panel. Primary: Geneformer (real weights); fallback:
build_omics_features-style panel -> linear (PCA) projection to d.
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter, _PCABackend


class OmicsAdapter(BaseAdapter):
    modality = "omics"

    def _make_fallback(self):
        return _PCABackend(self.d, tag="omics_linear")
