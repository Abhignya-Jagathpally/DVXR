"""CGMAdapter — continuous glucose. Primary: CGM-JEPA / Chronos-Bolt / MOMENT
(real weights); fallback: conformalized-Ridge-style latent summary
(mean, CV, MAGE, time-in-range, slope).
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter, _CGMSummaryBackend


class CGMAdapter(BaseAdapter):
    modality = "cgm"

    def _make_fallback(self):
        return _CGMSummaryBackend(self.d)
