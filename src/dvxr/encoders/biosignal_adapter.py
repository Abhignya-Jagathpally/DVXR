"""BiosignalAdapter — wearable physiology (HRV, GSR/EDA, resp, PPG, motion).
Primary: MOMENT-1 (momentfm, real weights); fallback: VQBiosignalEncoder / PCA.
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter


class BiosignalAdapter(BaseAdapter):
    modality = "wearable_phys"
