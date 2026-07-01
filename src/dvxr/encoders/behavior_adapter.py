"""BehaviorAdapter — VR/AR behavioral streams (gaze, interactions, motion).
Primary: MOMENT-1 on behavioral time-series (real weights); fallback:
behavior features -> VQBiosignalEncoder / PCA projection to d.
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter


class BehaviorAdapter(BaseAdapter):
    modality = "behavior"
