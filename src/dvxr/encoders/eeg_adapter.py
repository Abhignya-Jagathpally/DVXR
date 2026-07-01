"""EEGAdapter — EEG/BCI (Galea, EMOTIV). Primary: LaBraM (braindecode); fallback:
band-power + per-channel stats -> VQBiosignalEncoder.

Galea vs EMOTIV channel-count mismatch is handled upstream: the adapter operates on
the windowed *feature* table (band-power / per-channel stats), so variable raw channel
counts collapse to a feature vector that projects to the fixed latent width d.
"""
from __future__ import annotations

from dvxr.encoders.base import BaseAdapter


class EEGAdapter(BaseAdapter):
    modality = "eeg"
