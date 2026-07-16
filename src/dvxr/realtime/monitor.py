"""dvxr.realtime.monitor — DEPRECATED shim.

The heuristic streaming monitor moved to :mod:`dvxr.realtime.heuristic_demo` and was renamed to make
its status explicit: it is an EXPERIMENTAL heuristic demo, not the calibrated Sentinel predictor. This
shim re-exports the public names for backward compatibility and warns on import.
"""
from __future__ import annotations

import warnings

from dvxr.realtime.heuristic_demo import (  # noqa: F401
    EXPERIMENTAL_ONLY,
    NOT_FOR_CLINICAL_INFERENCE,
    FusedRealtimeMonitor,
    canonical_modalities,
    stream_fused_predictions,
)

warnings.warn(
    "dvxr.realtime.monitor moved to dvxr.realtime.heuristic_demo (it is a heuristic DEMO, not the "
    "calibrated predictor). Import from dvxr.realtime.heuristic_demo.",
    DeprecationWarning, stacklevel=2)
