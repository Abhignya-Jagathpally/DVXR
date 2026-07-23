"""Real-time RT-Demo streaming bridge (shared by the web and Unity clients).

Emits ``rt-demo-v1`` contract frames (see ``docs/RT_DEMO_STREAM_CONTRACT.md``) over an
async generator that a WebSocket or SSE endpoint drains. Two frame sources:

  * ``heuristic`` (default) — a deterministic, dependency-free generator. No randomness,
    no wall clock (frames are a pure function of the frame index), so it is reproducible
    and testable. The avatar ``command`` channel reuses ``bci_real.COMMAND_CLASSES``.
  * ``lsl`` — a live/replay tap over ``neuroglycemic.lsl`` when ``pylsl`` is installed
    (absent on this headless host); guarded so importing this module never requires it.

EXPERIMENTAL / DEMONSTRATION ONLY. Every frame carries ``experimental: True``. The glucose
channel abstains by construction (``abstained: True``, values ``None``) unless a clearly
labelled demo trace is supplied — no synchronized per-subject EEG+CGM data exists.
"""

from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Iterator, List, Optional, Sequence

from dvxr.bci_real import COMMAND_CLASSES

CONTRACT_VERSION = "rt-demo-v1"
DISCLAIMER = "EXPERIMENTAL demonstration stream — not clinical inference."

# A fixed, deterministic decoded-command pattern (the bci_real avatar analog). A real
# decoded sequence from bci_real can be substituted; the pattern here keeps the demo
# reproducible without a model at request time.
_DEFAULT_COMMAND_PATTERN: Sequence[str] = (
    "Neutral", "Left", "Neutral", "Right", "Push", "Neutral", "Pull", "Left", "Neutral",
)


def _stress_at(index: int) -> float:
    """Smooth deterministic stress index in [0, 1]."""
    return round(0.5 + 0.38 * math.sin(index / 7.0), 4)


def _confidence_at(index: int) -> float:
    """Deterministic command confidence in [0.55, 0.9]."""
    return round(0.55 + 0.35 * abs(math.sin(index / 5.0)), 4)


def build_frame(
    index: int,
    *,
    command_pattern: Sequence[str] = _DEFAULT_COMMAND_PATTERN,
    glucose_trace: Optional[Sequence[float]] = None,
) -> dict:
    """One ``rt-demo-v1`` frame. Pure function of ``index`` (reproducible).

    ``glucose_trace`` is an optional, clearly-labelled experimental demo series; when it is
    ``None`` (the default) the glucose channel abstains — no fabricated value.
    """
    command = command_pattern[index % len(command_pattern)]
    if glucose_trace:
        point = float(glucose_trace[index % len(glucose_trace)])
        frame_glucose = {
            "glucose_point": round(point, 1),
            "glucose_lower": round(point - 18.0, 1),
            "glucose_upper": round(point + 18.0, 1),
            "abstained": False,
            "evidence_status": "experimental",
        }
    else:
        frame_glucose = {
            "glucose_point": None,
            "glucose_lower": None,
            "glucose_upper": None,
            "abstained": True,
            "evidence_status": "abstained",
        }
    return {
        "contract": CONTRACT_VERSION,
        "t": int(index),
        "command": command,
        "command_confidence": _confidence_at(index),
        "stress": _stress_at(index),
        **frame_glucose,
        "experimental": True,
        "disclaimer": DISCLAIMER,
    }


def frame_iter(count: int, **kwargs) -> Iterator[dict]:
    """Yield ``count`` deterministic frames (synchronous; for tests/replay)."""
    for index in range(count):
        yield build_frame(index, **kwargs)


async def stream_frames(
    *,
    count: Optional[int] = None,
    interval_seconds: float = 0.1,
    source: str = "heuristic",
    glucose_trace: Optional[Sequence[float]] = None,
) -> AsyncIterator[dict]:
    """Async generator of contract frames.

    ``source='heuristic'`` (default) is dependency-free. ``source='lsl'`` requires the
    optional ``pylsl`` acquisition stack; it raises a clear error when absent rather than
    silently faking a live device.
    """
    if source == "lsl":
        _require_lsl()
    index = 0
    while count is None or index < count:
        yield build_frame(index, glucose_trace=glucose_trace)
        index += 1
        await asyncio.sleep(max(0.0, interval_seconds))


def _require_lsl() -> None:
    try:
        import pylsl  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "source='lsl' needs the optional pylsl acquisition stack "
            "(neuroglycemic-sentinel[acquisition]); it is not installed."
        ) from exc


def available_commands() -> List[str]:
    """The avatar command vocabulary (bci_real analog)."""
    return list(COMMAND_CLASSES)
