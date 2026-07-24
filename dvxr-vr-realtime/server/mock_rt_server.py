"""Standalone rt-demo-v1 mock server for the DVXR VR-Realtime package.

Lets you test the Unity VR HUD WITHOUT standing up the full DVXR pipeline. It serves the
exact same `rt-demo-v1` contract at ws://<host>:<port>/v1/realtime/stream?interval=<sec>,
so `RtStreamClient.baseUrl = http://<host>:<port>` connects and the world-space HUD animates
against live frames.

Fidelity: if the real backend is importable (`dvxr.serve.realtime_bridge`), we serve its
`build_frame` verbatim — byte-identical to production. Otherwise we use a faithful inline
port. Either way the glucose channel ABSTAINS by construction (no fabricated value) unless
you pass a clearly-labelled --glucose-trace.

Usage:
    python server/mock_rt_server.py                       # abstaining glucose (honest default)
    python server/mock_rt_server.py --port 8000
    python server/mock_rt_server.py --glucose-trace 120,135,150,140,128   # labelled demo trace

Requires: websockets (`pip install websockets`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from urllib.parse import parse_qs, urlparse

import websockets

CONTRACT_VERSION = "rt-demo-v1"
DISCLAIMER = "EXPERIMENTAL demonstration stream — not clinical inference."
_PATTERN = ("Neutral", "Left", "Neutral", "Right", "Push", "Neutral", "Pull", "Neutral")

# Prefer the real backend's frame builder for exact production parity.
try:  # pragma: no cover - depends on run location
    from dvxr.serve.realtime_bridge import build_frame as _backend_build_frame  # type: ignore
    _SOURCE = "dvxr.serve.realtime_bridge.build_frame (production parity)"
except Exception:  # noqa: BLE001
    _backend_build_frame = None
    _SOURCE = "inline faithful port"


def _inline_build_frame(index: int, glucose_trace=None) -> dict:
    """Faithful port of realtime_bridge.build_frame — pure function of the index."""
    command = _PATTERN[index % len(_PATTERN)]
    confidence = 0.55 + 0.35 * (0.5 + 0.5 * math.sin(index * 0.27 + 1.3))
    stress = max(0.0, min(1.0, 0.5 + 0.35 * math.sin(index * 0.15)))
    if glucose_trace:
        point = float(glucose_trace[index % len(glucose_trace)])
        glucose = {
            "glucose_point": round(point, 1),
            "glucose_lower": round(point - 18.0, 1),
            "glucose_upper": round(point + 18.0, 1),
            "abstained": False,
            "evidence_status": "experimental",
        }
    else:
        glucose = {
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
        "command_confidence": round(confidence, 4),
        "stress": round(stress, 4),
        **glucose,
        "experimental": True,
        "disclaimer": DISCLAIMER,
    }


def build_frame(index: int, glucose_trace=None) -> dict:
    if _backend_build_frame is not None:
        return _backend_build_frame(index, glucose_trace=glucose_trace)
    return _inline_build_frame(index, glucose_trace=glucose_trace)


async def handler(websocket, glucose_trace):
    path = getattr(websocket, "path", "") or ""
    q = parse_qs(urlparse(path).query)
    try:
        interval = max(0.02, float(q.get("interval", ["0.125"])[0]))
    except (TypeError, ValueError):
        interval = 0.125
    print(f"[mock-rt] client connected path={path!r} interval={interval}s")
    index = 0
    try:
        while True:
            await websocket.send(json.dumps(build_frame(index, glucose_trace)))
            index += 1
            await asyncio.sleep(interval)
    except websockets.ConnectionClosed:
        print("[mock-rt] client disconnected")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--glucose-trace", default="",
                    help="Comma-separated mg/dL demo trace. EMPTY = abstain (honest default).")
    args = ap.parse_args()
    trace = [float(x) for x in args.glucose_trace.split(",") if x.strip()] or None

    print(f"[mock-rt] rt-demo-v1 via {_SOURCE}")
    print(f"[mock-rt] glucose: {'labelled demo trace' if trace else 'ABSTAINING (honest default)'}")
    print(f"[mock-rt] serving ws://{args.host}:{args.port}/v1/realtime/stream")
    print("[mock-rt] point Unity RtStreamClient.baseUrl at "
          f"http://<this-host>:{args.port}")

    async with websockets.serve(lambda ws: handler(ws, trace), args.host, args.port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[mock-rt] stopped")
