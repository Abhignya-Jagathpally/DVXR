"""Verify the mock rt-demo-v1 server end-to-end (stands in for the Unity RtStreamClient).

Connects exactly like the Unity client does — ws://<host>:<port>/v1/realtime/stream?interval=
— reads a few frames, and asserts the contract the VR HUD depends on:
  * contract == "rt-demo-v1"
  * command in the BCI vocabulary; stress in [0,1]; confidence in [0,1]
  * glucose ABSTAINS by default (abstained==True, values null) — no fabricated number
Exit code 0 on success. This is the automated check for the VR data path.
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

VOCAB = {"Neutral", "Left", "Right", "Push", "Pull"}


async def main(url: str, n: int, expect_glucose: bool) -> int:
    frames = []
    async with websockets.connect(url) as ws:
        for _ in range(n):
            frames.append(json.loads(await ws.recv()))

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok  " if cond else " FAIL ") + msg)
        ok = ok and cond

    print(f"received {len(frames)} frames from {url}")
    check(all(f["contract"] == "rt-demo-v1" for f in frames), "every frame is rt-demo-v1")
    check(all(f["command"] in VOCAB for f in frames), f"command in {sorted(VOCAB)}")
    check(all(0.0 <= f["stress"] <= 1.0 for f in frames), "stress in [0,1]")
    check(all(0.0 <= f["command_confidence"] <= 1.0 for f in frames), "confidence in [0,1]")
    check(all(f["experimental"] is True for f in frames), "experimental flag always true")
    check(all(f["t"] == i for i, f in enumerate(frames)), "monotonic frame index t")

    if expect_glucose:
        check(all(not f["abstained"] and f["glucose_point"] is not None for f in frames),
              "glucose present (labelled demo trace)")
    else:
        check(all(f["abstained"] and f["glucose_point"] is None for f in frames),
              "glucose ABSTAINS (no fabricated value) — honest default")

    print("SAMPLE FRAME:", json.dumps(frames[1], indent=None))
    print("=== PASS ===" if ok else "=== FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = sys.argv[2] if len(sys.argv) > 2 else "8765"
    expect_glu = "--expect-glucose" in sys.argv
    url = f"ws://{host}:{port}/v1/realtime/stream?interval=0.05"
    sys.exit(asyncio.run(main(url, 6, expect_glu)))
