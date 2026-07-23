"""The RT-Demo streaming bridge honors the rt-demo-v1 contract and stays honest.

Frames are a pure function of the frame index (deterministic), glucose abstains by
construction, and every frame is flagged experimental.
"""

from __future__ import annotations

import unittest

from dvxr.serve.realtime_bridge import (
    CONTRACT_VERSION,
    available_commands,
    build_frame,
    frame_iter,
)

_REQUIRED = {
    "contract", "t", "command", "command_confidence", "stress",
    "glucose_point", "glucose_lower", "glucose_upper", "abstained",
    "evidence_status", "experimental", "disclaimer",
}


class FrameContract(unittest.TestCase):
    def test_frame_shape_and_ranges(self):
        frame = build_frame(3)
        self.assertEqual(set(frame), _REQUIRED)
        self.assertEqual(frame["contract"], CONTRACT_VERSION)
        self.assertIn(frame["command"], available_commands())
        self.assertTrue(0.0 <= frame["command_confidence"] <= 1.0)
        self.assertTrue(0.0 <= frame["stress"] <= 1.0)
        self.assertTrue(frame["experimental"])

    def test_glucose_abstains_by_construction(self):
        frame = build_frame(0)
        self.assertTrue(frame["abstained"])
        self.assertIsNone(frame["glucose_point"])
        self.assertIsNone(frame["glucose_lower"])
        self.assertIsNone(frame["glucose_upper"])
        self.assertEqual(frame["evidence_status"], "abstained")

    def test_labelled_demo_trace_never_claims_validation(self):
        frame = build_frame(0, glucose_trace=[120.0, 140.0])
        self.assertFalse(frame["abstained"])
        self.assertEqual(frame["glucose_point"], 120.0)
        self.assertEqual(frame["evidence_status"], "experimental")
        self.assertNotIn(frame["evidence_status"], {"validated", "clinical"})

    def test_frames_are_deterministic(self):
        self.assertEqual(build_frame(5), build_frame(5))
        seq = list(frame_iter(4))
        self.assertEqual([f["t"] for f in seq], [0, 1, 2, 3])


class WebSocketRoute(unittest.TestCase):
    def test_ws_streams_experimental_contract_frames(self):
        try:
            from starlette.testclient import TestClient
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"starlette testclient unavailable: {exc}")
        from dvxr.serve.api import create_app

        app = create_app(unsafe_dev=True)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/stream?count=3&interval=0") as ws:
                frames = [ws.receive_json() for _ in range(3)]
        self.assertEqual([f["t"] for f in frames], [0, 1, 2])
        for frame in frames:
            self.assertTrue(frame["experimental"])
            self.assertTrue(frame["abstained"])
            self.assertIsNone(frame["glucose_point"])
            self.assertIn(frame["command"], available_commands())


if __name__ == "__main__":
    unittest.main()
