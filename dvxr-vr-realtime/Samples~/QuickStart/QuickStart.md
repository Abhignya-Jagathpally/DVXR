# Quick Start

1. Scene with an **XR Rig** (OpenXR *XR Origin (VR)*) + your **avatar**.
2. Empty GameObject → add **`VRRealtimeBootstrap`**.
3. `Base Url` = `http://localhost:8000` (or leave the demo fallback on).
4. Optional: `Anchor Mode = FollowAnchor`, drag the avatar's **left wrist** into `Avatar Anchor`.
5. Press **Play** / enter VR.

Test data with no pipeline:
```bash
python ../../server/mock_rt_server.py --port 8000
python ../../server/verify_stream.py localhost 8000     # === PASS ===
```

EXPERIMENTAL demonstration — glucose abstains by construction; not for clinical use.
