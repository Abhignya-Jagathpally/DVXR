# DVXR VR-Realtime Stats HUD

A **drop-in, world-space VR HUD** that shows a person their **personalized DVXR stats** —
decoded BCI command, stress, and glucose (or an honest abstention) — floating beside their
**existing Unity avatar**, updating **in realtime** from the DVXR `rt-demo-v1` stream.

> The in-repo `Assets/Scripts/PhysiologyHUD.cs` is an IMGUI overlay that only shows in the flat
> game-view. This package is the **VR-native** replacement: a real world-space canvas the
> headset wearer can see in 3D. See [`GOAL.md`](GOAL.md) for the goal + acceptance criteria.

> **EXPERIMENTAL / demonstration only — not for clinical use.** Glucose **abstains by
> construction** (never fabricated); the decoded command is a single-subject engine-label demo.

## What's in the box
```
Runtime/
  RtFrame.cs               rt-demo-v1 frame + WebSocket client (RtStreamClient), demo fallback
  DemoFrameSource.cs       deterministic client-side generator (DEMO_MODE parity w/ backend)
  VRStatsHUD.cs            ← the VR-native world-space HUD (built from code, no prefab wiring)
  VRRealtimeBootstrap.cs   one-component setup that spawns + wires everything to your camera
  DVXR.VRRealtime.asmdef   assembly definition (references UnityEngine.UI)
server/
  mock_rt_server.py        standalone rt-demo-v1 server (test the HUD without the full pipeline)
  verify_stream.py         automated contract check for the data path (=== PASS ===)
docs/SETUP_VR.md           OpenXR / XR rig setup + how to attach to an avatar already in a scene
GOAL.md                    the /goal, acceptance criteria, honesty invariants
```

## Install (Unity 2021.3+)
Copy this `dvxr-vr-realtime/` folder into your project's `Packages/` (or `Assets/`), or add it
via Package Manager → *Add package from disk…* → `package.json`. It depends only on
`com.unity.ugui` (built in). For VR, have an OpenXR / XR Interaction Toolkit rig in the scene.

## Use it in 30 seconds
1. Open a scene that already has your **avatar** and an **XR Rig / camera**.
2. Create an empty GameObject → add **`VRRealtimeBootstrap`**.
3. Set **Base Url** to your DVXR backend (or the mock server), e.g. `http://localhost:8000`.
   - *FollowAnchor* mode? Drag your avatar's wrist/chest transform into **Avatar Anchor**.
4. Press **Play** / enter VR. The stats panel floats in front of you and animates in realtime.

No backend running? Leave `demoFallback` on (default) — the HUD animates from the deterministic
demo generator, so you can build the scene before wiring the server.

## Test the data path without Unity
```bash
pip install websockets
python server/mock_rt_server.py --port 8000            # serve rt-demo-v1 frames
python server/verify_stream.py localhost 8000          # -> === PASS ===  (glucose abstains)
```
Point Unity's `RtStreamClient.baseUrl` at `http://<this-host>:8000` and the same frames drive
the VR HUD. When run inside the DVXR repo, the mock server serves the backend's own
`build_frame` for byte-identical production parity; otherwise a faithful inline port.

## The frame contract (`rt-demo-v1`)
Shared with the web scene and the in-repo Unity scene — one schema everywhere. Full spec:
`../docs/RT_DEMO_STREAM_CONTRACT.md`. Client rule honored here: **on `abstained: true`, render an
explicit "insufficient data" state — never a fabricated value**, and keep the EXPERIMENTAL
badge visible while animating.

## Anchor modes
| Mode | Placement |
|---|---|
| `HeadLockedBillboard` (default) | floats at a comfortable offset in front of the headset, always facing you |
| `FollowAnchor` | parents to an avatar transform (wrist = smart-watch readout, chest = badge) |
| `WorldFixed` | stays where placed; only rotates to face the camera |
