# /goal — a person's avatar in Unity sees their personalized DVXR stats in VR, in realtime

## Goal statement
A person who **already has an avatar in a Unity scene** puts on a VR headset and sees **their
own personalized stats** — decoded BCI command, stress, and glucose (or an honest abstention)
— rendered on a **world-space HUD beside/attached to their avatar**, updating **in realtime**
from the DVXR stream. Drop-in: no bespoke backend, no manual Canvas wiring.

## Why this is a distinct piece of work
The DVXR repo already had a Unity readout (`Assets/Scripts/PhysiologyHUD.cs`), but it uses
**IMGUI / `OnGUI`**, which draws to the flat editor game-view and **is not visible in a VR
headset**. VR requires a **world-space canvas** the wearer can look at in 3D space. This
subproject supplies exactly that VR-native layer, reusing the existing `rt-demo-v1` stream
contract so no backend work is duplicated.

## Acceptance criteria
| # | Criterion | Status |
|---|---|---|
| 1 | World-space (VR-visible) HUD, not a screen-space/IMGUI overlay | ✅ `VRStatsHUD.cs` builds a `RenderMode.WorldSpace` uGUI canvas |
| 2 | Attaches to an avatar **already** in the scene (head-locked, wrist-follow, or world-fixed) | ✅ `HudAnchorMode` + `anchor`/`cameraOverride` |
| 3 | Realtime updates from the DVXR stream (`rt-demo-v1`) | ✅ `RtStreamClient` (WebSocket `/v1/realtime/stream`) |
| 4 | Zero manual wiring — one component, press Play | ✅ `VRRealtimeBootstrap.cs` / attach `VRStatsHUD` |
| 5 | Works with **no** backend (deterministic demo fallback) | ✅ `DemoFrameSource.cs`, `RtStreamClient.demoFallback` |
| 6 | Testable **without** Unity (data path verified) | ✅ `server/mock_rt_server.py` + `server/verify_stream.py` → PASS |
| 7 | Personalized to the wearer | ✅ per-session stream + `participantLabel`; per-patient kernel lives in the backend model |

## Honesty invariants (non-negotiable — carried from the DVXR framework)
- **Glucose abstains by construction.** When `abstained == true` the HUD shows
  `"abstained — insufficient data"` in amber — **never a fabricated number**. A glucose value
  appears only from a clearly-labelled demo trace. (Verified: `verify_stream.py`.)
- **EXPERIMENTAL badge always visible** while the scene animates.
- The decoded `command` is a **single-subject / EMOTIV engine-label** demonstration control —
  **not validated neural intent**.
- **Not for clinical use** (`validated_for_clinical_use = false` upstream).
- Galea/EMOTIV device data is **schema-only** upstream (ingestion demo, not training/validation).

## What "done" looks like (verification)
```bash
# data path (no Unity needed):
python server/mock_rt_server.py --port 8000                 # serve rt-demo-v1
python server/verify_stream.py localhost 8000               # -> === PASS ===  (glucose abstains)
python server/mock_rt_server.py --port 8000 --glucose-trace 120,135,150
python server/verify_stream.py localhost 8000 --expect-glucose   # -> === PASS ===
```
In Unity: add `VRRealtimeBootstrap` to any XR scene with a camera, set `baseUrl`, press Play →
the world-space stats panel floats in front of the headset and animates. See `docs/SETUP_VR.md`.

## Not in scope (honest boundaries)
- No new prediction model — this is a **view** over the existing stream.
- No synchronized per-subject EEG+CGM cohort exists, so the fused EEG→glucose readout **stays
  abstaining** by default; that is a data-availability limit, not a UI gap.
- The `.unity` scene / prefabs are intentionally **not** shipped as binaries; the HUD is built
  from code so it drops into *your* avatar scene without asset-GUID conflicts.
