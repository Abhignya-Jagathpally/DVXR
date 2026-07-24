# VR setup — attach the DVXR stats HUD to an avatar already in your scene

This package renders a **world-space** canvas, so it shows up in any VR headset once you have a
standard XR camera rig. Below is the minimal path plus how to bind the HUD to an existing avatar.

## 1. Have an XR rig in the scene
Use either:
- **OpenXR + XR Interaction Toolkit** (recommended, device-agnostic): Package Manager → install
  *XR Interaction Toolkit* and *OpenXR*; enable OpenXR under *Project Settings → XR Plug-in
  Management*; add an *XR Origin (VR)* to the scene (it contains a *Main Camera* tagged
  `MainCamera`).
- Or any rig whose head camera is `Camera.main`. The HUD auto-finds `Camera.main`; if your head
  camera is not tagged `MainCamera`, assign it to `VRStatsHUD.cameraOverride` /
  `VRRealtimeBootstrap.cameraOverride`.

## 2. Drop in the HUD
**Fastest:** empty GameObject → add `VRRealtimeBootstrap` → set `baseUrl` → Play.

**Manual:** add `VRStatsHUD` to a GameObject (it auto-adds an `RtStreamClient`), set the client's
`baseUrl`, and pick an `anchorMode`.

## 3. Bind it to your avatar
- **Wrist readout (smart-watch feel):** set `anchorMode = FollowAnchor` and drag the avatar's
  **left-hand / wrist** bone (or an empty child of it) into `anchor`. Tune `offset` (metres) so
  the panel sits just above the wrist; it billboards to face the head camera.
- **Chest badge:** `FollowAnchor` with the avatar's **upper-chest** transform.
- **Always-in-view:** `anchorMode = HeadLockedBillboard` (default) — the panel follows the gaze
  at `offset` and always faces the wearer. Keep `offset.z` ≈ 0.7–1.0 m for eye comfort.

Recommended comfort defaults: panel width `0.30–0.40 m`, distance `0.7–1.0 m`, slightly below
eye line (`offset.y ≈ -0.1`). Avoid locking the panel too close (< 0.4 m) — it causes eye strain.

## 4. Point it at data
- **Live DVXR backend:** `baseUrl = http://<backend-host>:<port>` (serves `/v1/realtime/stream`).
- **Local mock (no pipeline needed):**
  ```bash
  python server/mock_rt_server.py --port 8000
  ```
  then `baseUrl = http://localhost:8000` (use the machine's LAN IP if the headset is standalone,
  e.g. Quest: `http://192.168.x.y:8000`).
- **No server at all:** leave `demoFallback = true`; the HUD animates from the deterministic
  generator so you can lay out the scene first.

## 5. What you'll see
A dark panel titled *"<You> · live readout"* with a live/demo dot, then:
- **EEG command** + confidence (Neutral / Left / Right / Push / Pull),
- **Stress** bar (green→red),
- **Glucose** — a value **only** when the stream provides one; otherwise
  *"abstained — insufficient data"* in amber,
- **Active skill** glyph mapped from the command,
- an always-on **EXPERIMENTAL / not-for-clinical-use** footer.

## Notes / gotchas
- **TextMeshPro not required** — the HUD uses built-in uGUI `Text` to avoid the TMP-essentials
  import step. Swap to TMP later if you want crisper text at large scale.
- **Standalone headsets** (Quest, etc.) can't reach `localhost` on your PC — use the PC's LAN IP
  and make sure the mock server binds `--host 0.0.0.0` (its default).
- The C# compiles under Unity only (it references `UnityEngine.*`); there is no standalone build
  step. The **data path** is what's unit-tested here (`server/verify_stream.py`).
- This is a **view**, not a model: it never predicts. All numbers come from the stream; the
  glucose abstention and EXPERIMENTAL caveats are enforced client-side and server-side.
