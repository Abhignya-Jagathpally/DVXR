# Unity RT Demo (`DVXR_RT_Demo`)

A Unity scene + C# client that consumes the **same** `rt-demo-v1` stream contract
(`docs/RT_DEMO_STREAM_CONTRACT.md`) as the web react-three-fiber scene. The avatar cube
translates per the decoded BCI command, tints by the stress index, and shows the glucose
**abstention** state — it never fabricates a glucose value.

> **Authored headless — validate on first Editor open.** This scaffold was written on a
> Linux box with no Unity Editor, so it cannot be compiled or play-tested here. The scene
> YAML uses Unity's built-in cube/material fileIDs (no custom GUIDs to break), and the two
> scripts are intentionally **not** pre-wired into the scene so there are no dangling
> MonoBehaviour GUID references. You attach them in the Editor (two clicks, below). Open
> the scene once in the Editor to let Unity regenerate metadata.

## Files

```
Assets/Scenes/DVXR_RT_Demo.unity        Camera + light + Avatar cube + Ground + GlucoseHalo ring
Assets/Scripts/RTStreamClient.cs        WebSocket client → parses rt-demo-v1 frames
Assets/Scripts/AvatarController.cs      Minimal command→movement controller
Assets/Scripts/DigitalTwinController.cs Avatar as a physiological twin: stress→tint+breathing, glucose→halo
Assets/Scripts/SkillSystem.cs           Decoded EEG command → BCI skills (Focus/Ward/Surge/Recover), cooldowns
Assets/Scripts/PhysiologyHUD.cs         In-Editor IMGUI readout of the fused EEG+PHR+PPG prediction (no Canvas wiring)
```

The prediction the twin visualises is the framework's fused read from **EEG (Galea/EMOTIV)
+ PHR (wearable) + PPG (pulse)** — decoded command, stress, and the glucose forecast (which
abstains rather than inventing a value).

## Setup (in the Unity Editor)

1. Copy `Assets/` into a Unity project (2021 LTS or newer; Built-in or URP). Open
   `Assets/Scenes/DVXR_RT_Demo.unity` — you get the Avatar, a Ground plane, and a flat
   GlucoseHalo ring under the avatar.
2. On the **Avatar**: **Add Component → RT Stream Client** (set `Base Url`, e.g.
   `http://localhost:8000`), **→ Digital Twin Controller** (drag the Avatar's RT Stream
   Client into `Client`; drag the **GlucoseHalo** into `Glucose Halo`), and **→ Skill System**
   (drag in the same client).
3. On any GameObject: **Add Component → Physiology HUD**, and drag in the client + skill
   system. The HUD draws itself via IMGUI — no Canvas/UI setup needed.
4. Start the backend so the stream is live:

   ```bash
   uvicorn "dvxr.serve.api:app" --factory --port 8000
   # frames at ws://localhost:8000/v1/realtime/stream
   ```

5. Press **Play**. The twin translates with the decoded EEG command, **breathes and reddens**
   with PHR stress, its **glucose halo greys and stills** when the model abstains, decoded
   commands fire **skills** with cooldowns, and the HUD shows the live EEG/PHR/glucose/skill
   read-out with the EXPERIMENTAL caveat.

## Notes

- **Dependencies:** `RTStreamClient` uses `System.Net.WebSockets.ClientWebSocket` from the
  .NET/Mono runtime — no third-party package required. For an on-device LSL path instead
  of WebSocket, add `LSL4Unity` and read the `eeg` / `wearable` / `reference_glucose`
  streams defined in `neuroglycemic-sentinel/config/lsl_streams.json`.
- **Honesty:** the client keys off the `abstained` flag (Unity's `JsonUtility` maps a JSON
  `null` to `0`, so the numeric glucose fields are only meaningful when `abstained` is
  false). It never shows a glucose number the server abstained on. Every frame is
  `experimental: true` — demonstration only, not clinical inference.
- The web scene (`web/signal/src/components/rtdemo/`) is the runnable, screenshot-verifiable
  equivalent; this Unity scaffold satisfies the same contract for an XR/Editor deployment.

## Skill system (DVXR-lab-aligned digital twin)

The avatar is framed as a **BCI digital twin** whose physiology (predicted stress/glucose)
drives the scene, and whose decoded commands become **skills** — aligned to the DVXR Lab's
BCI + Digital-Twin + immersive-training work (e.g. VR Fire Evacuation with a BCI-driven
avatar). The command→skill mapping is shared with the web scene (`web/signal/src/components/rtdemo/skills.js`):

| Command | Skill | Effect |
|---|---|---|
| Neutral | 🧘 Focus | baseline calm; lowers twin stress |
| Left | 🛡️ Ward Left | left-side guard |
| Right | 🛡️ Ward Right | right-side guard |
| Push | ⚡ Surge | forward burst (high effort) |
| Pull | 💠 Recover | pull back / recover |

A skill fires only when the decoded command matches with confidence ≥ 0.6 and its cooldown
has elapsed. `AvatarController.cs` already maps command→transform; extend it with the same
cooldown/confidence gate to drive a Unity skill HUD. The twin's glucose ring **abstains**
(no value) when the stream reports insufficient data.

**Honesty caveat (keep visible in the HUD):** the command channel is a single-subject EMOTIV
mental-command engine label (~0.82 4-class, trial-grouped) — a demonstration control signal,
not validated neural intent.
