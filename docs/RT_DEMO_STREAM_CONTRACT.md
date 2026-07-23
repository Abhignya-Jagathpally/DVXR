# RT Demo stream contract (`rt-demo-v1`)

A single JSON frame schema shared by every real-time client — the web
react-three-fiber scene and the Unity scene both consume it, so the backend work is done
once. Served over WebSocket (`/v1/realtime/stream`) and SSE (`/v1/realtime/sse`), or
generated client-side in demo mode.

> **EXPERIMENTAL / DEMONSTRATION ONLY.** Every frame carries `"experimental": true`. The
> avatar `command` channel is the `bci_real.py` decoded cube-movement analog
> (Neutral/Left/Right/Push/Pull) — not validated neural decoding. `stress` is a
> transparent heuristic. `glucose_*` **abstains by construction** (no synchronized
> per-subject EEG+CGM data exists), so it is `null` with `evidence_status: "abstained"`
> unless a clearly-labelled demo trace is supplied. Nothing here is clinical inference.

## Frame

```json
{
  "contract": "rt-demo-v1",
  "t": 0,
  "command": "Left",
  "command_confidence": 0.71,
  "stress": 0.62,
  "glucose_point": null,
  "glucose_lower": null,
  "glucose_upper": null,
  "abstained": true,
  "evidence_status": "abstained",
  "experimental": true,
  "disclaimer": "EXPERIMENTAL demonstration stream — not clinical inference."
}
```

| Field | Type | Meaning |
|---|---|---|
| `contract` | string | Schema version (`rt-demo-v1`). |
| `t` | int | Monotonic frame index. |
| `command` | enum | Avatar movement: `Neutral`\|`Left`\|`Right`\|`Push`\|`Pull` (BCI analog). |
| `command_confidence` | float [0,1] | Decoder confidence for the command. |
| `stress` | float [0,1] | Heuristic stress index. |
| `glucose_point` | float\|null | Point glucose forecast (mg/dL); `null` when abstained. |
| `glucose_lower` / `glucose_upper` | float\|null | 95% interval bounds; `null` when abstained. |
| `abstained` | bool | True when the glucose channel abstains (the default). |
| `evidence_status` | enum | `abstained` \| `experimental`. Never `validated`/`clinical`. |
| `experimental` | bool | Always `true`. |
| `disclaimer` | string | Human-readable caveat. |

## Client rules

- On abstention (`abstained: true`), the client MUST render an explicit "insufficient
  data" state for the glucose channel — never a fabricated value.
- The `experimental` badge MUST be visible whenever the scene is animating.
- Clients degrade to a deterministic in-page generator when no backend is reachable
  (demo mode), mirroring the existing `web/signal` `DEMO_MODE` pattern.
