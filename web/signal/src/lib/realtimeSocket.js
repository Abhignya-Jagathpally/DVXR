// Real-time RT-Demo client for the rt-demo-v1 contract.
//
// Mirrors the DEMO_MODE pattern of researchPredictionApi.js: when demo mode is on (the
// default) or no backend is configured, it runs a deterministic in-page frame generator
// that reproduces the Python bridge's math exactly (dvxr/serve/realtime_bridge.py), so the
// scene animates identically in a static build. Otherwise it connects to the WebSocket
// endpoint and falls back to the local generator on any error.
//
// EXPERIMENTAL / DEMONSTRATION ONLY — every frame carries experimental:true; the glucose
// channel abstains by construction.

const RAW_DEMO = import.meta.env.VITE_RESEARCH_DEMO_MODE;
export const RT_DEMO_MODE = (RAW_DEMO ?? "true") !== "false";
export const RT_API_BASE = import.meta.env.VITE_RESEARCH_API_URL || "";
export const CONTRACT_VERSION = "rt-demo-v1";
export const EXPERIMENTAL_BADGE = "EXPERIMENTAL · SYNTHETIC DEMONSTRATION";

const COMMAND_PATTERN = [
  "Neutral", "Left", "Neutral", "Right", "Push", "Neutral", "Pull", "Left", "Neutral",
];

// Deterministic frame — a pure function of the index, matching the Python bridge.
export function buildFrame(index) {
  const command = COMMAND_PATTERN[index % COMMAND_PATTERN.length];
  const confidence = round(0.55 + 0.35 * Math.abs(Math.sin(index / 5)), 4);
  const stress = round(0.5 + 0.38 * Math.sin(index / 7), 4);
  return {
    contract: CONTRACT_VERSION,
    t: index,
    command,
    command_confidence: confidence,
    stress,
    glucose_point: null,
    glucose_lower: null,
    glucose_upper: null,
    abstained: true,
    evidence_status: "abstained",
    experimental: true,
    disclaimer: "EXPERIMENTAL demonstration stream — not clinical inference.",
  };
}

function round(x, n) {
  const f = 10 ** n;
  return Math.round(x * f) / f;
}

function wsUrl(base) {
  if (!base) return "";
  const url = base.replace(/^http/, "ws");
  return `${url}/v1/realtime/stream`;
}

// Local deterministic generator: calls onFrame every intervalMs. Returns a stop function.
function startLocal(onFrame, intervalMs) {
  let index = 0;
  const id = setInterval(() => {
    onFrame(buildFrame(index));
    index += 1;
  }, intervalMs);
  return () => clearInterval(id);
}

/**
 * Subscribe to RT-Demo frames. Returns an unsubscribe function.
 * @param {(frame:object)=>void} onFrame
 * @param {{intervalMs?:number, source?:'auto'|'local'|'ws'}} [opts]
 */
export function subscribeFrames(onFrame, opts = {}) {
  const intervalMs = opts.intervalMs ?? 120;
  const source = opts.source ?? "auto";
  const useLocal = source === "local" || (source === "auto" && (RT_DEMO_MODE || !RT_API_BASE));

  if (useLocal) {
    return startLocal(onFrame, intervalMs);
  }

  // Live WebSocket with graceful fallback to the local generator.
  let stopLocal = null;
  let ws = null;
  try {
    ws = new WebSocket(wsUrl(RT_API_BASE));
    ws.onmessage = (ev) => {
      try {
        onFrame(JSON.parse(ev.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onerror = () => {
      if (!stopLocal) stopLocal = startLocal(onFrame, intervalMs);
    };
    ws.onclose = () => {
      if (!stopLocal) stopLocal = startLocal(onFrame, intervalMs);
    };
  } catch {
    stopLocal = startLocal(onFrame, intervalMs);
  }

  return () => {
    if (ws) {
      try { ws.close(); } catch { /* noop */ }
    }
    if (stopLocal) stopLocal();
  };
}
