// SIGNAL — Research prediction API layer.
//
// Reads configuration from Vite env vars (NO API KEY IS EVER PLACED IN CLIENT
// CODE — the real backend is expected to be a same-origin / gateway route that
// handles auth server-side):
//   VITE_RESEARCH_API_URL   base URL of the research backend
//   VITE_RESEARCH_DEMO_MODE 'true' (default) => env-gated in-page mock adapter
//
// In demo mode the response is COMPUTED by the transparent synthetic model and
// stamped as SYNTHETIC DEMONSTRATION so it can never be mistaken for a real,
// backend-served clinical result.
import { runModel } from "./researchModel.js";
import { prefersReducedMotion } from "./useReveal.js";

const RAW_DEMO = import.meta.env.VITE_RESEARCH_DEMO_MODE;
// Default ON so the static build works fully offline.
export const DEMO_MODE = (RAW_DEMO ?? "true") !== "false";
export const API_BASE = import.meta.env.VITE_RESEARCH_API_URL || "";

export const SYNTHETIC_BADGE = "SYNTHETIC DEMONSTRATION";

/** Build the request contract object from current UI state. */
export function buildRequest({ sessionId, inputMode, selectedOutcome, inputs, targets }) {
  return {
    session_id: sessionId,
    input_mode: inputMode,
    selected_outcome: selectedOutcome,
    prediction_horizons_minutes: [30, 60],
    inputs,
    targets,
  };
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Env-gated mock adapter — computes the response from the synthetic model. */
async function mockPredict(request) {
  // A short synthetic "compute" delay so the progress UI reads naturally.
  await delay(prefersReducedMotion() ? 60 : 1850);
  const res = runModel(request.inputs, request.selected_outcome);
  return {
    ...res,
    synthetic: true,
    source: "mock",
    synthetic_badge: SYNTHETIC_BADGE,
  };
}

/** Real backend call — POSTs to `${base}/v1/research/predict`. */
async function realPredict(request) {
  if (!API_BASE) {
    const err = new Error("VITE_RESEARCH_API_URL is not configured.");
    err.code = "api_unavailable";
    throw err;
  }
  let resp;
  try {
    resp = await fetch(`${API_BASE}/v1/research/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(request),
    });
  } catch (networkErr) {
    const err = new Error("Research backend is unreachable.");
    err.code = "api_unavailable";
    err.cause = networkErr;
    throw err;
  }
  if (!resp.ok) {
    const err = new Error(`Research backend returned ${resp.status}.`);
    err.code = "api_unavailable";
    throw err;
  }
  const data = await resp.json();
  return { ...data, synthetic: !!data.synthetic, source: "backend" };
}

/**
 * Predict entry point. Uses the mock adapter when DEMO_MODE is on, otherwise
 * the real backend.
 * @param {import("./researchPredictionTypes.js").ResearchPredictionRequest} request
 * @returns {Promise<import("./researchPredictionTypes.js").ResearchPredictionResponse>}
 */
export function researchPredict(request) {
  return DEMO_MODE ? mockPredict(request) : realPredict(request);
}
