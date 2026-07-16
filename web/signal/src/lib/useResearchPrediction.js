// SIGNAL — useResearchPrediction hook.
// State machine: idle → loading → { validation-failure | api-unavailable |
// partial | abstention | completed }.
import { useCallback, useRef, useState } from "react";
import { researchPredict, buildRequest } from "./researchPredictionApi.js";
import { GROUP_KEYS } from "./researchPredictionTypes.js";
import { groupPresence } from "./researchModel.js";

/** A request is valid only when at least one modality group has data. */
function validate(inputs) {
  const anyPresent = GROUP_KEYS.some((g) => groupPresence(inputs, g) !== "none");
  if (!anyPresent) {
    return {
      ok: false,
      message:
        "No research-profile inputs were provided. Enter at least one modality to run the synthetic model.",
    };
  }
  return { ok: true };
}

export function useResearchPrediction() {
  const [status, setStatus] = useState("idle"); // PredictionState
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const runId = useRef(0);

  const run = useCallback(async ({ sessionId, inputMode, selectedOutcome, inputs, targets }) => {
    const check = validate(inputs);
    if (!check.ok) {
      setResult(null);
      setError(check.message);
      setStatus("validation-failure");
      return;
    }
    const myId = ++runId.current;
    setError(null);
    setResult(null);
    setStatus("loading");
    try {
      const request = buildRequest({ sessionId, inputMode, selectedOutcome, inputs, targets });
      const res = await researchPredict(request);
      if (myId !== runId.current) return; // superseded by a newer run
      setResult(res);
      // status precedence: abstention > partial > completed
      const next = res.abstained
        ? "abstention"
        : res.input_quality && res.input_quality.overall === "limited"
        ? "partial"
        : "completed";
      setStatus(next);
    } catch (e) {
      if (myId !== runId.current) return;
      setError(e && e.message ? e.message : "The research backend is unavailable.");
      setStatus("api-unavailable");
    }
  }, []);

  const reset = useCallback(() => {
    runId.current++;
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return {
    status,
    result,
    error,
    run,
    reset,
    isLoading: status === "loading",
    isDone: status === "completed" || status === "partial" || status === "abstention",
  };
}
