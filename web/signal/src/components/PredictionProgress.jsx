import { useEffect, useState } from "react";
import { prefersReducedMotion } from "../lib/useReveal.js";
import { SYNTHETIC_BADGE } from "../lib/researchPredictionApi.js";

const STEPS = [
  "Validating research inputs",
  "Constructing per-modality features",
  "Scoring task-specific models",
  "Composing the diabetes context model",
  "Deriving model contributions",
];

// Animated staged progress shown while the (mock or real) model runs.
export default function PredictionProgress() {
  const [active, setActive] = useState(0);
  useEffect(() => {
    if (prefersReducedMotion()) {
      setActive(STEPS.length);
      return;
    }
    const iv = setInterval(() => {
      setActive((i) => Math.min(i + 1, STEPS.length));
    }, 360);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="sim-progress" role="status" aria-live="polite">
      <div className="sim-progress-badge">{SYNTHETIC_BADGE}</div>
      <div className="sim-progress-title">Generating research profile…</div>
      <ol className="sim-progress-list">
        {STEPS.map((s, i) => (
          <li key={s} className={i < active ? "done" : i === active ? "active" : ""}>
            <span className="sim-progress-n">{String(i + 1).padStart(2, "0")}</span>
            {s}
          </li>
        ))}
      </ol>
    </div>
  );
}
