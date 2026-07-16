import { GROUP_KEYS } from "../lib/researchPredictionTypes.js";
import { FIELDS, groupPresence } from "../lib/researchModel.js";

const ROWS = [
  ...GROUP_KEYS.map((g) => ({ g, label: FIELDS[g].label })),
  { g: "molecular", label: "Molecular", frontier: true },
];

// Right-column readiness panel: per-modality availability. Molecular is always
// "not provided" (research direction only).
export default function SignalReadiness({ inputs }) {
  return (
    <div className="sim-readiness">
      <div className="sim-panel-h">Signal readiness</div>
      <div className="sim-rrows">
        {ROWS.map((row) => {
          let level = "none";
          let text = "Not provided";
          if (!row.frontier) {
            const p = groupPresence(inputs, row.g);
            level = p === "full" ? "available" : p === "partial" ? "partial" : "none";
            text = level === "available" ? "Available" : level === "partial" ? "Partial" : "Not provided";
          }
          return (
            <div className="sim-rrow" data-level={level} key={row.g}>
              <span className="sim-rrow-dot" data-mod={row.g} aria-hidden="true" />
              <span className="sim-rrow-name">{row.label}</span>
              <span className="sim-rrow-st">{text}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
