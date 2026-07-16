import { OUTCOMES } from "../lib/researchPredictionTypes.js";

// Select the primary research outcome (3 options).
export default function TargetSelector({ selected, onSelect }) {
  return (
    <div className="sim-targets" role="radiogroup" aria-label="Primary research outcome">
      <div className="sim-panel-h">Primary outcome</div>
      {OUTCOMES.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={selected === o.id}
          className="sim-target"
          onClick={() => onSelect(o.id)}
        >
          <span className="sim-target-dot" aria-hidden="true" />
          <span className="sim-target-text">
            <b>{o.label}</b>
            <small>{o.help}</small>
          </span>
        </button>
      ))}
    </div>
  );
}
