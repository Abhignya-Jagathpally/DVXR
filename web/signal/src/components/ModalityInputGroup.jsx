import { useState } from "react";
import { FIELDS, clamp, groupPresence } from "../lib/researchModel.js";

// Shared collapsible modality input group. Renders one FieldRow per field with:
// slider + editable numeric + unit + one-line help + range validation +
// per-field "not provided" toggle + per-field reset.
function FieldRow({ groupKey, field, entry, onChange }) {
  const idb = `${groupKey}_${field.k}`;
  const missing = !entry.present;

  const setValue = (raw) => {
    let v = parseFloat(raw);
    if (isNaN(v)) v = field.d;
    v = clamp(v, field.min, field.max);
    onChange(groupKey, field.k, { value: v, present: true });
  };
  const toggleMissing = (e) => {
    onChange(groupKey, field.k, { value: entry.value, present: !e.target.checked });
  };
  const reset = () => {
    onChange(groupKey, field.k, { value: field.d, present: true });
  };

  return (
    <div className={"sim-fld" + (missing ? " is-missing" : "")}>
      <div className="sim-fld-top">
        <label className="sim-fld-l" htmlFor={idb}>
          {field.l}
          {field.u ? <i>{field.u}</i> : null}
        </label>
        <label className="sim-miss">
          <input
            type="checkbox"
            className="sim-miss-cb"
            checked={missing}
            onChange={toggleMissing}
            aria-label={`Mark ${field.l} not provided`}
          />
          <span>n/a</span>
        </label>
      </div>
      <div className="sim-fld-ctl">
        <input
          type="range"
          id={idb}
          min={field.min}
          max={field.max}
          step={field.step}
          value={entry.value}
          disabled={missing}
          aria-describedby={`${idb}_h`}
          onChange={(e) => setValue(e.target.value)}
        />
        <input
          type="number"
          className="sim-fld-num"
          min={field.min}
          max={field.max}
          step={field.step}
          value={entry.value}
          disabled={missing}
          aria-label={`${field.l} value`}
          onChange={(e) => setValue(e.target.value)}
        />
        <button
          className="sim-fld-reset"
          type="button"
          title={`Reset ${field.l}`}
          aria-label={`Reset ${field.l}`}
          onClick={reset}
        >
          ⟲
        </button>
      </div>
      <div className="sim-fld-h" id={`${idb}_h`}>
        {field.h}
      </div>
    </div>
  );
}

export default function ModalityInputGroup({ groupKey, inputs, onChange, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const spec = FIELDS[groupKey];
  const presence = groupPresence(inputs, groupKey);
  const stateLabel = presence === "full" ? "Complete" : presence === "partial" ? "Partial" : "Not set";

  return (
    <div className="sim-ig" style={{ "--m": spec.color }} data-open={open}>
      <button
        type="button"
        className="sim-ig-sum"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="sim-ig-dot" aria-hidden="true" />
        <span className="sim-ig-name">{spec.label}</span>
        <span className={"sim-ig-state " + presence}>{stateLabel}</span>
        <span className="sim-ig-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <div className="sim-ig-body">
          {spec.fields.map((f) => (
            <FieldRow
              key={f.k}
              groupKey={groupKey}
              field={f}
              entry={inputs[groupKey][f.k]}
              onChange={onChange}
            />
          ))}
        </div>
      )}
    </div>
  );
}
