const MODES = [
  { id: "manual", label: "Manual values", help: "Enter research-profile inputs directly." },
  { id: "upload", label: "Signal upload", help: "Preview a biosignal file header." },
  { id: "sample", label: "Sample profile", help: "Load a prepared synthetic profile." },
];

export default function InputModeSelector({ mode, onMode }) {
  return (
    <div className="sim-modes" role="tablist" aria-label="Input mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          role="tab"
          className="sim-modetab"
          aria-selected={mode === m.id}
          title={m.help}
          onClick={() => onMode(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
