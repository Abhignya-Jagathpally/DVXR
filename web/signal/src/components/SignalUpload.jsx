import { useState } from "react";

const KIND = { csv: "tabular / CSV", json: "JSON", edf: "EEG · EDF", bdf: "EEG · BDF" };

// Client-side header preview ONLY. The synthetic build reads the first bytes to
// illustrate the ingestion flow; large biosignal files are parsed by the
// backend in the connected app.
export default function SignalUpload() {
  const [meta, setMeta] = useState(null);

  const onFile = (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    const kind = KIND[ext] || ext.toUpperCase();
    const reader = new FileReader();
    reader.onload = () => {
      const txt = String(reader.result || "");
      const lines = txt.split(/\r?\n/).filter(Boolean);
      const cols = (lines[0] || "").split(/[,\t;]/).length;
      setMeta({
        name: f.name,
        kind,
        rows: lines.length > 1 ? lines.length - 1 : "—",
        cols: cols || "—",
        size: (f.size / 1024).toFixed(1),
      });
    };
    reader.readAsText(f.slice(0, 64 * 1024));
  };

  return (
    <div className="sim-upload">
      <label className="sim-drop">
        <input
          type="file"
          className="sim-file"
          accept=".csv,.json,.edf,.bdf,.txt"
          onChange={onFile}
          aria-label="Upload a research signal file for header preview"
        />
        <span className="sim-drop-title">Drop or choose a signal file</span>
        <span className="sim-drop-sub">CSV · JSON · EDF · BDF — header preview only</span>
      </label>
      {meta && (
        <div className="sim-um">
          <div className="sim-um-row"><span>File</span><b>{meta.name}</b></div>
          <div className="sim-um-row"><span>Detected type</span><b>{meta.kind}</b></div>
          <div className="sim-um-row"><span>Rows / samples</span><b>{meta.rows}</b></div>
          <div className="sim-um-row"><span>Columns / channels</span><b>{meta.cols}</b></div>
          <div className="sim-um-row"><span>Size</span><b>{meta.size} KB</b></div>
          <p className="sim-um-note">
            Preview only. Large biosignal files are parsed by the backend in the connected app;
            this synthetic build reads headers to illustrate the flow.
          </p>
        </div>
      )}
    </div>
  );
}
