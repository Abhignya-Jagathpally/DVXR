import SignalCanvas from "./SignalCanvas.jsx";
import { worldDraw } from "../lib/signals.js";

export default function SignalWorld({ world }) {
  const inLabel = world.inputsLabel || "Inputs";
  const fnLabel = world.functionsLabel || "Research functions";
  return (
    <div className="world"><div className="wrap pad-y"><div className="grid">
      <div className="txt reveal">
        <div>
          <span className="idx">{world.idx}</span>
          <span className="name">{world.name}</span>
          {world.frontier ? <> &nbsp;<span className="pill frontier"><span className="d" />Research frontier</span></> : null}
        </div>
        <h3 className="display chapter-h stmt">{world.statement}</h3>
        <div className="cols">
          <div className="col"><h4>{inLabel}</h4><ul>{world.inputs.map((x) => <li key={x}>{x}</li>)}</ul></div>
          <div className="col"><h4>{fnLabel}</h4><ul>{world.functions.map((x) => <li key={x}>{x}</li>)}</ul></div>
        </div>
        {world.note ? <p className="mono" style={{ fontSize: "11.5px", color: world.frontier ? "var(--amber)" : "var(--faint)", marginTop: "20px", maxWidth: "52ch" }}>{world.note}</p> : null}
      </div>
      <div className="viz reveal"><SignalCanvas draw={worldDraw(world.viz)} /></div>
    </div></div></div>
  );
}
