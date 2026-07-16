import SignalCanvas from "./SignalCanvas.jsx";
import { sigDraw } from "../lib/signals.js";
import { engineSteps, fusionStrategies } from "../content.js";

export default function IntelligencePipeline() {
  return (
    <section id="engine" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal"><span className="eyebrow">The intelligence engine</span></div>
        <h2 className="display chapter-h reveal">Many signals.<br />One temporal intelligence layer.</h2>
        <div className="steps reveal">
          {engineSteps.map((s) => (
            <div className="step" key={s.n}>
              <span className="n">{s.n}</span>
              <h4>{s.title}</h4>
              <p>{s.body}</p>
              <SignalCanvas className="sig" draw={sigDraw(s.sig)} />
              <ul>{s.items.map((it) => <li key={it}>{it}</li>)}</ul>
            </div>
          ))}
        </div>
        <div className="fuse-list reveal">{fusionStrategies.map((f) => <span key={f}>{f}</span>)}</div>
        <div className="fusion-note reveal">Fusion is evaluated against the strongest unimodal baseline. It is retained <b>only</b> when it provides measurable, reproducible value — and on our real cohorts, learned fusion currently does not clear that bar. We report that openly below.</div>
      </div>
    </section>
  );
}
