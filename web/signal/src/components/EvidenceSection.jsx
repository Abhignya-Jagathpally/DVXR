import { metrics, evidenceCategories, honestNegative } from "../content.js";

export default function EvidenceSection() {
  return (
    <section id="evidence" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal"><span className="eyebrow">Research evidence</span></div>
        <h2 className="display chapter-h reveal">Built to be tested.<br />Not just demonstrated.</h2>

        <div className="metricrow reveal">
          {metrics.map((m) => (
            <div className="mcard" key={m.cap}>
              <div className="cap">{m.cap}</div>
              <div className="v">{m.value}{m.unit ? <span style={{ fontSize: "14px", color: "var(--gray)" }}>{m.unit}</span> : null}</div>
              <div className="ci">{m.detail}</div>
              <div className="track"><i style={{ width: m.pct + "%" }} /></div>
              <div className="src">{m.src}</div>
            </div>
          ))}
        </div>

        <div className="evgrid reveal">
          {evidenceCategories.map((cat) => (
            <div className="evcard" key={cat.title}>
              <h4>{cat.title}</h4>
              <ul>{cat.items.map((it) => <li key={it}>{it}</li>)}</ul>
            </div>
          ))}
        </div>

        <div className="negbox reveal">
          <h3 className="display" style={{ textTransform: "uppercase" }}>The honest negative result</h3>
          <p className="lead">{honestNegative.lead.split("does not beat")[0]}does <b style={{ color: "var(--white)" }}>not</b> beat{honestNegative.lead.split("does not beat")[1]}</p>
          <div className="scroll"><table className="negtable">
            <thead><tr><th>Task</th><th>Metric</th><th>Best baseline</th><th>Learned fusion</th><th>Rel. error reduction</th></tr></thead>
            <tbody>
              {honestNegative.rows.map((r) => (
                <tr key={r.task}><td>{r.task}</td><td>{r.metric}</td><td>{r.baseline}</td><td>{r.fusion}</td><td><span className="neg">{r.rer}</span></td></tr>
              ))}
            </tbody>
          </table></div>
          <div className="hold">
            {honestNegative.holds.map((h) => (
              <div className="h" key={h.v}><div className="v">{h.v}</div><div className="l">{h.l}</div></div>
            ))}
          </div>
        </div>

        <p className="thesis reveal">{honestNegative.thesis}</p>
      </div>
    </section>
  );
}
