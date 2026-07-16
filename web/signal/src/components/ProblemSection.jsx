import { problem } from "../content.js";

export default function ProblemSection() {
  return (
    <section id="problem" className="pad-y">
      <div className="wrap">
        <div className="lines">
          <div className="display big reveal">{problem.lines[0]}</div>
          <div className="display big muted reveal">{problem.lines[1]}</div>
        </div>
        <p className="para reveal">{problem.paras[0]}</p>
        <p className="para reveal" style={{ color: "var(--white)" }}>{problem.paras[1]}</p>
        <div className="display connect reveal" style={{ fontSize: "clamp(24px,4.4vw,54px)", marginTop: "34px" }}>{problem.connect}</div>
      </div>
    </section>
  );
}
