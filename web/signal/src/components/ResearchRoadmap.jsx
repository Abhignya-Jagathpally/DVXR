import { roadmap } from "../content.js";

export default function ResearchRoadmap() {
  return (
    <section id="roadmap" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal"><span className="eyebrow">Research roadmap</span></div>
        <h2 className="display chapter-h reveal">From signals to<br />translational evidence.</h2>
        <div className="road reveal">
          {roadmap.map((p) => (
            <div className={"phase " + p.cls} key={p.ph}>
              <div className="st">{p.state}</div>
              <div className="ph">{p.ph}</div>
              <h4>{p.title}</h4>
              {p.question ? <p className="q">{p.question}</p> : null}
              <ul>{p.items.map((it) => <li key={it}>{it}</li>)}</ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
