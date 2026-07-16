import { features } from "../content.js";

export default function FeatureGrid() {
  return (
    <section id="features" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal"><span className="eyebrow">Designed for credibility</span></div>
        <h2 className="display chapter-h reveal">The discipline behind<br />the intelligence.</h2>
        <div className="fgrid">
          {features.map((f) => (
            <div className={"tile " + f.size + " reveal"} key={f.n}>
              <span className="num">{f.n}</span>
              <div className="th">{f.title}</div>
              <p>{f.body}</p>
              {f.tags ? <div className="provlist">{f.tags.map((t) => <span key={t}>{t}</span>)}</div> : null}
              {f.footnote ? <p className="mono" style={{ fontSize: "10px", color: "var(--faint)", marginTop: "8px" }}>{f.footnote}</p> : null}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
