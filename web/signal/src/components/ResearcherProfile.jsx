import { researcher } from "../content.js";

export default function ResearcherProfile() {
  return (
    <section id="researcher" className="pad-y">
      <div className="wrap res">
        <div className="portrait reveal"><div className="mono-init">{researcher.initials}</div><div className="cap">{researcher.name}</div></div>
        <div className="reveal">
          <h2 className="display chapter-h stmt">{researcher.statement}</h2>
          <p className="bio">{researcher.bio}</p>
          <div className="pillars">
            {researcher.pillars.map((p) => (
              <div className="p" key={p.title}><h5>{p.title}</h5><p>{p.body}</p></div>
            ))}
          </div>
          <div className="actions">
            <a className="btn ghost" href="#engine">View methodology <span className="arw">→</span></a>
            <a className="btn ghost" href="#evidence">Read the research brief <span className="arw">→</span></a>
          </div>
        </div>
      </div>
    </section>
  );
}
