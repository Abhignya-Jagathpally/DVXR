import { stories } from "../content.js";

export default function UseCases() {
  return (
    <section id="stories">
      <div className="wrap pad-y" style={{ paddingBottom: "40px" }}>
        <div className="kicker reveal"><span className="eyebrow">Where it matters</span></div>
        <h2 className="display chapter-h reveal">Not applications.<br />Research questions.</h2>
      </div>
      {stories.map((s) => (
        <div className="story" key={s.idx}><div className="wrap pad-y">
          <div className="reveal" style={{ display: "grid", gridTemplateColumns: ".4fr 1.6fr", gap: "clamp(20px,4vw,60px)" }}>
            <div className="sidx">{s.idx}</div>
            <div>
              <h3 className="display chapter-h">{s.title}</h3>
              <p className="body">{s.body}</p>
              <div className="aud">{s.audience.map((a) => <span key={a}>{a}</span>)}</div>
            </div>
          </div>
        </div></div>
      ))}
    </section>
  );
}
