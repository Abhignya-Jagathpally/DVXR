import SignalCanvas from "./SignalCanvas.jsx";
import { heroDraw } from "../lib/signals.js";
import { hero } from "../content.js";

export default function HeroSection() {
  return (
    <section id="hero">
      <SignalCanvas draw={heroDraw} />
      <div className="hero-fade" />
      <div className="inner wrap">
        <span className="tag"><span className="pill"><span className="d" />{hero.tag}</span></span>
        <h1 className="display hero-h">
          {hero.headline.map((l, i) => (
            <span key={i}>{l}{i < hero.headline.length - 1 ? <br /> : null}</span>
          ))}
        </h1>
        <p className="subcopy">{hero.subcopy}</p>
        <div className="actions">
          <a className="btn solid" href={hero.primary.href}>{hero.primary.label} <span className="arw">→</span></a>
          <a className="btn ghost" href={hero.secondary.href}>{hero.secondary.label} <span className="arw">→</span></a>
        </div>
        <p className="qual">{hero.qualification}</p>
      </div>
      <div className="scrollcue"><span className="bar" /> Scroll</div>
    </section>
  );
}
