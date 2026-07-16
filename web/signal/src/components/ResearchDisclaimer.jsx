import SignalCanvas from "./SignalCanvas.jsx";
import { campaignDraw } from "../lib/signals.js";
import { brand, campaign, disclaimer } from "../content.js";

export default function ResearchDisclaimer() {
  return (
    <>
      <section id="campaign">
        <SignalCanvas draw={campaignDraw} />
        <div className="veil" />
        <div className="inner wrap">
          <div className="display l1 reveal">{campaign.l1}</div>
          <div className="display l2 reveal">{campaign.l2[0]}<br />{campaign.l2[1]}</div>
          <a className="btn solid reveal" href={campaign.cta.href}>{campaign.cta.label} <span className="arw">→</span></a>
        </div>
      </section>
      <footer>
        <div className="frow">
          <div>
            <div className="brand"><b>{brand.name}</b></div>
            <p className="disc" style={{ marginTop: "14px" }}><b>A multimodal health-intelligence research prototype.</b>{disclaimer.slice("A multimodal health-intelligence research prototype.".length)}</p>
          </div>
          <div className="meta">{brand.tagline}<br />EEG · Wearable · CGM · EHR · Omics<br />Offline · CPU · deterministic · reproducible</div>
        </div>
      </footer>
    </>
  );
}
