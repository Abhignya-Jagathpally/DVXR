import { useState } from "react";
import SignalCanvas from "./SignalCanvas.jsx";
import { cgmDraw, miniDraw } from "../lib/signals.js";

const TABS = [
  { id: "cgm", label: "01 · Glucose trajectory" },
  { id: "stress", label: "02 · Stress intelligence" },
  { id: "abstain", label: "03 · Multimodal request" },
];

const MINI = [
  [{ k: "eeg", label: "EEG spectrum", st: "unavailable", off: true }, { k: "hrv", label: "Heart-rate variability", st: "live" }],
  [{ k: "eda", label: "Electrodermal activity", st: "live" }, { k: "resp", label: "Respiration", st: "live" }],
  [{ k: "mot", label: "Motion", st: "live" }, { k: "conf", label: "Confidence", st: "moderate" }],
];

export default function InteractiveExperience() {
  const [active, setActive] = useState("cgm");
  return (
    <section id="experience" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal"><span className="eyebrow">The interactive experience</span></div>
        <h2 className="display chapter-h reveal">From raw signal<br />to research insight.</h2>
        <div className="tabs reveal" role="tablist" aria-label="Interactive demonstrations">
          {TABS.map((t) => (
            <button key={t.id} className="tab" role="tab" aria-selected={active === t.id} onClick={() => setActive(t.id)}>{t.label}</button>
          ))}
        </div>

        {active === "cgm" && (
          <div className="demo active" role="tabpanel">
            <div className="demobox">
              <div className="demobar"><span>CGM trajectory · 30-minute horizon</span><span className="demoflag">Demonstration · synthetic sample</span></div>
              <div className="demobody">
                <div className="left"><SignalCanvas className="cgm-canvas" draw={cgmDraw()} ariaLabel="Animated glucose forecast with expanding uncertainty interval" /></div>
                <div className="right">
                  <div className="readout"><div className="lab">Projected glucose · +30 min</div><div className="big tnum">126<span className="u"> mg/dL</span></div><div className="sub">Interval 114–139 mg/dL</div></div>
                  <div><div className="lab" style={{ fontFamily: "var(--mono)", fontSize: "10.5px", letterSpacing: ".14em", textTransform: "uppercase", color: "var(--faint)" }}>Trajectory state</div><div style={{ marginTop: "8px" }}><span className="state-chip stable">Stable · upward momentum</span></div></div>
                  <div className="interp"><b>Interpretation.</b> The current sequence indicates an upward glucose trend. Forecast uncertainty remains moderate because meal and activity context are unavailable. The language model narrates this evidence; it does not compute the value.</div>
                  <div className="mono" style={{ fontSize: "10px", color: "var(--faint)", letterSpacing: ".03em", lineHeight: 1.7 }}>model cgm-conformal · horizon 30m · cutoff t₀ · interval split-conformal 90%<br />real-cohort reference: CGMacros 30-min MAE 10.27 mg/dL · coverage 0.877</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {active === "stress" && (
          <div className="demo active" role="tabpanel">
            <div className="demobox">
              <div className="demobar"><span>Stress intelligence · missing-modality handling</span><span className="demoflag">Demonstration · sample signals</span></div>
              <div className="demobody">
                <div className="left">
                  {MINI.map((row, i) => (
                    <div className="miniwave" key={i}>
                      {row.map((m) => (
                        <div className={"m" + (m.off ? " off" : "")} key={m.k}>
                          <div className="lab"><span>{m.label}</span><span>{m.st}</span></div>
                          <SignalCanvas draw={miniDraw(m.k, !!m.off)} />
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
                <div className="right">
                  <div className="readout"><div className="lab">Physiological stress state</div><div style={{ marginTop: "10px" }}><span className="state-chip elevated">Elevated</span></div></div>
                  <div className="interp"><b>Evidence.</b> The estimate is primarily supported by reduced HRV and increased electrodermal activity. <b>EEG information is unavailable</b> for this observation window, so no neural contribution is claimed — the analysis path adapts to the signals actually present.</div>
                  <div className="mono" style={{ fontSize: "10px", color: "var(--faint)", letterSpacing: ".03em", lineHeight: 1.7 }}>path wearable-only · EEG=absent · attribution: HRV↓, EDA↑<br />real-cohort reference: wearable stress AUROC 0.955 (0.930–0.978)</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {active === "abstain" && (
          <div className="demo active" role="tabpanel">
            <div className="demobox">
              <div className="demobar"><span>Fused neural–physiological–metabolic request</span><span className="demoflag">Demonstration · honest abstention</span></div>
              <div className="demobody">
                <div className="left" style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: "18px" }}>
                  <div><span className="state-chip abstain">Insufficient synchronized evidence</span></div>
                  <p style={{ fontSize: "15px", color: "var(--gray)", maxWidth: "46ch", margin: 0 }}>A fused neural–physiological–metabolic estimate was <b style={{ color: "var(--white)" }}>not generated</b> because the required same-subject EEG, wearable, and CGM observations were not available within the selected time window.</p>
                  <p className="mono" style={{ fontSize: "11px", color: "var(--faint)", letterSpacing: ".03em", margin: 0, lineHeight: 1.7 }}>status: abstained · risk: null (no fabricated probability)<br />reason_codes: [ no_synchronized_cohort, fusion_claim_not_permitted ]</p>
                </div>
                <div className="right">
                  <div className="lab" style={{ fontFamily: "var(--mono)", fontSize: "10.5px", letterSpacing: ".14em", textTransform: "uppercase", color: "var(--faint)" }}>Modality availability</div>
                  <div className="avail">
                    <div className="row on"><span>CGM</span><span className="st">available</span></div>
                    <div className="row on"><span>Heart rate</span><span className="st">available</span></div>
                    <div className="row off"><span>EEG</span><span className="st">missing</span></div>
                    <div className="row off"><span>Clinical context</span><span className="st">missing</span></div>
                  </div>
                  <div className="interp" style={{ marginTop: "2px" }}><b>Result — single-modality analysis only.</b> Responsible abstention is a feature, not a failure: the system will not invent a fused result it cannot support.</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
