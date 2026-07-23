// EXPERIMENTAL RT-Demo section: a live avatar driven by the rt-demo-v1 stream.
//
// Substitutes for the (non-existent) Unity scene while remaining demonstrable in-browser.
// The avatar command is the bci_real analog; the glucose channel abstains by construction,
// rendered as an explicit "insufficient data" state — never a fabricated number.

import { useEffect, useRef, useState } from "react";
import AvatarScene from "./AvatarScene.jsx";
import SkillHUD from "./SkillHUD.jsx";
import { INITIAL_SKILL_STATE, nextSkillState } from "./skills.js";
import {
  EXPERIMENTAL_BADGE,
  RT_DEMO_MODE,
  buildFrame,
  subscribeFrames,
} from "../../lib/realtimeSocket.js";

export default function RTDemo() {
  const frameRef = useRef(buildFrame(0));
  const skillRef = useRef(INITIAL_SKILL_STATE);
  const [telemetry, setTelemetry] = useState(buildFrame(0));
  const [skillState, setSkillState] = useState(INITIAL_SKILL_STATE);

  useEffect(() => {
    let raf = 0;
    let clock = 0; // deterministic monotonic clock (avoids Date.now in the reducer path)
    const stop = subscribeFrames((frame) => {
      frameRef.current = frame;
      clock += 120;
      skillRef.current = nextSkillState(skillRef.current, frame, clock);
    }, { intervalMs: 120 });
    // throttle the DOM readout to animation frames (the 3D scene reads frameRef directly)
    const tick = () => {
      setTelemetry(frameRef.current);
      setSkillState(skillRef.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      stop();
      cancelAnimationFrame(raf);
    };
  }, []);

  const abstained = telemetry.abstained !== false;
  const stressPct = Math.round((telemetry.stress ?? 0) * 100);
  const confPct = Math.round((telemetry.command_confidence ?? 0) * 100);

  return (
    <section id="rt-demo" className="wrap pad-y">
      <div className="kicker reveal">
        <span className="eyebrow">BCI digital twin · real-time skills</span>
      </div>
      <h2 className="display chapter-h reveal">
        Your signals,<br />as a playable twin.
      </h2>
      <p className="reveal" style={{ maxWidth: 660, opacity: 0.8 }}>
        A digital twin driven by live decoded EMOTIV commands — each becomes a
        <strong> skill</strong> (Focus · Ward · Surge · Recover) that fires on intent and
        confidence. The twin reddens with predicted stress; its glucose ring abstains
        when no synchronized CGM stream exists — no value is invented. Built in the spirit
        of the UNT DVXR Lab's BCI + digital-twin immersive work.
      </p>

      <div
        className="reveal"
        style={{
          position: "relative",
          marginTop: 24,
          borderRadius: 16,
          overflow: "hidden",
          border: "1px solid rgba(148,163,184,0.25)",
          background: "#0b1020",
          height: 420,
        }}
      >
        <span
          style={{
            position: "absolute", top: 12, left: 12, zIndex: 3,
            fontSize: 11, letterSpacing: "0.08em", fontWeight: 700,
            color: "#fbbf24", background: "rgba(0,0,0,0.55)",
            padding: "4px 10px", borderRadius: 999,
          }}
        >
          {EXPERIMENTAL_BADGE}{RT_DEMO_MODE ? " · in-page" : " · live"}
        </span>

        <AvatarScene frameRef={frameRef} />

        <SkillHUD frame={telemetry} skillState={skillState} />

        {/* telemetry readout */}
        <div
          style={{
            position: "absolute", right: 12, bottom: 12, zIndex: 3,
            display: "flex", gap: 10, flexWrap: "wrap",
            fontSize: 12, color: "#e2e8f0",
          }}
        >
          <Chip label="command" value={telemetry.command} />
          <Chip label="confidence" value={`${confPct}%`} />
          <Chip label="stress" value={`${stressPct}%`} />
          <Chip
            label="glucose"
            value={abstained ? "abstained" : `${telemetry.glucose_point} mg/dL`}
            warn={abstained}
          />
        </div>

        {/* explicit abstention overlay for the glucose channel */}
        {abstained && (
          <div
            style={{
              position: "absolute", left: 12, bottom: 12, zIndex: 3,
              maxWidth: 260, fontSize: 12, lineHeight: 1.4,
              color: "#cbd5e1", background: "rgba(15,23,42,0.72)",
              border: "1px solid rgba(148,163,184,0.3)",
              padding: "8px 12px", borderRadius: 10,
            }}
          >
            <strong style={{ color: "#94a3b8" }}>Glucose: insufficient data.</strong>{" "}
            The model abstains — no synchronized per-subject CGM stream. This is
            decision-support demonstration only, not clinical inference.
          </div>
        )}
      </div>
    </section>
  );
}

function Chip({ label, value, warn }) {
  return (
    <span
      style={{
        background: warn ? "rgba(148,163,184,0.18)" : "rgba(37,99,235,0.22)",
        border: `1px solid ${warn ? "rgba(148,163,184,0.4)" : "rgba(37,99,235,0.5)"}`,
        borderRadius: 8, padding: "4px 8px",
      }}
    >
      <span style={{ opacity: 0.65 }}>{label}:</span>{" "}
      <strong>{value}</strong>
    </span>
  );
}
