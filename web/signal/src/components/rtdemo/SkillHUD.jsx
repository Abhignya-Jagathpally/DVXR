// Skill HUD for the BCI digital-twin demo: shows the five skills, the active one,
// cooldown state, the decoded command confidence, and the digital-twin physiology
// (stress + glucose-abstention). Honest EXPERIMENTAL framing throughout.

import { SKILLS, CONFIDENCE_GATE } from "./skills.js";

export default function SkillHUD({ frame, skillState }) {
  const stressPct = Math.round((frame.stress ?? 0) * 100);
  const confPct = Math.round((frame.command_confidence ?? 0) * 100);
  const abstained = frame.abstained !== false;

  return (
    <div
      style={{
        position: "absolute", left: 12, top: 48, zIndex: 3,
        display: "flex", flexDirection: "column", gap: 10, maxWidth: 300,
        color: "#e2e8f0", fontSize: 12,
      }}
    >
      {/* skill bar */}
      <div style={{ display: "flex", gap: 8 }}>
        {Object.entries(SKILLS).map(([command, skill]) => {
          const isActive = skillState.activeSkill?.id === skill.id;
          const onCooldown = (skillState.cooldownUntil?.[skill.id] || 0) > Date.now();
          return (
            <div
              key={skill.id}
              title={`${skill.label}: ${skill.description}`}
              style={{
                width: 46, height: 46, borderRadius: 10,
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center",
                background: isActive ? skill.color : "rgba(15,23,42,0.7)",
                border: `1px solid ${isActive ? "#fff" : "rgba(148,163,184,0.35)"}`,
                opacity: onCooldown ? 0.4 : 1,
                transition: "all 120ms ease",
                boxShadow: isActive && skillState.fired ? `0 0 16px ${skill.color}` : "none",
              }}
            >
              <span style={{ fontSize: 18 }}>{skill.icon}</span>
              <span style={{ fontSize: 8, opacity: 0.85 }}>{command}</span>
            </div>
          );
        })}
      </div>

      {/* decoder + twin physiology */}
      <div
        style={{
          background: "rgba(15,23,42,0.72)",
          border: "1px solid rgba(148,163,184,0.3)",
          borderRadius: 10, padding: "8px 12px",
          display: "flex", flexDirection: "column", gap: 6,
        }}
      >
        <Row label="Active skill" value={`${skillState.activeSkill?.icon || ""} ${skillState.activeSkill?.label || "—"}`} />
        <Row label="Decoder confidence" value={`${confPct}%`} warn={confPct < CONFIDENCE_GATE * 100} />
        <Row label="Combo" value={skillState.combo} />
        <Bar label="Twin stress" pct={stressPct} color={stressPct > 60 ? "#dc2626" : "#12b76a"} />
        <Row label="Twin glucose" value={abstained ? "abstained (insufficient data)" : `${frame.glucose_point} mg/dL`} warn={abstained} />
      </div>

      <div style={{ fontSize: 10, opacity: 0.7, lineHeight: 1.4 }}>
        EXPERIMENTAL · command channel is a single-subject EMOTIV engine label
        (~0.82 4-class), a demonstration control signal — not validated neural intent.
      </div>
    </div>
  );
}

function Row({ label, value, warn }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
      <span style={{ opacity: 0.7 }}>{label}</span>
      <strong style={{ color: warn ? "#fbbf24" : "#e2e8f0" }}>{value}</strong>
    </div>
  );
}

function Bar({ label, pct, color }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", opacity: 0.7 }}>
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 4, background: "rgba(148,163,184,0.25)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width 200ms ease" }} />
      </div>
    </div>
  );
}
