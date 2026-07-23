// BCI "skill" system for the digital-twin avatar (DVXR-lab-aligned demo).
//
// The lab's signature is BCI + Digital Twin + immersive training (e.g. VR Fire Evacuation
// with a BCI-driven avatar). Here the five decoded EMOTIV commands become avatar SKILLS a
// "player" triggers with intent. Each skill fires only when the decoded command matches
// with enough confidence and its cooldown has elapsed.
//
// HONEST CAVEAT (kept visible in the UI): the command channel is a single-subject EMOTIV
// mental-command engine label (~0.82 4-class, trial-grouped) — a demonstration control
// signal, NOT validated neural-intent decoding.

export const SKILLS = {
  Neutral: {
    id: "focus",
    label: "Focus",
    icon: "🧘",
    color: "#38bdf8",
    cooldownMs: 0,
    description: "Baseline calm — steadies the twin and slowly lowers stress.",
  },
  Left: {
    id: "ward-left",
    label: "Ward Left",
    icon: "🛡️",
    color: "#a78bfa",
    cooldownMs: 1200,
    description: "Raise a left-side guard.",
  },
  Right: {
    id: "ward-right",
    label: "Ward Right",
    icon: "🛡️",
    color: "#f472b6",
    cooldownMs: 1200,
    description: "Raise a right-side guard.",
  },
  Push: {
    id: "surge",
    label: "Surge",
    icon: "⚡",
    color: "#f59e0b",
    cooldownMs: 2000,
    description: "Forward burst — high-effort action.",
  },
  Pull: {
    id: "recover",
    label: "Recover",
    icon: "💠",
    color: "#34d399",
    cooldownMs: 2000,
    description: "Pull back and recover.",
  },
};

export const CONFIDENCE_GATE = 0.6; // a skill fires only above this decoder confidence

/**
 * Reduce a stream of frames into skill activation state.
 * Pure function of (previous state, frame, nowMs) so it stays testable.
 */
export function nextSkillState(prev, frame, nowMs) {
  const skill = SKILLS[frame.command] || SKILLS.Neutral;
  const cooldownUntil = prev.cooldownUntil || {};
  const ready = (cooldownUntil[skill.id] || 0) <= nowMs;
  const fired =
    frame.command !== "Neutral" &&
    (frame.command_confidence ?? 0) >= CONFIDENCE_GATE &&
    ready;

  const nextCooldown = { ...cooldownUntil };
  if (fired) nextCooldown[skill.id] = nowMs + skill.cooldownMs;

  return {
    activeSkill: skill,
    fired,
    firedAt: fired ? nowMs : prev.firedAt || 0,
    cooldownUntil: nextCooldown,
    combo: fired ? (prev.combo || 0) + 1 : prev.combo || 0,
  };
}

export const INITIAL_SKILL_STATE = {
  activeSkill: SKILLS.Neutral,
  fired: false,
  firedAt: 0,
  cooldownUntil: {},
  combo: 0,
};
