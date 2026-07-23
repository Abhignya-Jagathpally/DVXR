// DVXR BCI skill system — decoded EEG commands become avatar "skills".
//
// Mirrors the web scene (web/signal/src/components/rtdemo/skills.js): each of the five
// EMOTIV commands maps to a skill that fires only when the decoded command matches with
// enough confidence and its cooldown has elapsed. Aligned to the DVXR Lab's BCI +
// digital-twin immersive work.
//
// HONEST: the command channel is a single-subject EMOTIV mental-command engine label
// (~0.82 4-class, trial-grouped) — a demonstration control signal, not validated intent.

using System.Collections.Generic;
using UnityEngine;

namespace DVXR.RTDemo
{
    public struct Skill
    {
        public string id, label, glyph;
        public Color color;
        public float cooldown;
        public Skill(string id, string label, string glyph, Color color, float cooldown)
        { this.id = id; this.label = label; this.glyph = glyph; this.color = color; this.cooldown = cooldown; }
    }

    public class SkillSystem : MonoBehaviour
    {
        public RTStreamClient client;
        [Range(0f, 1f)] public float confidenceGate = 0.6f;

        public static readonly Dictionary<string, Skill> Skills = new Dictionary<string, Skill>
        {
            { "Neutral", new Skill("focus", "Focus", "○", new Color(0.22f, 0.74f, 0.97f), 0f) },
            { "Left",    new Skill("ward-left", "Ward Left", "◀", new Color(0.65f, 0.55f, 0.98f), 1.2f) },
            { "Right",   new Skill("ward-right", "Ward Right", "▶", new Color(0.96f, 0.45f, 0.71f), 1.2f) },
            { "Push",    new Skill("surge", "Surge", "▲", new Color(0.96f, 0.62f, 0.04f), 2.0f) },
            { "Pull",    new Skill("recover", "Recover", "◆", new Color(0.20f, 0.83f, 0.60f), 2.0f) },
        };

        public Skill ActiveSkill { get; private set; }
        public bool JustFired { get; private set; }
        public int Combo { get; private set; }

        readonly Dictionary<string, float> _cooldownUntil = new Dictionary<string, float>();

        void Update()
        {
            if (client == null) return;
            var f = client.Latest;
            if (f == null) return;
            var skill = Skills.TryGetValue(f.command, out var s) ? s : Skills["Neutral"];
            ActiveSkill = skill;

            float now = Time.time;
            bool ready = !_cooldownUntil.TryGetValue(skill.id, out var until) || until <= now;
            JustFired = f.command != "Neutral" && f.command_confidence >= confidenceGate && ready;
            if (JustFired)
            {
                _cooldownUntil[skill.id] = now + skill.cooldown;
                Combo++;
            }
        }

        public bool OnCooldown(string id)
        {
            return _cooldownUntil.TryGetValue(id, out var until) && until > Time.time;
        }
    }
}
