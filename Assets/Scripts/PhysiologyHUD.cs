// DVXR digital-twin HUD — an in-Editor readout of the fused multimodal prediction.
//
// Uses IMGUI (OnGUI) so it renders with ZERO manual Canvas/UI wiring: attach this to any
// GameObject in the scene and press Play. Shows the framework's read-out from
// EEG + PHR + PPG — decoded command, stress, glucose forecast (or honest abstention), and
// the active BCI skill — with the EXPERIMENTAL caveat always visible.

using UnityEngine;

namespace DVXR.RTDemo
{
    public class PhysiologyHUD : MonoBehaviour
    {
        public RTStreamClient client;
        public SkillSystem skills;

        GUIStyle _h, _lbl, _val, _warn, _foot;

        void InitStyles()
        {
            _h = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.90f, 0.94f, 0.99f) } };
            _lbl = new GUIStyle(GUI.skin.label) { fontSize = 12, normal = { textColor = new Color(0.62f, 0.69f, 0.80f) } };
            _val = new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.90f, 0.94f, 0.99f) } };
            _warn = new GUIStyle(GUI.skin.label) { fontSize = 12, fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.98f, 0.75f, 0.18f) } };
            _foot = new GUIStyle(GUI.skin.label) { fontSize = 10, wordWrap = true,
                normal = { textColor = new Color(0.55f, 0.61f, 0.72f) } };
        }

        static Texture2D _bg;
        static Texture2D Bg()
        {
            if (_bg == null)
            {
                _bg = new Texture2D(1, 1);
                _bg.SetPixel(0, 0, new Color(0.04f, 0.06f, 0.10f, 0.82f));
                _bg.Apply();
            }
            return _bg;
        }

        void OnGUI()
        {
            if (_h == null) InitStyles();
            var f = client != null ? client.Latest : null;

            const float w = 300f, x = 14f, y = 14f;
            GUI.DrawTexture(new Rect(x, y, w, 232), Bg());
            GUILayout.BeginArea(new Rect(x + 14, y + 12, w - 28, 232 - 20));
            GUILayout.Label("DVXR digital twin", _h);
            GUILayout.Label("fused from EEG · PHR · PPG", _lbl);
            GUILayout.Space(8);

            if (f == null)
            {
                GUILayout.Label(client != null && client.Connected ? "waiting for frames…" : "not connected", _warn);
            }
            else
            {
                Row("EEG command", $"{f.command}  ({Mathf.RoundToInt(f.command_confidence * 100)}%)");
                Bar("PHR stress", Mathf.Clamp01(f.stress), new Color(0.86f, 0.2f, 0.2f));
                if (f.abstained)
                    Row("Glucose", "abstained — insufficient data", true);
                else
                    Row("Glucose", $"{Mathf.RoundToInt(f.glucose_point)} mg/dL  [{Mathf.RoundToInt(f.glucose_lower)}–{Mathf.RoundToInt(f.glucose_upper)}]");
                if (skills != null)
                    Row("Skill", $"{skills.ActiveSkill.glyph} {skills.ActiveSkill.label}   x{skills.Combo}");
            }

            GUILayout.Space(8);
            GUILayout.Label("EXPERIMENTAL · single-subject EMOTIV engine label — demonstration control, not validated neural intent. Not for clinical use.", _foot);
            GUILayout.EndArea();
        }

        void Row(string label, string value, bool warn = false)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _lbl, GUILayout.Width(96));
            GUILayout.Label(value, warn ? _warn : _val);
            GUILayout.EndHorizontal();
        }

        void Bar(string label, float pct, Color fill)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _lbl, GUILayout.Width(96));
            var r = GUILayoutUtility.GetRect(150, 12);
            GUI.DrawTexture(r, Bg());
            GUI.DrawTexture(new Rect(r.x, r.y, r.width * Mathf.Clamp01(pct), r.height), Tex(fill));
            GUILayout.Label($"{Mathf.RoundToInt(pct * 100)}%", _val, GUILayout.Width(38));
            GUILayout.EndHorizontal();
        }

        static Texture2D _fillTex; static Color _fillColor;
        static Texture2D Tex(Color c)
        {
            if (_fillTex == null || _fillColor != c)
            {
                _fillTex = new Texture2D(1, 1); _fillTex.SetPixel(0, 0, c); _fillTex.Apply(); _fillColor = c;
            }
            return _fillTex;
        }
    }
}
