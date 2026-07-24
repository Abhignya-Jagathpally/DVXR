// DVXR VR-Realtime — world-space personalized-stats HUD for VR.
//
// THE VR-native readout. The in-repo `PhysiologyHUD.cs` uses IMGUI/OnGUI, which renders to
// the flat game view but NOT to a headset. This builds a real WORLD-SPACE uGUI canvas
// (visible in VR) entirely from code — attach one component, press Play, no prefab/Canvas
// wiring — and updates it every frame from the rt-demo-v1 stream so the person wearing the
// headset sees THEIR live stats floating in the scene beside their avatar.
//
// Honesty invariants (mirrored from the backend contract, non-negotiable):
//   * glucose ABSTAINS -> render an explicit "insufficient data" state, never a fake number;
//   * the EXPERIMENTAL badge is always visible while animating;
//   * single-subject / engine-label decoding — a demonstration control, not clinical intent.

using UnityEngine;
using UnityEngine.UI;

namespace DVXR.VRRealtime
{
    /// <summary>How the stats panel is placed relative to the wearer / their avatar.</summary>
    public enum HudAnchorMode
    {
        /// <summary>Floats at a fixed comfortable offset in front of the headset, always facing the wearer.</summary>
        HeadLockedBillboard,
        /// <summary>Parents to <see cref="anchor"/> (e.g. the avatar's left wrist) — a smart-watch readout.</summary>
        FollowAnchor,
        /// <summary>Stays where it is placed in the world; only billboards to face the camera.</summary>
        WorldFixed,
    }

    public class VRStatsHUD : MonoBehaviour
    {
        [Header("Data source")]
        [Tooltip("The realtime client. If left empty one is added to this GameObject automatically.")]
        public RtStreamClient client;

        [Header("Who is this? (personalization label only — no fabricated numbers)")]
        [Tooltip("Shown in the panel header, e.g. the participant handle. Purely a label.")]
        public string participantLabel = "You";

        [Header("Placement")]
        public HudAnchorMode anchorMode = HudAnchorMode.HeadLockedBillboard;

        [Tooltip("Camera the panel faces / anchors to. Defaults to the XR/main camera.")]
        public Transform cameraOverride;

        [Tooltip("FollowAnchor: the avatar transform to attach to (wrist, chest, HMD…).")]
        public Transform anchor;

        [Tooltip("Local offset (metres) from the camera (HeadLocked) or from the anchor (Follow).")]
        public Vector3 offset = new Vector3(0.28f, -0.12f, 0.85f);

        [Tooltip("Physical width of the panel in metres.")]
        public float panelWidthMeters = 0.34f;

        // --- runtime UI handles (all built in code) ---
        private Transform _cam;
        private RectTransform _panel;
        private Text _title, _source, _cmd, _stressVal, _glucose, _skill, _footer;
        private Image _stressFill;
        private Image _connDot;
        private Font _font;

        private const int PxW = 420;   // canvas pixel width  (scaled to panelWidthMeters)
        private const int PxH = 300;   // canvas pixel height

        private void Awake()
        {
            if (client == null) client = GetComponent<RtStreamClient>() ?? gameObject.AddComponent<RtStreamClient>();
            _cam = cameraOverride != null ? cameraOverride : (Camera.main != null ? Camera.main.transform : null);
            _font = LoadFont();
            BuildCanvas();
        }

        private static Font LoadFont()
        {
            // Unity 2022+ ships LegacyRuntime.ttf; older editors ship Arial.ttf.
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }

        // ---------- UI construction (pure code, no scene wiring) ----------

        private void BuildCanvas()
        {
            var canvasGO = new GameObject("DVXR_VRStatsHUD_Canvas");
            canvasGO.transform.SetParent(transform, false);
            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();

            _panel = canvasGO.GetComponent<RectTransform>();
            _panel.sizeDelta = new Vector2(PxW, PxH);
            // Scale canvas pixels -> metres so the panel is `panelWidthMeters` wide in the world.
            var s = panelWidthMeters / PxW;
            _panel.localScale = new Vector3(s, s, s);

            AddPanelImage(canvasGO.transform, new Color(0.04f, 0.06f, 0.10f, 0.90f), PxW, PxH, Vector2.zero);
            // subtle accent rail on the left edge
            AddPanelImage(canvasGO.transform, new Color(0.05f, 0.65f, 0.55f, 1f), 6, PxH, new Vector2(-PxW / 2f + 3, 0));

            float y = PxH / 2f - 22;
            _title = AddText(canvasGO.transform, $"{participantLabel} · live readout", 18, FontStyle.Bold,
                             new Color(0.92f, 0.95f, 0.99f), new Vector2(24, y), PxW - 90, TextAnchor.MiddleLeft);
            _connDot = AddPanelImage(canvasGO.transform, new Color(0.5f, 0.5f, 0.5f, 1f), 12, 12, new Vector2(PxW / 2f - 54, y));
            _source = AddText(canvasGO.transform, "demo", 12, FontStyle.Bold, new Color(0.62f, 0.69f, 0.80f),
                              new Vector2(PxW / 2f - 44, y), 40, TextAnchor.MiddleLeft);

            y -= 26;
            AddText(canvasGO.transform, "fused from EEG · wearable · glucose", 12, FontStyle.Normal,
                    new Color(0.55f, 0.61f, 0.72f), new Vector2(24, y), PxW - 48, TextAnchor.MiddleLeft);

            y -= 34;
            _cmd = AddRow(canvasGO.transform, "EEG command", "Neutral (0%)", ref y);
            // stress bar: track and fill share the same centre-relative rect so they align.
            AddText(canvasGO.transform, "Stress", 13, FontStyle.Normal, new Color(0.62f, 0.69f, 0.80f),
                    new Vector2(24, y), 96, TextAnchor.MiddleLeft);
            var barCentre = new Vector2(25, y);          // 130px from left edge, 210px wide
            AddPanelImage(canvasGO.transform, new Color(1, 1, 1, 0.08f), 210, 14, barCentre);
            _stressFill = AddBar(canvasGO.transform, barCentre, 210, 14);
            _stressVal = AddText(canvasGO.transform, "0%", 13, FontStyle.Bold, new Color(0.92f, 0.95f, 0.99f),
                                 new Vector2(PxW - 66, y), 42, TextAnchor.MiddleRight);
            y -= 30;
            _glucose = AddRow(canvasGO.transform, "Glucose", "abstained — insufficient data", ref y);
            _skill = AddRow(canvasGO.transform, "Active skill", "—", ref y);

            _footer = AddText(canvasGO.transform,
                "EXPERIMENTAL · single-subject EMOTIV engine label — demonstration control, not validated neural intent. Not for clinical use.",
                10, FontStyle.Normal, new Color(0.55f, 0.61f, 0.72f),
                new Vector2(24, -PxH / 2f + 30), PxW - 48, TextAnchor.UpperLeft);
        }

        private Image AddPanelImage(Transform parent, Color c, float w, float h, Vector2 anchoredPos)
        {
            var go = new GameObject("img");
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.color = c;
            var rt = img.rectTransform;
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = anchoredPos;
            return img;
        }

        private Image AddBar(Transform parent, Vector2 anchoredPos, float w, float h)
        {
            var img = AddPanelImage(parent, new Color(0.86f, 0.2f, 0.2f, 1f), w, h, anchoredPos);
            img.type = Image.Type.Filled;
            img.fillMethod = Image.FillMethod.Horizontal;
            img.fillOrigin = (int)Image.OriginHorizontal.Left;
            img.fillAmount = 0f;
            return img;
        }

        private Text AddText(Transform parent, string s, int size, FontStyle style, Color c,
                             Vector2 anchoredPos, float width, TextAnchor align)
        {
            var go = new GameObject("txt");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = _font;
            t.text = s;
            t.fontSize = size;
            t.fontStyle = style;
            t.color = c;
            t.alignment = align;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            var rt = t.rectTransform;
            rt.sizeDelta = new Vector2(width, size + 8);
            // anchoredPosition is measured from panel centre; convert left-origin x to centre-origin.
            rt.anchoredPosition = new Vector2(anchoredPos.x - PxW / 2f + width / 2f, anchoredPos.y);
            return t;
        }

        private Text AddRow(Transform parent, string label, string value, ref float y)
        {
            AddText(parent, label, 13, FontStyle.Normal, new Color(0.62f, 0.69f, 0.80f),
                    new Vector2(24, y), 110, TextAnchor.MiddleLeft);
            var v = AddText(parent, value, 13, FontStyle.Bold, new Color(0.92f, 0.95f, 0.99f),
                            new Vector2(140, y), PxW - 164, TextAnchor.MiddleLeft);
            y -= 30;
            return v;
        }

        // ---------- per-frame update ----------

        private void LateUpdate()
        {
            PlacePanel();
            Render(client != null ? client.Latest : null);
        }

        private void PlacePanel()
        {
            if (_cam == null && Camera.main != null) _cam = Camera.main.transform;

            switch (anchorMode)
            {
                case HudAnchorMode.HeadLockedBillboard:
                    if (_cam != null)
                    {
                        transform.position = _cam.position + _cam.rotation * offset;
                        transform.rotation = Quaternion.LookRotation(transform.position - _cam.position, Vector3.up);
                    }
                    break;
                case HudAnchorMode.FollowAnchor:
                    if (anchor != null)
                    {
                        transform.position = anchor.TransformPoint(offset);
                        if (_cam != null)
                            transform.rotation = Quaternion.LookRotation(transform.position - _cam.position, Vector3.up);
                    }
                    break;
                case HudAnchorMode.WorldFixed:
                    if (_cam != null)
                        transform.rotation = Quaternion.LookRotation(transform.position - _cam.position, Vector3.up);
                    break;
            }
        }

        private void Render(RtFrame f)
        {
            bool live = client != null && client.Connected;
            _source.text = client != null ? client.SourceLabel : "demo";
            _connDot.color = live ? new Color(0.2f, 0.85f, 0.45f) : new Color(0.98f, 0.75f, 0.18f);

            if (f == null)
            {
                _cmd.text = live ? "waiting for frames…" : "not connected";
                return;
            }

            _cmd.text = $"{f.command}  ({Mathf.RoundToInt(f.command_confidence * 100f)}%)";

            float stress = Mathf.Clamp01(f.stress);
            _stressFill.fillAmount = stress;
            _stressFill.color = Color.Lerp(new Color(0.20f, 0.70f, 0.45f), new Color(0.90f, 0.22f, 0.22f), stress);
            _stressVal.text = $"{Mathf.RoundToInt(stress * 100f)}%";

            if (f.abstained || !f.IsGlucoseKnown)
            {
                _glucose.text = "abstained — insufficient data";
                _glucose.color = new Color(0.98f, 0.75f, 0.18f);   // honest amber, never a number
            }
            else
            {
                _glucose.text = $"{Mathf.RoundToInt(f.glucose_point)} mg/dL  [{Mathf.RoundToInt(f.glucose_lower)}–{Mathf.RoundToInt(f.glucose_upper)}]";
                _glucose.color = new Color(0.92f, 0.95f, 0.99f);
            }

            // Active skill: map the decoded command to a labelled skill glyph (display only).
            _skill.text = SkillFor(f.command);
        }

        private static string SkillFor(string command)
        {
            switch (command)
            {
                case "Left":  return "◀ Steer-Left";
                case "Right": return "▶ Steer-Right";
                case "Push":  return "⭱ Push";
                case "Pull":  return "⭳ Pull";
                default:       return "• Idle";
            }
        }
    }
}
