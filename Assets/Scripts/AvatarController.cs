// EXPERIMENTAL RT-Demo avatar controller for Unity.
//
// Reads the latest rt-demo-v1 frame from an RTStreamClient and drives the avatar with the
// SAME semantics as the web react-three-fiber scene: translate per command, tint by stress,
// and reflect the glucose abstention state on an optional UI label. Never fabricates a
// glucose value.

using UnityEngine;

namespace DVXR.RTDemo
{
    public class AvatarController : MonoBehaviour
    {
        [Tooltip("The stream client providing decoded frames.")]
        public RTStreamClient client;

        [Tooltip("Optional UnityEngine.UI.Text for the glucose/abstention readout.")]
        public UnityEngine.UI.Text statusLabel;

        [Header("Motion")]
        public float translateLerp = 0.12f;
        public float commandOffset = 1.4f;
        public float pushPullOffset = 1.2f;

        private static readonly Color Calm = new Color(0.15f, 0.39f, 0.92f);
        private static readonly Color Stressed = new Color(0.86f, 0.15f, 0.15f);

        private Vector3 _basePos;
        private Renderer _renderer;

        private void Start()
        {
            _basePos = transform.localPosition;
            _renderer = GetComponent<Renderer>();
        }

        private void Update()
        {
            if (client == null) return;
            var frame = client.Latest;
            if (frame == null) return;

            // command -> target pose (identical mapping to the web scene)
            Vector3 offset = Vector3.zero;
            switch (frame.command)
            {
                case "Left": offset = new Vector3(-commandOffset, 0, 0); break;
                case "Right": offset = new Vector3(commandOffset, 0, 0); break;
                case "Push": offset = new Vector3(0, 0, -pushPullOffset); break;
                case "Pull": offset = new Vector3(0, 0, pushPullOffset); break;
                default: offset = Vector3.zero; break; // Neutral
            }
            transform.localPosition = Vector3.Lerp(
                transform.localPosition, _basePos + offset, translateLerp);
            transform.Rotate(Vector3.up, (0.6f + 1.8f * Mathf.Clamp01(frame.stress)) * Time.deltaTime * 60f);

            // stress -> tint
            if (_renderer != null)
            {
                _renderer.material.color = Color.Lerp(Calm, Stressed, Mathf.Clamp01(frame.stress));
            }

            // glucose abstention -> honest readout (never a fabricated value)
            if (statusLabel != null)
            {
                statusLabel.text = frame.abstained
                    ? "Glucose: insufficient data (abstained)"
                    : $"Glucose: {frame.glucose_point:0} mg/dL (EXPERIMENTAL)";
            }
        }
    }
}
