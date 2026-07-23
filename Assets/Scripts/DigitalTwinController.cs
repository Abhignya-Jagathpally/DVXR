// DVXR digital-twin controller — the avatar as a live physiological twin.
//
// Reads the rt-demo-v1 frame from RTStreamClient (fed by the framework's fused
// EEG + PHR + PPG prediction) and maps it to the twin's appearance:
//   - decoded EEG command  -> pose / skill trigger (via SkillSystem)
//   - PHR stress index      -> emissive tint (calm blue -> stressed red) + breathing pulse
//   - glucose forecast      -> a halo ring; GREY + still when the model abstains
//
// EXPERIMENTAL demonstration. The command channel is a single-subject EMOTIV engine label,
// not validated neural intent; the glucose channel abstains rather than inventing a value.

using UnityEngine;

namespace DVXR.RTDemo
{
    [RequireComponent(typeof(Renderer))]
    public class DigitalTwinController : MonoBehaviour
    {
        [Tooltip("Stream client providing decoded rt-demo-v1 frames.")]
        public RTStreamClient client;

        [Tooltip("Optional halo ring transform (a torus/quad) for the glucose channel.")]
        public Transform glucoseHalo;

        [Header("Motion")]
        public float translateLerp = 0.12f;
        public float commandOffset = 1.4f;
        public float pushPullOffset = 1.2f;
        public float breatheAmplitude = 0.06f;

        static readonly Color Calm = new Color(0.15f, 0.42f, 0.92f);
        static readonly Color Stressed = new Color(0.86f, 0.15f, 0.15f);
        static readonly Color Abstain = new Color(0.58f, 0.64f, 0.72f);

        Vector3 _basePos;
        Vector3 _baseScale;
        Renderer _renderer;
        MaterialPropertyBlock _mpb;

        void Start()
        {
            _basePos = transform.localPosition;
            _baseScale = transform.localScale;
            _renderer = GetComponent<Renderer>();
            _mpb = new MaterialPropertyBlock();
        }

        void Update()
        {
            if (client == null) return;
            var f = client.Latest;
            if (f == null) return;

            // EEG command -> commanded pose (SkillSystem may override with a skill flourish)
            Vector3 offset = Vector3.zero;
            switch (f.command)
            {
                case "Left": offset = new Vector3(-commandOffset, 0, 0); break;
                case "Right": offset = new Vector3(commandOffset, 0, 0); break;
                case "Push": offset = new Vector3(0, 0, -pushPullOffset); break;
                case "Pull": offset = new Vector3(0, 0, pushPullOffset); break;
            }
            transform.localPosition = Vector3.Lerp(transform.localPosition, _basePos + offset, translateLerp);

            // PHR stress -> tint + a breathing pulse (faster/deeper as stress rises)
            float stress = Mathf.Clamp01(f.stress);
            var tint = Color.Lerp(Calm, Stressed, stress);
            float breathe = 1f + Mathf.Sin(Time.time * (1.4f + 2.6f * stress)) * breatheAmplitude * (0.6f + stress);
            transform.localScale = _baseScale * breathe;
            _renderer.GetPropertyBlock(_mpb);
            _mpb.SetColor("_Color", tint);
            _mpb.SetColor("_BaseColor", tint);       // URP/Lit
            _mpb.SetColor("_EmissionColor", tint * (0.25f + 0.6f * stress));
            _renderer.SetPropertyBlock(_mpb);

            // glucose forecast -> halo; abstain = grey + still (never a fabricated value)
            if (glucoseHalo != null)
            {
                bool abstained = f.abstained;
                var halo = glucoseHalo.GetComponent<Renderer>();
                if (halo != null)
                {
                    var c = abstained ? Abstain : new Color(0.07f, 0.72f, 0.45f);
                    c.a = abstained ? 0.25f : 0.65f;
                    halo.material.color = c;
                }
                glucoseHalo.Rotate(Vector3.up, (abstained ? 6f : 34f) * Time.deltaTime);
            }
        }
    }
}
