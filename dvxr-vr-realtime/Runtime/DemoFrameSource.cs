// Deterministic client-side rt-demo-v1 generator — the VR scene's DEMO_MODE fallback.
//
// A faithful port of the backend `realtime_bridge.build_frame` (pure function of the frame
// index) so the VR HUD animates identically whether frames come from the live socket or from
// here. Glucose ABSTAINS by default — a value is emitted ONLY when the caller passes a
// clearly-labelled demo trace, never fabricated.

using UnityEngine;

namespace DVXR.VRRealtime
{
    public static class DemoFrameSource
    {
        // Matches realtime_bridge._DEFAULT_COMMAND_PATTERN (bci_real avatar analog).
        private static readonly string[] Pattern =
            { "Neutral", "Left", "Neutral", "Right", "Push", "Neutral", "Pull", "Neutral" };

        // realtime_bridge._stress_at: smooth deterministic stress index in [0,1].
        private static float StressAt(int i) => 0.5f + 0.35f * Mathf.Sin(i * 0.15f);

        // realtime_bridge._confidence_at: deterministic command confidence in [0.55, 0.9].
        private static float ConfidenceAt(int i) => 0.55f + 0.35f * (0.5f + 0.5f * Mathf.Sin(i * 0.27f + 1.3f));

        public static RtFrame Build(int index, float[] glucoseTrace = null)
        {
            var f = new RtFrame
            {
                contract = "rt-demo-v1",
                t = index,
                command = Pattern[((index % Pattern.Length) + Pattern.Length) % Pattern.Length],
                command_confidence = ConfidenceAt(index),
                stress = Mathf.Clamp01(StressAt(index)),
                experimental = true,
                disclaimer = "EXPERIMENTAL demonstration stream — not clinical inference.",
            };

            if (glucoseTrace != null && glucoseTrace.Length > 0)
            {
                var point = glucoseTrace[((index % glucoseTrace.Length) + glucoseTrace.Length) % glucoseTrace.Length];
                f.glucose_point = Mathf.Round(point * 10f) / 10f;
                f.glucose_lower = f.glucose_point - 18f;
                f.glucose_upper = f.glucose_point + 18f;
                f.abstained = false;
                f.evidence_status = "experimental";
            }
            else
            {
                // Honest default: no synchronized per-subject EEG+CGM data -> abstain.
                f.abstained = true;
                f.evidence_status = "abstained";
            }
            return f;
        }
    }
}
