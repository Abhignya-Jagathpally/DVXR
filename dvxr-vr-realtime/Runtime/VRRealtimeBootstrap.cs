// One-click setup for the DVXR VR-Realtime HUD.
//
// Attach this ONE component to any GameObject in a scene that already has an XR Rig / camera
// and press Play: it spawns the realtime client + the world-space VR stats HUD and wires them
// to your camera (and, optionally, to an avatar anchor). No manual Canvas/prefab work.
//
// If you already dropped `VRStatsHUD` on a GameObject yourself, you do NOT need this.

using UnityEngine;

namespace DVXR.VRRealtime
{
    public class VRRealtimeBootstrap : MonoBehaviour
    {
        [Header("Backend")]
        [Tooltip("DVXR backend (or the bundled mock server), e.g. http://localhost:8000.\n" +
                 "If unreachable the HUD animates from the deterministic demo generator.")]
        public string baseUrl = "http://localhost:8000";

        [Header("Personalization label (display only)")]
        public string participantLabel = "You";

        [Header("Placement")]
        public HudAnchorMode anchorMode = HudAnchorMode.HeadLockedBillboard;

        [Tooltip("FollowAnchor mode: the avatar transform to attach the readout to (e.g. left wrist).")]
        public Transform avatarAnchor;

        [Tooltip("Camera the HUD faces. Empty = Camera.main (the XR head camera).")]
        public Transform cameraOverride;

        private void Start()
        {
            var hudGO = new GameObject("DVXR_VRStatsHUD");
            hudGO.transform.SetParent(null, false);

            var client = hudGO.AddComponent<RtStreamClient>();
            client.baseUrl = baseUrl;

            var hud = hudGO.AddComponent<VRStatsHUD>();
            hud.client = client;
            hud.participantLabel = participantLabel;
            hud.anchorMode = anchorMode;
            hud.anchor = avatarAnchor;
            hud.cameraOverride = cameraOverride;

            Debug.Log($"[DVXR] VR-Realtime HUD spawned (backend={baseUrl}, anchor={anchorMode}). " +
                      "EXPERIMENTAL demonstration — not for clinical use.");
        }
    }
}
