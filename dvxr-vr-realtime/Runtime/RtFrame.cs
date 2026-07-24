// DVXR VR-Realtime — rt-demo-v1 frame + streaming client.
//
// Consumes the SAME `rt-demo-v1` contract the DVXR backend serves over WebSocket at
// ws(s)://<host>/v1/realtime/stream (and that the web react-three-fiber scene + the in-repo
// Unity DVXR_RT_Demo scene already use). One schema, shared everywhere.
//
// EXPERIMENTAL / DEMONSTRATION ONLY. The glucose channel abstains by construction — this
// client NEVER fabricates a glucose value; it surfaces `abstained` so the VR HUD renders an
// explicit "insufficient data" state. Not clinical inference.

using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DVXR.VRRealtime
{
    /// <summary>One rt-demo-v1 frame. Field names match the JSON contract exactly so
    /// UnityEngine.JsonUtility maps them directly. Missing/null JSON numbers map to 0 in
    /// Unity — always consult <see cref="abstained"/> before trusting the glucose fields.</summary>
    [Serializable]
    public class RtFrame
    {
        public string contract = "rt-demo-v1";
        public int t;
        public string command = "Neutral";
        public float command_confidence;
        public float stress;
        public float glucose_point;
        public float glucose_lower;
        public float glucose_upper;
        public bool abstained = true;
        public string evidence_status = "abstained";
        public bool experimental = true;
        public string disclaimer = "EXPERIMENTAL demonstration stream — not clinical inference.";

        public bool IsGlucoseKnown => !abstained && evidence_status != "abstained";
    }

    /// <summary>Connects to the DVXR realtime stream and exposes the latest decoded frame on
    /// the main thread. Falls back to a deterministic in-process demo generator
    /// (<see cref="DemoFrameSource"/>) when no backend is reachable, so the VR scene always
    /// animates — mirroring the web scene's DEMO_MODE.</summary>
    public class RtStreamClient : MonoBehaviour
    {
        [Tooltip("Backend base URL, e.g. http://localhost:8000 (http/https auto-mapped to ws/wss).")]
        public string baseUrl = "http://localhost:8000";

        [Tooltip("Frames per second requested from the server (interval = 1/rate).")]
        public float requestRateHz = 8f;

        [Tooltip("If the backend is unreachable, animate from the deterministic demo generator.")]
        public bool demoFallback = true;

        [Tooltip("Optional: supply a clearly-labelled demo glucose trace (mg/dL) to the demo\n" +
                 "generator ONLY. Leave empty to keep glucose abstaining (the honest default).")]
        public float[] demoGlucoseTrace = new float[0];

        /// <summary>Latest frame (never null). Read from the main thread (Update/OnGUI).</summary>
        public RtFrame Latest { get; private set; } = new RtFrame();

        /// <summary>True while a live backend socket is open. False in demo-fallback mode.</summary>
        public bool Connected { get; private set; }

        /// <summary>"live" when streaming from the backend, "demo" when generated locally.</summary>
        public string SourceLabel => Connected ? "live" : "demo";

        private ClientWebSocket _socket;
        private CancellationTokenSource _cts;
        private volatile string _pendingJson;   // set on socket thread, parsed on main thread
        private int _demoIndex;
        private float _demoClock;

        private void OnEnable()
        {
            _cts = new CancellationTokenSource();
            _ = RunAsync(_cts.Token);
        }

        private void OnDisable()
        {
            try { _cts?.Cancel(); } catch { /* noop */ }
            try { _socket?.Dispose(); } catch { /* noop */ }
            Connected = false;
        }

        private void Update()
        {
            // Parse any frame the socket thread queued (JsonUtility must run on main thread).
            var json = _pendingJson;
            if (json != null)
            {
                _pendingJson = null;
                try
                {
                    var frame = JsonUtility.FromJson<RtFrame>(json);
                    if (frame != null && frame.contract == "rt-demo-v1") Latest = frame;
                }
                catch (Exception e) { Debug.LogWarning($"[DVXR] malformed frame: {e.Message}"); }
            }

            // Demo fallback: advance the deterministic generator at the requested rate.
            if (!Connected && demoFallback)
            {
                _demoClock += Time.deltaTime;
                var interval = 1f / Mathf.Max(0.1f, requestRateHz);
                if (_demoClock >= interval)
                {
                    _demoClock -= interval;
                    Latest = DemoFrameSource.Build(_demoIndex++, demoGlucoseTrace);
                }
            }
        }

        private Uri StreamUri()
        {
            var ws = baseUrl.Replace("https://", "wss://").Replace("http://", "ws://");
            var interval = Mathf.Max(0f, 1f / Mathf.Max(0.1f, requestRateHz));
            var inv = interval.ToString(System.Globalization.CultureInfo.InvariantCulture);
            return new Uri($"{ws}/v1/realtime/stream?interval={inv}");
        }

        private async Task RunAsync(CancellationToken token)
        {
            var buffer = new byte[8192];
            try
            {
                _socket = new ClientWebSocket();
                await _socket.ConnectAsync(StreamUri(), token);
                Connected = true;
                var sb = new StringBuilder();
                while (!token.IsCancellationRequested && _socket.State == WebSocketState.Open)
                {
                    sb.Clear();
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await _socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", token);
                            Connected = false;
                            return;
                        }
                        sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                    } while (!result.EndOfMessage);
                    _pendingJson = sb.ToString();   // hand off to main thread for parsing
                }
            }
            catch (OperationCanceledException) { /* normal shutdown */ }
            catch (Exception e)
            {
                Debug.LogWarning($"[DVXR] realtime connection error: {e.Message}. " +
                                 "Falling back to deterministic demo frames.");
                Connected = false;
            }
        }
    }
}
