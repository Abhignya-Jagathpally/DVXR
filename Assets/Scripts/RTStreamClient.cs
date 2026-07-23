// EXPERIMENTAL RT-Demo WebSocket client for Unity.
//
// Consumes the rt-demo-v1 contract (see docs/RT_DEMO_STREAM_CONTRACT.md) from the DVXR
// backend at ws(s)://<host>/v1/realtime/stream and exposes the latest decoded frame. The
// SAME contract the web react-three-fiber scene uses, so the backend work is shared.
//
// DEMONSTRATION ONLY. The glucose channel abstains by construction; this client never
// fabricates a glucose value and surfaces Abstained so the UI can show "insufficient data".

using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DVXR.RTDemo
{
    [Serializable]
    public class RTFrame
    {
        public string contract;
        public int t;
        public string command = "Neutral";
        public float command_confidence;
        public float stress;
        // Nullable-in-JSON glucose fields; Unity's JsonUtility maps missing/null to 0, so
        // rely on `abstained` to decide whether the glucose values are meaningful.
        public float glucose_point;
        public float glucose_lower;
        public float glucose_upper;
        public bool abstained = true;
        public string evidence_status = "abstained";
        public bool experimental = true;
        public string disclaimer;
    }

    public class RTStreamClient : MonoBehaviour
    {
        [Tooltip("Backend base URL, e.g. http://localhost:8000 (http/https auto-mapped to ws/wss).")]
        public string baseUrl = "http://localhost:8000";

        [Tooltip("Frames per second requested from the server (interval = 1/rate).")]
        public float requestRateHz = 8f;

        public RTFrame Latest { get; private set; } = new RTFrame();
        public bool Connected { get; private set; }

        private ClientWebSocket _socket;
        private CancellationTokenSource _cts;

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

        private Uri StreamUri()
        {
            var ws = baseUrl.Replace("https://", "wss://").Replace("http://", "ws://");
            var interval = Mathf.Max(0f, 1f / Mathf.Max(0.1f, requestRateHz));
            return new Uri($"{ws}/v1/realtime/stream?interval={interval.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
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

                    try
                    {
                        var frame = JsonUtility.FromJson<RTFrame>(sb.ToString());
                        if (frame != null && frame.contract == "rt-demo-v1")
                        {
                            Latest = frame;
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[RTStreamClient] malformed frame: {e.Message}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // normal shutdown
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[RTStreamClient] connection error: {e.Message}. " +
                                 "Falling back to a static Neutral frame.");
                Connected = false;
            }
        }
    }
}
