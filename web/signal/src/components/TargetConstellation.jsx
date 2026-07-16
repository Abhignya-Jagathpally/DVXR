import { useEffect, useRef } from "react";
import { COLORS, hexA } from "../lib/signals.js";
import { clamp } from "../lib/researchModel.js";
import { TARGET_ORDER, TARGET_NAMES } from "../lib/researchPredictionTypes.js";
import { prefersReducedMotion } from "../lib/useReveal.js";

const MONO = '"Space Mono",ui-monospace,Menlo,monospace';
const { neural: NEU, physio: PHY, metabolic: MET, clinical: CLI, acc: ACC, white: WHT } = COLORS;
const COLS = {
  stress: PHY,
  anxiety: "#FF8FA8",
  depression: NEU,
  cognitive_workload: CLI,
  glucose_instability: MET,
};

// Node size = probability · opacity = confidence · link thickness = backend
// model contribution (node_contrib), NEVER derived from probability alone.
export default function TargetConstellation({ result }) {
  const ref = useRef(null);
  const dataRef = useRef(result);
  dataRef.current = result;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const rm = prefersReducedMotion();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ctx = cv.getContext("2d");
    let W = 0, H = 0, raf = 0, t0 = 0;

    function fit() {
      const r = cv.getBoundingClientRect();
      W = r.width;
      H = r.height;
      cv.width = Math.max(1, Math.round(W * dpr));
      cv.height = Math.max(1, Math.round(H * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    fit();
    const onResize = () => fit();
    window.addEventListener("resize", onResize);

    function frame(ts) {
      const r = dataRef.current;
      if (!t0) t0 = ts;
      const t = (ts - t0) / 1000;
      ctx.clearRect(0, 0, W, H);
      const cx = W * 0.5, cy = H * 0.52, R = Math.min(W, H) * 0.33;
      const spin = rm ? 0 : t * 0.15;

      // links (thickness from model contribution)
      TARGET_ORDER.forEach((k, i) => {
        const a = -Math.PI / 2 + (i / TARGET_ORDER.length) * Math.PI * 2 + spin;
        const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
        const contrib = (r.node_contrib && r.node_contrib[k]) || 0;
        const lw = clamp(contrib * 14, 0.5, 7);
        ctx.strokeStyle = hexA(COLS[k], 0.35);
        ctx.lineWidth = lw;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // center outcome node
      const sel = r.selected && r.selected.probability;
      if (sel != null) {
        ctx.fillStyle = hexA(ACC, 0.16);
        ctx.beginPath();
        ctx.arc(cx, cy, 26, 0, 7);
        ctx.fill();
        ctx.fillStyle = ACC;
        ctx.font = "600 13px " + MONO;
        ctx.textAlign = "center";
        ctx.fillText(Math.round(sel * 100) + "%", cx, cy + 4);
      }

      // target nodes
      TARGET_ORDER.forEach((k, i) => {
        const a = -Math.PI / 2 + (i / TARGET_ORDER.length) * Math.PI * 2 + spin;
        const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
        const tp = r.target_predictions[k];
        const rad = 6 + tp.probability * 20;
        const op = 0.35 + tp.confidence * 0.6;
        const g = ctx.createRadialGradient(x, y, 1, x, y, rad * 1.6);
        g.addColorStop(0, hexA(COLS[k], op));
        g.addColorStop(1, hexA(COLS[k], 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, rad * 1.6, 0, 7);
        ctx.fill();
        ctx.fillStyle = hexA(COLS[k], op);
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, 7);
        ctx.fill();
        ctx.fillStyle = WHT;
        ctx.font = "10px " + MONO;
        ctx.textAlign = "center";
        const lx = cx + Math.cos(a) * (R + rad + 14), ly = cy + Math.sin(a) * (R + rad + 14);
        ctx.fillText(TARGET_NAMES[k], lx, ly);
      });

      if (!rm) raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    if (rm) frame(0);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      className="sim-const-canvas"
      aria-label="Target constellation. Node size shows probability, opacity shows confidence, link thickness shows this model's contribution of each signal to the estimate."
    />
  );
}
