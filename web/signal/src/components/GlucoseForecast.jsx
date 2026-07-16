import { useEffect, useRef } from "react";
import { COLORS, hexA } from "../lib/signals.js";
import { clamp } from "../lib/researchModel.js";

const MONO = '"Space Mono",ui-monospace,Menlo,monospace';
const { metabolic: MET, white: WHT } = COLORS;

// Near-term glucose outlook: recent history + 30/60-min points + interval band.
export default function GlucoseForecast({ forecast }) {
  const ref = useRef(null);
  const fcRef = useRef(forecast);
  fcRef.current = forecast;

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !forecast) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ctx = cv.getContext("2d");
    let W = 0, H = 0;

    function draw() {
      const r = cv.getBoundingClientRect();
      W = r.width;
      H = r.height;
      cv.width = Math.max(1, Math.round(W * dpr));
      cv.height = Math.max(1, Math.round(H * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const fc = fcRef.current;
      const ymap = (v) => {
        const lo = 60, hi = 260;
        return H - 18 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (H - 34);
      };
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = hexA(WHT, 0.06);
      [80, 120, 160, 200].forEach((g) => {
        const y = ymap(g);
        ctx.beginPath();
        ctx.moveTo(28, y);
        ctx.lineTo(W, y);
        ctx.stroke();
        ctx.fillStyle = hexA(WHT, 0.3);
        ctx.font = "9px " + MONO;
        ctx.fillText(String(g), 4, y + 3);
      });
      const x0 = 34, xn = W - 8, xnow = x0 + (xn - x0) * 0.34;
      // history (gentle wobble up to now)
      ctx.beginPath();
      for (let i = 0; i <= 20; i++) {
        const x = x0 + (xnow - x0) * (i / 20);
        const val = fc.history_last + Math.sin(i * 0.6) * 4;
        const y = ymap(val);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = hexA(WHT, 0.7);
      ctx.lineWidth = 1.8;
      ctx.stroke();
      const p30 = fc["30"], p60 = fc["60"];
      const xn30 = xnow + (xn - xnow) * 0.5, xn60 = xn;
      // interval band
      ctx.beginPath();
      ctx.moveTo(xnow, ymap(fc.history_last));
      ctx.lineTo(xn30, ymap(p30.upper_mg_dl));
      ctx.lineTo(xn60, ymap(p60.upper_mg_dl));
      ctx.lineTo(xn60, ymap(p60.lower_mg_dl));
      ctx.lineTo(xn30, ymap(p30.lower_mg_dl));
      ctx.closePath();
      ctx.fillStyle = hexA(MET, 0.14);
      ctx.fill();
      // point line
      ctx.beginPath();
      ctx.moveTo(xnow, ymap(fc.history_last));
      ctx.lineTo(xn30, ymap(p30.point_mg_dl));
      ctx.lineTo(xn60, ymap(p60.point_mg_dl));
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = hexA(MET, 0.95);
      ctx.lineWidth = 1.9;
      ctx.stroke();
      ctx.setLineDash([]);
      // "now" marker
      ctx.strokeStyle = hexA(MET, 0.3);
      ctx.beginPath();
      ctx.moveTo(xnow, 0);
      ctx.lineTo(xnow, H);
      ctx.stroke();
      [[xn30, p30], [xn60, p60]].forEach((pp) => {
        ctx.fillStyle = MET;
        ctx.beginPath();
        ctx.arc(pp[0], ymap(pp[1].point_mg_dl), 4, 0, 7);
        ctx.fill();
      });
      ctx.fillStyle = hexA(WHT, 0.4);
      ctx.font = "9px " + MONO;
      ctx.fillText("now", xnow - 8, H - 4);
      ctx.fillText("+30", xn30 - 8, H - 4);
      ctx.fillText("+60", xn60 - 16, H - 4);
    }

    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [forecast]);

  if (!forecast) return null;
  const p30 = forecast["30"], p60 = forecast["60"];
  return (
    <div className="sim-fc">
      <canvas
        ref={ref}
        className="sim-fc-canvas"
        aria-label={`Glucose forecast. Plus 30 minutes ${p30.point_mg_dl} mg/dL, interval ${p30.lower_mg_dl} to ${p30.upper_mg_dl}. Plus 60 minutes ${p60.point_mg_dl} mg/dL, interval ${p60.lower_mg_dl} to ${p60.upper_mg_dl}.`}
      />
      <div className="sim-fc-reads">
        <div>
          <span>+30 min</span>
          <b>
            {p30.point_mg_dl} <i>mg/dL</i>
          </b>
          <small>
            {p30.lower_mg_dl}–{p30.upper_mg_dl}
          </small>
        </div>
        <div>
          <span>+60 min</span>
          <b>
            {p60.point_mg_dl} <i>mg/dL</i>
          </b>
          <small>
            {p60.lower_mg_dl}–{p60.upper_mg_dl}
          </small>
        </div>
      </div>
    </div>
  );
}
