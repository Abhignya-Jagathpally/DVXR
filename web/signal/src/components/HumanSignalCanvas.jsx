import { useEffect, useRef } from "react";
import { COLORS, hexA } from "../lib/signals.js";
import { clamp } from "../lib/researchModel.js";
import { prefersReducedMotion } from "../lib/useReveal.js";

const { neural: NEU, physio: PHY, metabolic: MET, clinical: CLI, acc: ACC } = COLORS;

// Center human silhouette. Regions react to current inputs:
// head = neural, chest = cardiovascular, hand = electrodermal, abdomen = metabolic.
export default function HumanSignalCanvas({ inputs }) {
  const ref = useRef(null);
  const inputsRef = useRef(inputs);
  inputsRef.current = inputs;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const rm = prefersReducedMotion();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ctx = cv.getContext("2d");
    let W = 0, H = 0, raf = 0;

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

    const v = (g, k) => {
      const s = inputsRef.current[g] && inputsRef.current[g][k];
      return s && s.present ? s.value : null;
    };

    function frame(ts) {
      const t = (ts || 0) / 1000;
      const neu = clamp((v("neural", "beta_alpha_ratio") || 0.9) / 2, 0, 1);
      const phy = clamp((90 - (v("physiological", "hrv_rmssd_ms") || 40)) / 70, 0, 1);
      const eda = clamp((v("physiological", "eda_microsiemens") || 3) / 12, 0, 1);
      const met = clamp((v("metabolic", "glucose_cv_percent") || 18) / 40, 0, 1);

      ctx.clearRect(0, 0, W, H);
      const cx = W * 0.5, top = H * 0.08, hw = Math.min(W * 0.16, 74), hh = hw * 1.15;
      const sil = () => {
        ctx.beginPath();
        ctx.moveTo(cx - hw * 1.7, H);
        ctx.bezierCurveTo(cx - hw * 1.5, H * 0.55, cx - hw * 1.05, top + hh * 1.2, cx - hw * 0.7, top + hh);
        ctx.bezierCurveTo(cx - hw * 1.0, top + hh * 0.5, cx - hw, top, cx, top);
        ctx.bezierCurveTo(cx + hw, top, cx + hw * 1.0, top + hh * 0.5, cx + hw * 0.7, top + hh);
        ctx.bezierCurveTo(cx + hw * 1.05, top + hh * 1.2, cx + hw * 1.5, H * 0.55, cx + hw * 1.7, H);
        ctx.closePath();
      };
      sil();
      ctx.save();
      ctx.clip();
      const gr = ctx.createLinearGradient(0, 0, 0, H);
      gr.addColorStop(0, "#0e1220");
      gr.addColorStop(1, "#0a0c14");
      ctx.fillStyle = gr;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();
      sil();
      ctx.strokeStyle = hexA(ACC, 0.18);
      ctx.lineWidth = 1;
      ctx.stroke();

      const glow = (x, y, r, color, inten) => {
        const pulse = rm ? 1 : 0.85 + 0.15 * Math.sin(t * 2);
        const rr = r * (0.7 + inten * 0.9) * pulse;
        const g2 = ctx.createRadialGradient(x, y, 1, x, y, rr);
        g2.addColorStop(0, hexA(color, 0.5 + inten * 0.4));
        g2.addColorStop(1, hexA(color, 0));
        ctx.fillStyle = g2;
        ctx.beginPath();
        ctx.arc(x, y, rr, 0, 7);
        ctx.fill();
        ctx.fillStyle = hexA(color, 0.9);
        ctx.beginPath();
        ctx.arc(x, y, 2.4, 0, 7);
        ctx.fill();
      };
      const headY = top + hh * 0.5, chestY = top + hh * 1.7, handY = top + hh * 2.0, abY = top + hh * 2.6;
      glow(cx, headY, 42, NEU, neu);
      glow(cx, chestY, 40, PHY, phy);
      glow(cx - hw * 1.35, handY, 26, CLI, eda);
      glow(cx, abY, 44, MET, met);

      if (!rm) raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    // Redraw on input change even under reduced motion.
    const poll = rm
      ? setInterval(() => frame(0), 400)
      : null;

    return () => {
      cancelAnimationFrame(raf);
      if (poll) clearInterval(poll);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      className="sim-human"
      aria-label="Human signal map: head reacts to neural arousal, chest to cardiovascular state, hand to electrodermal activity, abdomen to metabolic variability."
    />
  );
}
