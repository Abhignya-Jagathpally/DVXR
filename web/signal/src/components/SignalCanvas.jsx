import { useEffect, useRef } from "react";
import { prefersReducedMotion } from "../lib/useReveal.js";

const MONO = '"IBM Plex Mono",ui-monospace,Menlo,monospace';

// Reusable animated canvas. `draw(ctx, w, h, t, rm)` is called each frame.
export default function SignalCanvas({ draw, className, style, ariaLabel }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const rm = prefersReducedMotion();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let raf = 0, t0 = 0, visible = true;
    const ctx = cv.getContext("2d");
    function fit() {
      const r = cv.getBoundingClientRect();
      cv.width = Math.max(1, Math.round(r.width * dpr));
      cv.height = Math.max(1, Math.round(r.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return r;
    }
    let rect = fit();
    const onResize = () => { rect = fit(); };
    window.addEventListener("resize", onResize);
    const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0.03 });
    io.observe(cv);
    function loop(ts) {
      if (!t0) t0 = ts;
      const t = (ts - t0) / 1000;
      if (visible) draw(ctx, rect.width, rect.height, t, rm, MONO);
      if (!rm) raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);
    if (rm) draw(ctx, rect.width, rect.height, 0.4, true, MONO);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); io.disconnect(); };
  }, [draw]);
  return <canvas ref={ref} className={className} style={style} aria-label={ariaLabel} aria-hidden={ariaLabel ? undefined : "true"} />;
}
