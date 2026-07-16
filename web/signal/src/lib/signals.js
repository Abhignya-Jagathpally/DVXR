// Shared canvas draw functions — ported from the artifact for visual parity.
export const COLORS = {
  acc: "#2EE6C6", acc2: "#66F2DD", white: "#F3F5FA",
  gray: "#606a80", amber: "#FFB44C", red: "#FF5D6C",
  neural: "#B07CFF", physio: "#FF6584", metabolic: "#FFB44C",
  clinical: "#38C6FF", molecular: "#4BE38F",
};
const MC = { neural: "#B07CFF", physio: "#FF6584", metabolic: "#FFB44C", clinical: "#38C6FF", molecular: "#4BE38F" };
const SC = { capture: "#B07CFF", align: "#FF6584", encode: "#FFB44C", fuse: "#38C6FF", predict: "#4BE38F", explain: "#2EE6C6" };
const MINI = { hrv: "#FF6584", eda: "#38C6FF", resp: "#4BE38F", mot: "#2EE6C6", conf: "#606a80" };

export function hexA(hex, a) {
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = hex.split("").map((x) => x + x).join("");
  const n = parseInt(hex, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

const C = COLORS;

export function silhouette(c, w, h) {
  const cx = w * 0.62, top = h * 0.2, hw = Math.min(w * 0.12, 120), hh = hw * 1.25;
  c.beginPath();
  c.moveTo(cx - hw * 1.9, h);
  c.bezierCurveTo(cx - hw * 1.7, h * 0.62, cx - hw * 1.15, top + hh * 1.15, cx - hw * 0.75, top + hh);
  c.bezierCurveTo(cx - hw * 1.05, top + hh * 0.55, cx - hw, top, cx, top);
  c.bezierCurveTo(cx + hw, top, cx + hw * 1.05, top + hh * 0.55, cx + hw * 0.75, top + hh);
  c.bezierCurveTo(cx + hw * 1.15, top + hh * 1.15, cx + hw * 1.7, h * 0.62, cx + hw * 1.9, h);
  c.closePath();
  return { cx, top, hw, hh };
}

export function heroDraw(c, w, h, t, rm) {
  c.clearRect(0, 0, w, h);
  const g = silhouette(c, w, h);
  c.save(); c.clip();
  const grad = c.createLinearGradient(0, g.top, 0, h);
  grad.addColorStop(0, "#0d1020"); grad.addColorStop(1, "#080a10");
  c.fillStyle = grad; c.fillRect(0, 0, w, h);
  const chestY = g.top + g.hh * 1.7;
  c.beginPath();
  for (let x = 0; x <= w; x += 3) {
    const y = chestY + Math.sin(x * 0.02 + t) * 3 + (Math.abs(((x * 0.9 - (rm ? 0 : t * 90)) % 180) - 90) < 6 ? -18 * Math.sign(Math.sin(x)) : 0);
    x === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
  }
  c.strokeStyle = hexA(C.physio, 0.55); c.lineWidth = 1.5; c.stroke();
  const abY = g.top + g.hh * 2.7;
  c.beginPath();
  for (let x = 0; x <= w; x += 4) {
    const y = abY - (Math.sin(x * 0.011 + (rm ? 0 : t * 0.5)) * 10 + Math.sin(x * 0.03 - (rm ? 0 : t * 0.3)) * 5);
    x === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
  }
  c.strokeStyle = hexA(C.metabolic, 0.4); c.lineWidth = 1.3; c.stroke();
  c.restore();
  const hx = g.cx, hy = g.top + g.hh * 0.5;
  for (let i = 0; i < 3; i++) {
    c.beginPath();
    const rad = g.hw * (1.25 + i * 0.28);
    for (let a = -Math.PI * 0.9; a <= -Math.PI * 0.1; a += 0.05) {
      const jr = rad + Math.sin(a * 7 + (rm ? 0 : t * 2) + i) * 4;
      const px = hx + Math.cos(a) * jr, py = hy + Math.sin(a) * jr * 1.05;
      a <= -Math.PI * 0.9 + 0.05 ? c.moveTo(px, py) : c.lineTo(px, py);
    }
    c.strokeStyle = hexA(C.neural, 0.16 + i * 0.07); c.lineWidth = 1.2; c.stroke();
  }
  for (let k = 0; k < 28; k++) {
    const seed = k * 97.13, px = (seed * 13.7) % w, py = (seed * 7.3) % h;
    const tw = rm ? 0.5 : 0.5 + 0.5 * Math.sin(t * 1.2 + k);
    c.globalAlpha = 0.06 + tw * 0.12; c.fillStyle = k % 3 === 0 ? C.clinical : (k % 3 === 1 ? C.molecular : C.white); c.beginPath(); c.arc(px, py, 1.2, 0, 7); c.fill();
  }
  c.globalAlpha = 1;
  silhouette(c, w, h); c.strokeStyle = hexA(C.acc, 0.22); c.lineWidth = 1.1; c.stroke();
}

export function worldDraw(kind) {
  const col = MC[kind] || C.acc;
  return (c, w, h, t) => {
    c.clearRect(0, 0, w, h);
    c.strokeStyle = hexA(C.white, 0.05);
    for (let gy = 0; gy < h; gy += Math.max(20, h / 9)) { c.beginPath(); c.moveTo(0, gy); c.lineTo(w, gy); c.stroke(); }
    if (kind === "neural") {
      for (let ch = 0; ch < 5; ch++) {
        const mid = h * (0.2 + ch * 0.15); c.beginPath();
        for (let x = 0; x <= w; x += 3) { const y = mid + (Math.sin(x * 0.05 + t * 2 + ch) * 0.5 + Math.sin(x * 0.13 + t * 3 + ch * 2) * 0.3 + Math.sin(x * 0.4 + t * 5) * 0.12) * h * 0.05; x === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }
        c.strokeStyle = hexA(ch === 2 ? col : C.white, ch === 2 ? 0.95 : 0.22); c.lineWidth = ch === 2 ? 1.8 : 1; c.stroke();
      }
    } else if (kind === "physio") {
      const midY = h * 0.55; c.beginPath();
      for (let x = 0; x <= w; x += 2) { const pk = (x % (w / 4)) / (w / 4); const sp = Math.exp(-Math.pow((pk - 0.5) * 8, 2)) * h * 0.22; const y = midY - sp + Math.sin(x * 0.02 + t) * 3; x === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }
      c.strokeStyle = hexA(col, 0.95); c.lineWidth = 1.8; c.stroke();
      c.beginPath(); for (let x = 0; x <= w; x += 3) { const y = h * 0.82 + Math.sin(x * 0.04 + t * 1.5) * h * 0.05; x === 0 ? c.moveTo(x, y) : c.lineTo(x, y); } c.strokeStyle = hexA(C.white, 0.2); c.lineWidth = 1; c.stroke();
    } else if (kind === "metabolic") {
      const by = h * 0.6, pts = []; for (let x = 0; x <= w; x += 4) { const y = by - (Math.sin(x * 0.012 + t * 0.5) * h * 0.12 + Math.sin(x * 0.03 - t * 0.3) * h * 0.05); pts.push([x, y]); }
      const cut = Math.floor(pts.length * 0.62);
      c.beginPath(); pts.slice(0, cut + 1).forEach((p, i) => (i === 0 ? c.moveTo(p[0], p[1]) : c.lineTo(p[0], p[1]))); c.strokeStyle = hexA(col, 0.95); c.lineWidth = 1.9; c.stroke();
      const up = [], dn = []; c.beginPath(); c.moveTo(pts[cut][0], pts[cut][1]);
      for (let i = cut; i < pts.length; i++) { const f = (i - cut) / (pts.length - cut); const spread = f * h * 0.16; up.push([pts[i][0], pts[i][1] - spread]); dn.push([pts[i][0], pts[i][1] + spread]); c.lineTo(pts[i][0], pts[i][1]); }
      c.setLineDash([4, 4]); c.strokeStyle = hexA(col, 0.8); c.lineWidth = 1.4; c.stroke(); c.setLineDash([]);
      c.beginPath(); up.forEach((p, i) => (i === 0 ? c.moveTo(p[0], p[1]) : c.lineTo(p[0], p[1]))); for (let j = dn.length - 1; j >= 0; j--) c.lineTo(dn[j][0], dn[j][1]); c.closePath(); c.fillStyle = hexA(col, 0.12); c.fill();
    } else if (kind === "clinical") {
      const ty = h * 0.5; c.strokeStyle = hexA(C.white, 0.18); c.beginPath(); c.moveTo(w * 0.08, ty); c.lineTo(w * 0.92, ty); c.stroke();
      for (let k = 0; k < 9; k++) { const tx = w * (0.12 + k * 0.09); const conv = Math.sin(t * 0.8 + k) * 0.5 + 0.5; const oy = ty + (k % 2 ? -1 : 1) * (1 - conv) * h * 0.28; c.fillStyle = hexA(k === 4 ? col : C.white, k === 4 ? 0.95 : 0.4); c.beginPath(); c.arc(tx, oy, k === 4 ? 4 : 2.5, 0, 7); c.fill(); c.strokeStyle = hexA(col, 0.14); c.beginPath(); c.moveTo(tx, oy); c.lineTo(tx, ty); c.stroke(); }
    } else if (kind === "molecular") {
      const N = 12, nodes = []; for (let i = 0; i < N; i++) { const an = (i / N) * Math.PI * 2 + t * 0.15; const rr = h * 0.3 * (0.6 + 0.4 * Math.sin(t * 0.5 + i)); nodes.push([w / 2 + Math.cos(an) * rr, h / 2 + Math.sin(an) * rr * 0.8]); }
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if ((i * j) % 3 === 0) { c.strokeStyle = hexA(col, 0.10); c.beginPath(); c.moveTo(nodes[i][0], nodes[i][1]); c.lineTo(nodes[j][0], nodes[j][1]); c.stroke(); }
      nodes.forEach((n) => { c.fillStyle = hexA(col, 0.6); c.beginPath(); c.arc(n[0], n[1], 2.2, 0, 7); c.fill(); });
    }
  };
}

export function sigDraw(kind) {
  const col = SC[kind] || C.acc;
  return (c, w, h, t) => {
    c.clearRect(0, 0, w, h); const mid = h / 2; c.strokeStyle = hexA(col, 0.85); c.lineWidth = 1.4;
    if (kind === "capture") { c.beginPath(); for (let x = 0; x <= w; x += 3) { const y = mid + Math.sin(x * 0.2 + t * 3) * h * 0.28 * Math.exp((-x / w) * 1.5); c.lineTo(x, y); } c.stroke(); }
    else if (kind === "align") { for (let i = 0; i < 4; i++) { const yy = h * (0.25 + i * 0.18); c.strokeStyle = hexA(i === 1 ? col : C.white, i === 1 ? 0.85 : 0.2); c.beginPath(); c.moveTo(0, yy); c.lineTo(w * (0.4 + 0.15 * i) + Math.sin(t + i) * 6, yy); c.stroke(); } c.strokeStyle = hexA(C.acc, 0.5); c.beginPath(); c.moveTo(w * 0.72, 0); c.lineTo(w * 0.72, h); c.stroke(); }
    else if (kind === "encode") { for (let i = 0; i < 6; i++) { const bx = (i / 6) * w + 4; c.strokeStyle = hexA(i === 2 ? col : C.white, i === 2 ? 0.85 : 0.22); c.strokeRect(bx, mid - 8 + Math.sin(t + i) * 4, w / 6 - 6, 16); } }
    else if (kind === "fuse") { for (let i = 0; i < 5; i++) { c.strokeStyle = hexA(C.white, 0.2); c.beginPath(); c.moveTo(0, h * (0.15 + i * 0.18)); c.lineTo(w * 0.5, mid); c.stroke(); } c.strokeStyle = hexA(col, 0.9); c.beginPath(); c.moveTo(w * 0.5, mid); c.lineTo(w, mid); c.stroke(); c.fillStyle = hexA(col, 0.95); c.beginPath(); c.arc(w * 0.5, mid, 3, 0, 7); c.fill(); }
    else if (kind === "predict") { const bars = [0.4, 0.75, 0.55, 0.9, 0.3]; bars.forEach((b, i) => { const bx = (i / 5) * w + 6; const bh = b * h * 0.6 * (0.7 + 0.3 * Math.sin(t * 2 + i)); c.fillStyle = hexA(i === 3 ? col : C.white, i === 3 ? 0.9 : 0.2); c.fillRect(bx, mid + h * 0.28 - bh, w / 5 - 8, bh); }); }
    else if (kind === "explain") { c.strokeStyle = hexA(col, 0.7); for (let i = 0; i < 3; i++) { c.beginPath(); c.moveTo(4, h * (0.3 + i * 0.2)); c.lineTo(4 + w * (0.5 + 0.3 * Math.abs(Math.sin(t + i))), h * (0.3 + i * 0.2)); c.stroke(); } }
  };
}

export function cgmDraw(mono) {
  return (c, w, h, t, rm) => {
    c.clearRect(0, 0, w, h);
    c.strokeStyle = hexA(C.white, 0.05); for (let gy = h * 0.15; gy < h; gy += h * 0.2) { c.beginPath(); c.moveTo(0, gy); c.lineTo(w, gy); c.stroke(); }
    const cut = w * 0.6, mid = h * 0.55, hist = [];
    for (let x = 0; x <= cut; x += 3) { const y = mid - Math.sin(x * 0.02 + 0.5) * h * 0.12 - (x / cut) * h * 0.06; hist.push([x, y]); }
    c.beginPath(); hist.forEach((p, i) => (i === 0 ? c.moveTo(p[0], p[1]) : c.lineTo(p[0], p[1]))); c.strokeStyle = hexA(C.white, 0.75); c.lineWidth = 1.8; c.stroke();
    const last = hist[hist.length - 1], fc = [], up = [], dn = [];
    for (let x = cut; x <= w; x += 4) { const f = (x - cut) / (w - cut); const y = last[1] - f * h * 0.12; const sp = f * h * 0.18 * (0.9 + 0.1 * Math.sin(t)); fc.push([x, y]); up.push([x, y - sp]); dn.push([x, y + sp]); }
    c.beginPath(); up.forEach((p, i) => (i === 0 ? c.moveTo(p[0], p[1]) : c.lineTo(p[0], p[1]))); for (let j = dn.length - 1; j >= 0; j--) c.lineTo(dn[j][0], dn[j][1]); c.closePath(); c.fillStyle = hexA(C.metabolic, 0.14); c.fill();
    c.beginPath(); c.moveTo(last[0], last[1]); fc.forEach((p) => c.lineTo(p[0], p[1])); c.setLineDash([5, 5]); c.strokeStyle = hexA(C.metabolic, 0.95); c.lineWidth = 1.8; c.stroke(); c.setLineDash([]);
    c.strokeStyle = hexA(C.metabolic, 0.35); c.beginPath(); c.moveTo(cut, 0); c.lineTo(cut, h); c.stroke();
    c.fillStyle = C.gray; c.font = `10px ${mono}`; c.fillText("now", cut + 5, 14);
    const ep = fc[fc.length - 1]; c.fillStyle = C.metabolic; c.beginPath(); c.arc(ep[0] - 2, ep[1], 3.6, 0, 7); c.fill();
    const pr = rm ? 7 : 7 + Math.sin(t * 3) * 3; c.strokeStyle = hexA(C.metabolic, 0.45); c.beginPath(); c.arc(ep[0] - 2, ep[1], pr, 0, 7); c.stroke();
  };
}

export function miniDraw(kind, off, mono) {
  const col = MINI[kind] || C.acc;
  return (c, w, h, t, rm) => {
    const mid = h / 2; c.clearRect(0, 0, w, h);
    if (off) { c.strokeStyle = hexA(C.red, 0.35); c.setLineDash([3, 4]); c.beginPath(); c.moveTo(0, mid); c.lineTo(w, mid); c.stroke(); c.setLineDash([]); c.fillStyle = hexA(C.red, 0.5); c.font = `9px ${mono}`; c.fillText("no signal", w / 2 - 24, mid - 6); return; }
    c.beginPath();
    for (let x = 0; x <= w; x += 2) {
      let y;
      if (kind === "hrv") y = mid + Math.sin(x * 0.15 + t * 2) * h * 0.28;
      else if (kind === "eda") y = mid - (x / w) * h * 0.18 + Math.sin(x * 0.05 + t) * h * 0.06;
      else if (kind === "resp") y = mid + Math.sin(x * 0.04 + t * 1.2) * h * 0.32;
      else if (kind === "mot") y = mid + (Math.sin(x * 0.5 + t * 3) * 0.4 + Math.sin(x * 0.2 + t) * 0.3) * h * 0.28;
      else y = mid - (x / w) * h * 0.14;
      x === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    }
    c.strokeStyle = hexA(col, 0.9); c.lineWidth = 1.4; c.stroke();
  };
}

export function campaignDraw(c, w, h, t) {
  const pal = [C.neural, C.physio, C.metabolic, C.clinical, C.molecular, C.acc];
  c.clearRect(0, 0, w, h);
  for (let ch = 0; ch < 6; ch++) {
    const mid = h * (0.15 + ch * 0.14); c.beginPath();
    for (let x = 0; x <= w; x += 3) { const y = mid + (Math.sin(x * 0.01 + t * 0.6 + ch) * 0.5 + Math.sin(x * 0.04 + t * 1.1 + ch) * 0.3) * h * 0.05; x === 0 ? c.moveTo(x, y) : c.lineTo(x, y); }
    c.strokeStyle = hexA(pal[ch], 0.22); c.lineWidth = 1.2; c.stroke();
  }
}
