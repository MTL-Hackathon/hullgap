"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const ELEMENTS = [
  { s: "Mg", n: "Magnesium",  en: 1.31, group: "alkaline",  col: 1,  row: 2 },
  { s: "N",  n: "Nitrogen",   en: 3.04, group: "nonmetal",  col: 14, row: 1 },
  { s: "Al", n: "Aluminium",  en: 1.61, group: "post-tm",   col: 12, row: 2 },
  { s: "Si", n: "Silicon",    en: 1.90, group: "metalloid", col: 13, row: 2 },
  { s: "P",  n: "Phosphorus", en: 2.19, group: "nonmetal",  col: 14, row: 2 },
  { s: "S",  n: "Sulfur",     en: 2.58, group: "nonmetal",  col: 15, row: 2 },
  { s: "Ti", n: "Titanium",   en: 1.54, group: "tm",        col: 3,  row: 3 },
  { s: "V",  n: "Vanadium",   en: 1.63, group: "tm",        col: 4,  row: 3 },
  { s: "Cr", n: "Chromium",   en: 1.66, group: "tm",        col: 5,  row: 3 },
  { s: "Mn", n: "Manganese",  en: 1.55, group: "tm",        col: 6,  row: 3 },
  { s: "Fe", n: "Iron",       en: 1.83, group: "tm",        col: 7,  row: 3 },
  { s: "Co", n: "Cobalt",     en: 1.88, group: "tm",        col: 8,  row: 3 },
  { s: "Ni", n: "Nickel",     en: 1.91, group: "tm",        col: 9,  row: 3 },
  { s: "Cu", n: "Copper",     en: 1.90, group: "tm",        col: 10, row: 3 },
  { s: "Ga", n: "Gallium",    en: 1.81, group: "post-tm",   col: 12, row: 3 },
  { s: "Ge", n: "Germanium",  en: 2.01, group: "metalloid", col: 13, row: 3 },
  { s: "As", n: "Arsenic",    en: 2.18, group: "metalloid", col: 14, row: 3 },
  { s: "Se", n: "Selenium",   en: 2.55, group: "nonmetal",  col: 15, row: 3 },
  { s: "Mo", n: "Molybdenum", en: 2.16, group: "tm",        col: 5,  row: 4 },
  { s: "Pd", n: "Palladium",  en: 2.20, group: "tm",        col: 9,  row: 4 },
  { s: "Ag", n: "Silver",     en: 1.93, group: "tm",        col: 10, row: 4 },
  { s: "In", n: "Indium",     en: 1.78, group: "post-tm",   col: 12, row: 4 },
  { s: "Sn", n: "Tin",        en: 1.96, group: "post-tm",   col: 13, row: 4 },
  { s: "Sb", n: "Antimony",   en: 2.05, group: "metalloid", col: 14, row: 4 },
  { s: "Te", n: "Tellurium",  en: 2.10, group: "metalloid", col: 15, row: 4 },
  { s: "W",  n: "Tungsten",   en: 2.36, group: "tm",        col: 5,  row: 5 },
  { s: "Pt", n: "Platinum",   en: 2.28, group: "tm",        col: 9,  row: 5 },
  { s: "Au", n: "Gold",       en: 2.54, group: "tm",        col: 10, row: 5 },
  { s: "Pb", n: "Lead",       en: 2.33, group: "post-tm",   col: 13, row: 5 },
  { s: "Bi", n: "Bismuth",    en: 2.02, group: "post-tm",   col: 14, row: 5 },
] as const;

type Group = "alkaline" | "tm" | "post-tm" | "metalloid" | "nonmetal";
type Vec2  = { x: number; y: number };

const GROUP_COLORS: Record<Group, { fill: string; stroke: string; text: string }> = {
  alkaline:  { fill: "#fff7ed", stroke: "#f97316", text: "#c2410c" },
  tm:        { fill: "#eff6ff", stroke: "#289ff0", text: "#1a7fd4" },
  "post-tm": { fill: "#f5f3ff", stroke: "#8b5cf6", text: "#6d28d9" },
  metalloid: { fill: "#f0fdf4", stroke: "#10b981", text: "#047857" },
  nonmetal:  { fill: "#fdf4ff", stroke: "#d946ef", text: "#a21caf" },
};

const GROUP_LABELS: Record<Group, string> = {
  alkaline:  "Alkaline earth",
  tm:        "Transition metal",
  "post-tm": "Post-transition metal",
  metalloid: "Metalloid / semiconductor",
  nonmetal:  "Nonmetal / pnictide",
};

// Angular sector (radians) each group is centred on in scatter mode
const GROUP_ANGLE: Record<string, number> = {
  nonmetal:  (-5 * Math.PI) / 6, // upper-left
  metalloid: (-1 * Math.PI) / 6, // upper-right
  "post-tm":  (1 * Math.PI) / 6, // lower-right
  tm:         Math.PI,            // left  (spread wide for 14 elements)
  alkaline:   (5 * Math.PI) / 6, // lower-left
};
const GROUP_SPREAD: Record<string, number> = {
  nonmetal:  0.50,
  metalloid: 0.55,
  "post-tm": 0.70,
  tm:        2.20, // ~126 ° arc for the large TM family
  alkaline:  0.15,
};

const N_COLS = 18;
const MIN_ROW = 1;
const N_ROWS  = 5;
const PAD     = 20;
const GAP     = 3;
const SCATTER_H = 500; // canvas height in scatter mode
const EN_SCALE  = 2.0; // ΔEN 2.0 → max radius
const LERP      = 0.12;

function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y,   x + w, y + r,     r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x,      y + h, x,         y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x,      y,     x + r,     y,         r);
  ctx.closePath();
}

/** Compute scatter targets: radius ∝ |ΔEN|, angle by chemical group */
function computeScatter(
  anchorIdx: number,
  cx: number, cy: number,
  cs: number, w: number, h: number,
): Vec2[] {
  const anchor = ELEMENTS[anchorIdx];
  const maxR   = Math.min(w / 2 - cs - 8, 220);
  const minR   = Math.max(cs * 1.5, 72);

  const byGroup: Record<string, number[]> = {};
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx) return;
    if (!byGroup[el.group]) byGroup[el.group] = [];
    byGroup[el.group].push(i);
  });

  const pts: Vec2[] = ELEMENTS.map(() => ({ x: cx, y: cy }));

  for (const [group, idxs] of Object.entries(byGroup)) {
    const base   = GROUP_ANGLE[group]  ?? 0;
    const spread = GROUP_SPREAD[group] ?? 0.5;
    idxs.forEach((idx, k) => {
      const dEN = Math.abs(anchor.en - ELEMENTS[idx].en);
      const r   = minR + Math.min(dEN / EN_SCALE, 1) * (maxR - minR);
      const off = idxs.length === 1 ? 0 : spread * (k / (idxs.length - 1) - 0.5);
      const a   = base + off;
      pts[idx]  = { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r };
    });
  }

  // 2-D collision separation (preserves rough radial distance, separates overlaps)
  const minD = cs + 6;
  for (let iter = 0; iter < 120; iter++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++) {
      if (i === anchorIdx) continue;
      for (let j = i + 1; j < pts.length; j++) {
        if (j === anchorIdx) continue;
        const dx = pts[j].x - pts[i].x;
        const dy = pts[j].y - pts[i].y;
        const d  = Math.hypot(dx, dy);
        if (d < minD && d > 0.001) {
          const push = (minD - d) / 2 + 0.5;
          const nx = dx / d, ny = dy / d;
          pts[i].x -= nx * push; pts[i].y -= ny * push;
          pts[j].x += nx * push; pts[j].y += ny * push;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  // Clamp to canvas bounds
  const m = cs / 2 + 4;
  pts.forEach((p, i) => {
    if (i === anchorIdx) return;
    p.x = Math.max(m, Math.min(w - m, p.x));
    p.y = Math.max(m, Math.min(h - m, p.y));
  });

  return pts;
}

export function ElementMap() {
  const wrapRef   = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // All animation state lives in refs to avoid triggering re-renders per frame
  const posRef  = useRef<Vec2[]>([]);   // current centres
  const tgtRef  = useRef<Vec2[]>([]);   // target centres
  const cellRef = useRef(0);
  const cwRef   = useRef(0);
  const chRef   = useRef(0);
  const rafRef  = useRef(0);
  const selRef  = useRef<number | null>(null);

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  // ── helpers ──────────────────────────────────────────────────────────────

  const resizeCanvas = useCallback((w: number, h: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";
    canvas.getContext("2d")?.scale(dpr, dpr);
    cwRef.current = w;
    chRef.current = h;
  }, []);

  const gridTargets = useCallback((w: number, cs: number): Vec2[] => {
    const step = cs + GAP;
    return Array.from(ELEMENTS, el => ({
      x: PAD + el.col * step + cs / 2,
      y: PAD + (el.row - MIN_ROW) * step + cs / 2,
    }));
  }, []);

  // ── draw ─────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w   = cwRef.current;
    const h   = chRef.current;
    const cs  = cellRef.current;
    const sel = selRef.current;
    ctx.clearRect(0, 0, w, h);

    ELEMENTS.forEach((el, i) => {
      const { x: cx, y: cy } = posRef.current[i] ?? { x: 0, y: 0 };
      const bx = cx - cs / 2;
      const by = cy - cs / 2;

      if (i === sel) {
        // Anchor: solid accent fill, white text
        drawRoundRect(ctx, bx, by, cs, cs, 6);
        ctx.fillStyle = "#289ff0";
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.font = `700 ${Math.round(cs * 0.34)}px ui-sans-serif,system-ui,sans-serif`;
        ctx.textAlign    = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(el.s, cx, cy);
      } else {
        const c = GROUP_COLORS[el.group as Group];
        drawRoundRect(ctx, bx, by, cs, cs, 6);
        ctx.fillStyle   = c.fill;
        ctx.fill();
        ctx.strokeStyle = c.stroke;
        ctx.lineWidth   = 1.5;
        ctx.stroke();
        ctx.fillStyle   = c.text;
        ctx.font = `600 ${Math.max(9, Math.round(cs * 0.31))}px ui-sans-serif,system-ui,sans-serif`;
        ctx.textAlign    = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(el.s, cx, cy);
      }
    });
  }, []);

  // ── animation loop ────────────────────────────────────────────────────────

  const tick = useCallback(() => {
    let done = true;
    posRef.current.forEach((p, i) => {
      const t = tgtRef.current[i];
      if (!t) return;
      const dx = t.x - p.x, dy = t.y - p.y;
      if (Math.abs(dx) > 0.4 || Math.abs(dy) > 0.4) {
        p.x += dx * LERP; p.y += dy * LERP;
        done = false;
      } else {
        p.x = t.x; p.y = t.y;
      }
    });
    draw();
    if (!done) rafRef.current = requestAnimationFrame(tick);
  }, [draw]);

  const startAnim = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  // ── layout transitions ────────────────────────────────────────────────────

  const goGrid = useCallback((w: number) => {
    const cs = Math.floor((w - PAD * 2) / N_COLS);
    const h  = N_ROWS * (cs + GAP) + PAD * 2 - GAP;
    cellRef.current = cs;
    resizeCanvas(w, h);
    tgtRef.current = gridTargets(w, cs);
  }, [resizeCanvas, gridTargets]);

  const goScatter = useCallback((w: number, anchorIdx: number) => {
    resizeCanvas(w, SCATTER_H);
    tgtRef.current = computeScatter(anchorIdx, w / 2, SCATTER_H / 2, cellRef.current, w, SCATTER_H);
  }, [resizeCanvas]);

  // ── event handlers ────────────────────────────────────────────────────────

  const onCanvasClick = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    const wrap   = wrapRef.current;
    if (!canvas || !wrap) return;
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const half = cellRef.current / 2 + 3;

    const hit = posRef.current.findIndex(
      p => Math.abs(mx - p.x) <= half && Math.abs(my - p.y) <= half,
    );
    if (hit === -1) return;

    const w = wrap.clientWidth;
    if (hit === selRef.current) {
      selRef.current = null;
      setSelectedIdx(null);
      goGrid(w);
    } else {
      selRef.current = hit;
      setSelectedIdx(hit);
      goScatter(w, hit);
    }
    startAnim();
  }, [goGrid, goScatter, startAnim]);

  const onMouseMove = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const half = cellRef.current / 2 + 3;
    const hit  = posRef.current.some(p => Math.abs(mx - p.x) <= half && Math.abs(my - p.y) <= half);
    canvas.style.cursor = hit ? "pointer" : "default";
  }, []);

  const onResize = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const w = wrap.clientWidth;
    if (w === 0) return;
    if (selRef.current !== null) goScatter(w, selRef.current);
    else goGrid(w);
    // Snap to new positions on resize (no animation)
    posRef.current = tgtRef.current.map(p => ({ ...p }));
    draw();
  }, [goGrid, goScatter, draw]);

  // ── mount ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    const wrap   = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const w  = wrap.clientWidth;
    const cs = Math.floor((w - PAD * 2) / N_COLS);
    cellRef.current = cs;
    const h  = N_ROWS * (cs + GAP) + PAD * 2 - GAP;
    resizeCanvas(w, h);

    const tgts = gridTargets(w, cs);
    tgtRef.current = tgts;
    posRef.current = tgts.map(p => ({ ...p }));
    draw();

    canvas.addEventListener("click",     onCanvasClick);
    canvas.addEventListener("mousemove", onMouseMove);
    const ro = new ResizeObserver(onResize);
    ro.observe(wrap);

    return () => {
      canvas.removeEventListener("click",     onCanvasClick);
      canvas.removeEventListener("mousemove", onMouseMove);
      ro.disconnect();
      cancelAnimationFrame(rafRef.current);
    };
  }, [resizeCanvas, gridTargets, draw, onCanvasClick, onMouseMove, onResize]);

  // ── render ────────────────────────────────────────────────────────────────

  const selectedEl = selectedIdx !== null ? ELEMENTS[selectedIdx] : null;

  return (
    <section className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
          Element Space
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          {selectedEl
            ? `${selectedEl.n} anchored — distance from centre encodes |ΔEN|. Click ${selectedEl.s} again to reset.`
            : "Click any element to anchor it at the centre and explore binary compound space."}
        </p>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-[var(--shadow)]">
        <div ref={wrapRef}>
          <canvas ref={canvasRef} style={{ display: "block" }} />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1">
        {(Object.keys(GROUP_COLORS) as Group[]).map(group => {
          const c = GROUP_COLORS[group];
          return (
            <div key={group} className="flex items-center gap-1.5 text-xs text-slate-500">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: c.fill, border: `1.5px solid ${c.stroke}` }}
              />
              {GROUP_LABELS[group]}
            </div>
          );
        })}
        {selectedEl && (
          <span className="ml-auto text-xs text-[var(--muted)]">
            Distance&nbsp;∝&nbsp;|ΔEN| from {selectedEl.s}&nbsp;(EN&nbsp;=&nbsp;{selectedEl.en})
          </span>
        )}
      </div>
    </section>
  );
}
