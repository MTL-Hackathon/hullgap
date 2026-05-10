"use client";

import { useCallback, useEffect, useRef } from "react";

// Microelectronics-relevant elements only:
// semiconductors (Si, Ge, GaAs, InP, GaN family), contacts (Co, Ni, Cu, Au, Pt),
// barriers (Ti, W, Mo, Cr, V), pnictides/chalcogenides (N, P, As, Sb, Bi, S, Se, Te)
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

const N_COLS = 18;
const MIN_ROW = 1;
const MAX_ROW = 5;
const N_ROWS = MAX_ROW - MIN_ROW + 1;
const PAD = 20;
const CELL_GAP = 3;

function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

export function ElementMap() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    if (w === 0) return;

    const cellSize = Math.floor((w - PAD * 2) / N_COLS);
    const step = cellSize + CELL_GAP;
    const h = N_ROWS * step + PAD * 2 - CELL_GAP;

    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    for (const el of ELEMENTS) {
      const x = PAD + el.col * step;
      const y = PAD + (el.row - MIN_ROW) * step;
      const colors = GROUP_COLORS[el.group as Group];

      drawRoundRect(ctx, x, y, cellSize, cellSize, 5);
      ctx.fillStyle = colors.fill;
      ctx.fill();
      ctx.strokeStyle = colors.stroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const fontSize = Math.max(9, Math.round(cellSize * 0.31));
      ctx.fillStyle = colors.text;
      ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(el.s, x + cellSize / 2, y + cellSize / 2);
    }
  }, []);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <section className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
          Element Space
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Microelectronics-relevant elements arranged by periodic table position.
          Click any element to explore its binary compound space.
        </p>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-[var(--shadow)]">
        <div ref={wrapRef}>
          <canvas ref={canvasRef} style={{ display: "block" }} />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 px-1">
        {(Object.keys(GROUP_COLORS) as Group[]).map((group) => {
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
      </div>
    </section>
  );
}
