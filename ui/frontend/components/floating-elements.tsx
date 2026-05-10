"use client";

import { useMemo, useEffect, useState, useRef } from "react";

type Cat =
  | "alkali" | "alkaline" | "tm" | "post-tm"
  | "metalloid" | "nonmetal" | "noble"
  | "lanthanide" | "actinide";

const CC: Record<Cat, { bg: string; bd: string; tx: string }> = {
  alkali:     { bg: "#fff7ed", bd: "#fb923c", tx: "#c2410c" },
  alkaline:   { bg: "#fffbeb", bd: "#fbbf24", tx: "#b45309" },
  tm:         { bg: "#eff6ff", bd: "#60a5fa", tx: "#1d4ed8" },
  "post-tm":  { bg: "#f5f3ff", bd: "#a78bfa", tx: "#6d28d9" },
  metalloid:  { bg: "#ecfdf5", bd: "#34d399", tx: "#047857" },
  nonmetal:   { bg: "#fdf4ff", bd: "#e879f9", tx: "#a21caf" },
  noble:      { bg: "#f1f5f9", bd: "#cbd5e1", tx: "#94a3b8" },
  lanthanide: { bg: "#f0fdfa", bd: "#2dd4bf", tx: "#0f766e" },
  actinide:   { bg: "#fef2f2", bd: "#fca5a5", tx: "#b91c1c" },
};

// [symbol, category, col(0-17), row(0-9)]
const ELS: [string, Cat, number, number][] = [
  ["H","nonmetal",0,0],["He","noble",17,0],
  ["Li","alkali",0,1],["Be","alkaline",1,1],["B","metalloid",12,1],["C","nonmetal",13,1],
  ["N","nonmetal",14,1],["O","nonmetal",15,1],["F","nonmetal",16,1],["Ne","noble",17,1],
  ["Na","alkali",0,2],["Mg","alkaline",1,2],["Al","post-tm",12,2],["Si","metalloid",13,2],
  ["P","nonmetal",14,2],["S","nonmetal",15,2],["Cl","nonmetal",16,2],["Ar","noble",17,2],
  ["K","alkali",0,3],["Ca","alkaline",1,3],["Sc","tm",2,3],["Ti","tm",3,3],
  ["V","tm",4,3],["Cr","tm",5,3],["Mn","tm",6,3],["Fe","tm",7,3],
  ["Co","tm",8,3],["Ni","tm",9,3],["Cu","tm",10,3],["Zn","tm",11,3],
  ["Ga","post-tm",12,3],["Ge","metalloid",13,3],["As","metalloid",14,3],["Se","nonmetal",15,3],
  ["Br","nonmetal",16,3],["Kr","noble",17,3],
  ["Rb","alkali",0,4],["Sr","alkaline",1,4],["Y","tm",2,4],["Zr","tm",3,4],
  ["Nb","tm",4,4],["Mo","tm",5,4],["Tc","tm",6,4],["Ru","tm",7,4],
  ["Rh","tm",8,4],["Pd","tm",9,4],["Ag","tm",10,4],["Cd","tm",11,4],
  ["In","post-tm",12,4],["Sn","post-tm",13,4],["Sb","metalloid",14,4],["Te","metalloid",15,4],
  ["I","nonmetal",16,4],["Xe","noble",17,4],
  ["Cs","alkali",0,5],["Ba","alkaline",1,5],
  ["Hf","tm",3,5],["Ta","tm",4,5],["W","tm",5,5],["Re","tm",6,5],
  ["Os","tm",7,5],["Ir","tm",8,5],["Pt","tm",9,5],["Au","tm",10,5],
  ["Hg","tm",11,5],["Tl","post-tm",12,5],["Pb","post-tm",13,5],["Bi","post-tm",14,5],
  ["Po","metalloid",15,5],["At","nonmetal",16,5],["Rn","noble",17,5],
  ["Fr","alkali",0,6],["Ra","alkaline",1,6],
  ["Rf","tm",3,6],["Db","tm",4,6],["Sg","tm",5,6],["Bh","tm",6,6],
  ["Hs","tm",7,6],["Mt","tm",8,6],["Ds","tm",9,6],["Rg","tm",10,6],
  ["Cn","tm",11,6],["Nh","post-tm",12,6],["Fl","post-tm",13,6],["Mc","post-tm",14,6],
  ["Lv","post-tm",15,6],["Ts","nonmetal",16,6],["Og","noble",17,6],
  ["La","lanthanide",2,8],["Ce","lanthanide",3,8],["Pr","lanthanide",4,8],["Nd","lanthanide",5,8],
  ["Pm","lanthanide",6,8],["Sm","lanthanide",7,8],["Eu","lanthanide",8,8],["Gd","lanthanide",9,8],
  ["Tb","lanthanide",10,8],["Dy","lanthanide",11,8],["Ho","lanthanide",12,8],["Er","lanthanide",13,8],
  ["Tm","lanthanide",14,8],["Yb","lanthanide",15,8],["Lu","lanthanide",16,8],
  ["Ac","actinide",2,9],["Th","actinide",3,9],["Pa","actinide",4,9],["U","actinide",5,9],
  ["Np","actinide",6,9],["Pu","actinide",7,9],["Am","actinide",8,9],["Cm","actinide",9,9],
  ["Bk","actinide",10,9],["Cf","actinide",11,9],["Es","actinide",12,9],["Fm","actinide",13,9],
  ["Md","actinide",14,9],["No","actinide",15,9],["Lr","actinide",16,9],
];

const RELEVANT = new Set([
  "H","B","C","N","O","F",
  "Mg","Al","Si","P","S","Cl",
  "Ti","Cr","Co","Ni","Cu","Zn",
  "Ga","Ge","As","Se",
  "Sr","Y","Zr","Nb","Mo","Ru","Pd","Ag",
  "In","Sn","Sb","Te",
  "Ba","La",
  "Hf","Ta","W","Ir","Pt","Au","Pb","Bi",
]);

const INACTIVE_ALPHA = 0.15;
const PAD = 16;
const GAP = 2;
const CONNECTION_DIST = 150;

function srand(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}

function gridY(row: number, step: number): number {
  if (row <= 6) return PAD + row * step;
  return PAD + (row - 0.5) * step;
}

interface Props {
  assembled: boolean;
  onLanded?: () => void;
  onDisassembled?: () => void;
}

export function FloatingElements({ assembled, onLanded, onDisassembled }: Props) {
  const [dims, setDims] = useState<{ vw: number; vh: number } | null>(null);
  const [fading, setFading] = useState(false);
  const [posReady, setPosReady] = useState(false);
  const [linesVisible, setLinesVisible] = useState(true);
  const [floatEpoch, setFloatEpoch] = useState(0);

  const phaseRef = useRef<"float" | "fly" | "done">("float");
  const elemRefs = useRef<(HTMLDivElement | null)[]>([]);
  const posRef = useRef<{ x: number; y: number; s: number; o: number }[]>([]);
  const velRef = useRef<{ vx: number; vy: number }[]>([]);
  const posInitRef = useRef(false);
  const floatRafRef = useRef(0);
  const flyRafRef = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const lineCanvasRef = useRef<HTMLCanvasElement>(null);
  const prevAssembledRef = useRef(assembled);
  const onLandedRef = useRef(onLanded);
  onLandedRef.current = onLanded;
  const onDisassembledRef = useRef(onDisassembled);
  onDisassembledRef.current = onDisassembled;

  useEffect(() => {
    const update = () => setDims({ vw: window.innerWidth, vh: window.innerHeight });
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const randoms = useMemo(
    () =>
      ELS.map((_, i) => ({
        xPct: srand(i * 3 + 1),
        yPct: srand(i * 3 + 2),
        scale: 0.35 + srand(i * 3 + 3) * 0.55,
        opacity: 0.18 + srand(i * 13) * 0.42,
      })),
    [],
  );

  const tileSize = dims ? Math.max(28, Math.min(42, dims.vw / 30)) : 36;

  // Initialize positions and velocities
  useEffect(() => {
    if (!dims || posInitRef.current) return;
    posInitRef.current = true;
    posRef.current = ELS.map((_, i) => ({
      x: randoms[i].xPct * (dims.vw - tileSize),
      y: randoms[i].yPct * (dims.vh - tileSize),
      s: randoms[i].scale,
      o: randoms[i].opacity,
    }));
    velRef.current = ELS.map((_, i) => {
      const speed = 0.3 + srand(i * 7 + 17) * 0.4;
      const angle = srand(i * 7 + 11) * Math.PI * 2;
      return { vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed };
    });
    setPosReady(true);
  }, [dims, randoms, tileSize]);

  // Size the connection-line canvas
  useEffect(() => {
    if (!dims) return;
    const lc = lineCanvasRef.current;
    if (!lc) return;
    const dpr = window.devicePixelRatio || 1;
    lc.width = Math.round(dims.vw * dpr);
    lc.height = Math.round(dims.vh * dpr);
    lc.style.width = dims.vw + "px";
    lc.style.height = dims.vh + "px";
    const ctx = lc.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, [dims]);

  // Screensaver drift + proximity connection lines
  useEffect(() => {
    if (!dims || !posReady || phaseRef.current !== "float") return;

    const tick = () => {
      if (phaseRef.current !== "float") return;

      const pos = posRef.current;
      const vel = velRef.current;
      const maxX = dims.vw - tileSize;
      const maxY = dims.vh - tileSize;

      for (let i = 0; i < pos.length; i++) {
        const p = pos[i];
        const v = vel[i];
        if (!v) continue;

        p.x += v.vx;
        p.y += v.vy;

        if (p.x <= 0)    { p.x = 0;    v.vx = Math.abs(v.vx); }
        else if (p.x >= maxX) { p.x = maxX; v.vx = -Math.abs(v.vx); }
        if (p.y <= 0)    { p.y = 0;    v.vy = Math.abs(v.vy); }
        else if (p.y >= maxY) { p.y = maxY; v.vy = -Math.abs(v.vy); }

        const el = elemRefs.current[i];
        if (el) {
          el.style.transform = `translate(${p.x}px, ${p.y}px) scale(${p.s})`;
        }
      }

      // Draw proximity connection lines
      const lc = lineCanvasRef.current;
      if (lc) {
        const ctx = lc.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, dims.vw, dims.vh);
          ctx.lineWidth = 2;

          for (let i = 0; i < pos.length; i++) {
            const cxi = pos[i].x + tileSize * pos[i].s * 0.5;
            const cyi = pos[i].y + tileSize * pos[i].s * 0.5;
            for (let j = i + 1; j < pos.length; j++) {
              const cxj = pos[j].x + tileSize * pos[j].s * 0.5;
              const cyj = pos[j].y + tileSize * pos[j].s * 0.5;
              const dist = Math.hypot(cxj - cxi, cyj - cyi);
              if (dist < CONNECTION_DIST) {
                const alpha = (1 - dist / CONNECTION_DIST) * 0.18;
                ctx.strokeStyle = `rgba(148,163,184,${alpha})`;
                ctx.beginPath();
                ctx.moveTo(cxi, cyi);
                ctx.lineTo(cxj, cyj);
                ctx.stroke();
              }
            }
          }
        }
      }

      floatRafRef.current = requestAnimationFrame(tick);
    };

    floatRafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(floatRafRef.current);
  }, [dims, tileSize, posReady, floatEpoch]);

  // Disassembly: when assembled goes true → false, restart floating
  useEffect(() => {
    const wasAssembled = prevAssembledRef.current;
    prevAssembledRef.current = assembled;

    if (wasAssembled && !assembled && phaseRef.current !== "float") {
      cancelAnimationFrame(flyRafRef.current);
      phaseRef.current = "float";
      setFading(false);
      setLinesVisible(true);

      if (dims) {
        const seed = Date.now() % 100000;
        posRef.current = ELS.map((_, i) => ({
          x: srand(i * 3 + 1 + seed) * (dims.vw - tileSize),
          y: srand(i * 3 + 2 + seed) * (dims.vh - tileSize),
          s: randoms[i].scale,
          o: randoms[i].opacity,
        }));
        velRef.current = ELS.map((_, i) => {
          const speed = 0.3 + srand(i * 7 + 17 + seed) * 0.4;
          const angle = srand(i * 7 + 11 + seed) * Math.PI * 2;
          return { vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed };
        });
        elemRefs.current.forEach((el, i) => {
          if (!el) return;
          const p = posRef.current[i];
          if (!p) return;
          el.style.transform = `translate(${p.x}px, ${p.y}px) scale(${p.s})`;
          el.style.opacity = String(p.o);
        });
      }

      onDisassembledRef.current?.();
      setFloatEpoch((e) => e + 1);
    }
  }, [assembled, dims, tileSize, randoms]);

  // Fly-to-grid animation when assembled
  useEffect(() => {
    if (!assembled || phaseRef.current !== "float" || !dims) return;
    phaseRef.current = "fly";
    cancelAnimationFrame(floatRafRef.current);
    setLinesVisible(false);

    const LERP_MIN = 0.045;
    const LERP_MAX = 0.25;
    const OPA_LERP = 0.1;
    const SNAP = 0.5;
    let startDists: number[] | null = null;
    let landedFired = false;

    const tick = () => {
      const canvasEl = document.querySelector(
        "#periodic-table canvas",
      ) as HTMLCanvasElement | null;
      if (!canvasEl) {
        flyRafRef.current = requestAnimationFrame(tick);
        return;
      }

      const cRect = canvasEl.getBoundingClientRect();
      const cs = Math.floor((cRect.width - PAD * 2) / 18);
      const step = cs + GAP;
      const targetScale = cs / tileSize;

      if (!startDists) {
        startDists = ELS.map(([, , col, row], i) => {
          const pos = posRef.current[i];
          if (!pos) return 1;
          const tx = cRect.left + PAD + col * step;
          const ty = cRect.top + gridY(row, step);
          return Math.hypot(tx - pos.x, ty - pos.y) || 1;
        });
      }

      let allDone = true;
      let maxRemaining = 0;

      ELS.forEach(([sym, , col, row], i) => {
        const el = elemRefs.current[i];
        const pos = posRef.current[i];
        if (!el || !pos) return;

        const tx = cRect.left + PAD + col * step;
        const ty = cRect.top + gridY(row, step);
        const tOpa = RELEVANT.has(sym) ? 1 : INACTIVE_ALPHA;

        const dx = tx - pos.x;
        const dy = ty - pos.y;
        const dist = Math.hypot(dx, dy);

        const ratio = dist / (startDists![i] || 1);
        if (ratio > maxRemaining) maxRemaining = ratio;

        if (dist > SNAP) {
          const progress = 1 - ratio;
          const lerp = LERP_MIN + (LERP_MAX - LERP_MIN) * progress * progress;
          pos.x += dx * lerp;
          pos.y += dy * lerp;
          pos.s += (targetScale - pos.s) * lerp;
          pos.o += (tOpa - pos.o) * OPA_LERP;
          allDone = false;
        } else {
          pos.x = tx;
          pos.y = ty;
          pos.s = targetScale;
          pos.o = tOpa;
        }

        el.style.transform = `translate(${pos.x}px, ${pos.y}px) scale(${pos.s})`;
        el.style.opacity = String(Math.min(1, pos.o));
      });

      if (maxRemaining < 0.5 && !landedFired) {
        landedFired = true;
        onLandedRef.current?.();
      }

      if (allDone) {
        phaseRef.current = "done";
        if (!landedFired) onLandedRef.current?.();
        setFading(true);
        return;
      } else {
        flyRafRef.current = requestAnimationFrame(tick);
      }
    };

    flyRafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(flyRafRef.current);
  }, [assembled, dims, tileSize]);

  if (!dims) return null;

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-[15] pointer-events-none overflow-hidden"
      style={{
        opacity: fading ? 0 : 1,
        transition: "opacity 0.35s ease-out",
      }}
    >
      <canvas
        ref={lineCanvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{
          opacity: linesVisible ? 1 : 0,
          transition: "opacity 0.8s ease-out",
        }}
      />

      {ELS.map(([sym, cat], i) => {
        const r = randoms[i];
        const c = CC[cat];
        const fx = r.xPct * (dims.vw - tileSize);
        const fy = r.yPct * (dims.vh - tileSize);

        return (
          <div
            key={i}
            ref={(el) => {
              elemRefs.current[i] = el;
            }}
            className="absolute left-0 top-0 will-change-transform"
            style={{
              width: tileSize,
              height: tileSize,
              transformOrigin: "0 0",
              transform: `translate(${fx}px, ${fy}px) scale(${r.scale})`,
              opacity: r.opacity,
            }}
          >
            <div
              className="flex items-center justify-center rounded-[4px] w-full h-full select-none"
              style={{
                backgroundColor: c.bg,
                border: `1.5px solid ${c.bd}`,
                color: c.tx,
                fontSize: Math.max(7, tileSize * 0.3),
                fontWeight: 600,
                fontFamily:
                  "ui-sans-serif, system-ui, -apple-system, sans-serif",
              }}
            >
              {sym}
            </div>
          </div>
        );
      })}
    </div>
  );
}
