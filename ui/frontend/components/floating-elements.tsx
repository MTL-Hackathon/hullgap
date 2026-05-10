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
  onComplete?: () => void;
}

export function FloatingElements({ assembled, onLanded, onComplete }: Props) {
  const [dims, setDims] = useState<{ vw: number; vh: number } | null>(null);
  const [visible, setVisible] = useState(true);

  const phaseRef = useRef<"float" | "fly" | "fade" | "done">("float");
  const elemRefs = useRef<(HTMLDivElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const posRef = useRef<{ x: number; y: number; s: number; o: number }[]>([]);
  const posInitRef = useRef(false);
  const rafRef = useRef(0);
  const onLandedRef = useRef(onLanded);
  onLandedRef.current = onLanded;
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

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
        duration: 5 + srand(i * 5 + 6) * 7,
        animOffset: srand(i * 11 + 8) * 10,
      })),
    [],
  );

  const tileSize = dims ? Math.max(28, Math.min(42, dims.vw / 30)) : 36;

  // One-time position init
  useEffect(() => {
    if (!dims || posInitRef.current) return;
    posInitRef.current = true;
    posRef.current = ELS.map((_, i) => ({
      x: randoms[i].xPct * (dims.vw - tileSize),
      y: randoms[i].yPct * (dims.vh - tileSize),
      s: randoms[i].scale,
      o: randoms[i].opacity,
    }));
  }, [dims, randoms, tileSize]);

  // Fly animation — reads the real canvas position each frame so it tracks scroll
  useEffect(() => {
    if (!assembled || phaseRef.current !== "float" || !dims) return;
    phaseRef.current = "fly";

    // Kill CSS float animations immediately
    elemRefs.current.forEach((el) => {
      if (!el) return;
      const inner = el.firstElementChild as HTMLElement | null;
      if (inner) inner.style.animation = "none";
    });

    const LERP = 0.09;

    const tick = () => {
      const canvasEl = document.querySelector(
        "#periodic-table canvas",
      ) as HTMLCanvasElement | null;
      if (!canvasEl) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const cRect = canvasEl.getBoundingClientRect();
      const cs = Math.floor((cRect.width - PAD * 2) / 18);
      const step = cs + GAP;
      const targetScale = cs / tileSize;

      let allDone = true;

      ELS.forEach(([sym, , col, row], i) => {
        const el = elemRefs.current[i];
        const pos = posRef.current[i];
        if (!el || !pos) return;

        const tx = cRect.left + PAD + col * step;
        const ty = cRect.top + gridY(row, step);
        const tOpa = RELEVANT.has(sym) ? 1 : INACTIVE_ALPHA;

        const dx = tx - pos.x;
        const dy = ty - pos.y;

        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
          pos.x += dx * LERP;
          pos.y += dy * LERP;
          pos.s += (targetScale - pos.s) * LERP;
          pos.o += (tOpa - pos.o) * 0.14;
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

      if (allDone) {
        phaseRef.current = "fade";
        onLandedRef.current?.();
        if (containerRef.current) {
          containerRef.current.style.transition = "opacity 0.4s ease";
          containerRef.current.style.opacity = "0";
        }
        setTimeout(() => {
          setVisible(false);
          onCompleteRef.current?.();
        }, 450);
      } else {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [assembled, dims, tileSize]);

  if (!dims || !visible) return null;

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-[15] pointer-events-none overflow-hidden"
    >
      <style>{`
        @keyframes el-float {
          0%, 100% { transform: translate(0, 0); }
          25%  { transform: translate(7px, -11px); }
          50%  { transform: translate(-5px, 9px); }
          75%  { transform: translate(9px, 4px); }
        }
      `}</style>

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
              transform: `translate(${fx}px, ${fy}px) scale(${r.scale})`,
              opacity: r.opacity,
            }}
          >
            <div
              className="w-full h-full"
              style={{
                animation: `el-float ${r.duration}s ease-in-out infinite ${-r.animOffset}s`,
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
          </div>
        );
      })}
    </div>
  );
}
