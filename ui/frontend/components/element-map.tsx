"use client";

import { useCallback, useEffect, useRef, useState, useMemo } from "react";

// ─── Types ──────────────────────────────────────────────────────────────
type Cat =
  | "alkali" | "alkaline" | "tm" | "post-tm"
  | "metalloid" | "nonmetal" | "noble"
  | "lanthanide" | "actinide";

type Vec2 = { x: number; y: number };

interface El {
  z: number; s: string; n: string; en: number; m: number;
  cat: Cat; col: number; row: number; relevant: boolean;
  rad: number;
}

// ─── Microelectronics-relevant elements ─────────────────────────────────
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

// ─── Element data ───────────────────────────────────────────────────────
// [Z, symbol, name, EN(Pauling), mass, category, col(0-17), row(0-9)]
// rows 0-6 = periods 1-7,  row 8 = lanthanides,  row 9 = actinides
const RAW: [number,string,string,number,number,Cat,number,number][] = [
  // Period 1
  [1,"H","Hydrogen",2.20,1.01,"nonmetal",0,0],
  [2,"He","Helium",0,4.00,"noble",17,0],
  // Period 2
  [3,"Li","Lithium",0.98,6.94,"alkali",0,1],
  [4,"Be","Beryllium",1.57,9.01,"alkaline",1,1],
  [5,"B","Boron",2.04,10.81,"metalloid",12,1],
  [6,"C","Carbon",2.55,12.01,"nonmetal",13,1],
  [7,"N","Nitrogen",3.04,14.01,"nonmetal",14,1],
  [8,"O","Oxygen",3.44,16.00,"nonmetal",15,1],
  [9,"F","Fluorine",3.98,19.00,"nonmetal",16,1],
  [10,"Ne","Neon",0,20.18,"noble",17,1],
  // Period 3
  [11,"Na","Sodium",0.93,22.99,"alkali",0,2],
  [12,"Mg","Magnesium",1.31,24.31,"alkaline",1,2],
  [13,"Al","Aluminium",1.61,26.98,"post-tm",12,2],
  [14,"Si","Silicon",1.90,28.09,"metalloid",13,2],
  [15,"P","Phosphorus",2.19,30.97,"nonmetal",14,2],
  [16,"S","Sulfur",2.58,32.07,"nonmetal",15,2],
  [17,"Cl","Chlorine",3.16,35.45,"nonmetal",16,2],
  [18,"Ar","Argon",0,39.95,"noble",17,2],
  // Period 4
  [19,"K","Potassium",0.82,39.10,"alkali",0,3],
  [20,"Ca","Calcium",1.00,40.08,"alkaline",1,3],
  [21,"Sc","Scandium",1.36,44.96,"tm",2,3],
  [22,"Ti","Titanium",1.54,47.87,"tm",3,3],
  [23,"V","Vanadium",1.63,50.94,"tm",4,3],
  [24,"Cr","Chromium",1.66,52.00,"tm",5,3],
  [25,"Mn","Manganese",1.55,54.94,"tm",6,3],
  [26,"Fe","Iron",1.83,55.85,"tm",7,3],
  [27,"Co","Cobalt",1.88,58.93,"tm",8,3],
  [28,"Ni","Nickel",1.91,58.69,"tm",9,3],
  [29,"Cu","Copper",1.90,63.55,"tm",10,3],
  [30,"Zn","Zinc",1.65,65.38,"tm",11,3],
  [31,"Ga","Gallium",1.81,69.72,"post-tm",12,3],
  [32,"Ge","Germanium",2.01,72.63,"metalloid",13,3],
  [33,"As","Arsenic",2.18,74.92,"metalloid",14,3],
  [34,"Se","Selenium",2.55,78.97,"nonmetal",15,3],
  [35,"Br","Bromine",2.96,79.90,"nonmetal",16,3],
  [36,"Kr","Krypton",3.00,83.80,"noble",17,3],
  // Period 5
  [37,"Rb","Rubidium",0.82,85.47,"alkali",0,4],
  [38,"Sr","Strontium",0.95,87.62,"alkaline",1,4],
  [39,"Y","Yttrium",1.22,88.91,"tm",2,4],
  [40,"Zr","Zirconium",1.33,91.22,"tm",3,4],
  [41,"Nb","Niobium",1.60,92.91,"tm",4,4],
  [42,"Mo","Molybdenum",2.16,95.95,"tm",5,4],
  [43,"Tc","Technetium",1.90,97.91,"tm",6,4],
  [44,"Ru","Ruthenium",2.20,101.1,"tm",7,4],
  [45,"Rh","Rhodium",2.28,102.9,"tm",8,4],
  [46,"Pd","Palladium",2.20,106.4,"tm",9,4],
  [47,"Ag","Silver",1.93,107.9,"tm",10,4],
  [48,"Cd","Cadmium",1.69,112.4,"tm",11,4],
  [49,"In","Indium",1.78,114.8,"post-tm",12,4],
  [50,"Sn","Tin",1.96,118.7,"post-tm",13,4],
  [51,"Sb","Antimony",2.05,121.8,"metalloid",14,4],
  [52,"Te","Tellurium",2.10,127.6,"metalloid",15,4],
  [53,"I","Iodine",2.66,126.9,"nonmetal",16,4],
  [54,"Xe","Xenon",2.60,131.3,"noble",17,4],
  // Period 6 (col 2 empty — La in lanthanide row)
  [55,"Cs","Caesium",0.79,132.9,"alkali",0,5],
  [56,"Ba","Barium",0.89,137.3,"alkaline",1,5],
  [72,"Hf","Hafnium",1.30,178.5,"tm",3,5],
  [73,"Ta","Tantalum",1.50,181.0,"tm",4,5],
  [74,"W","Tungsten",2.36,183.8,"tm",5,5],
  [75,"Re","Rhenium",1.90,186.2,"tm",6,5],
  [76,"Os","Osmium",2.20,190.2,"tm",7,5],
  [77,"Ir","Iridium",2.20,192.2,"tm",8,5],
  [78,"Pt","Platinum",2.28,195.1,"tm",9,5],
  [79,"Au","Gold",2.54,197.0,"tm",10,5],
  [80,"Hg","Mercury",2.00,200.6,"tm",11,5],
  [81,"Tl","Thallium",1.62,204.4,"post-tm",12,5],
  [82,"Pb","Lead",1.87,207.2,"post-tm",13,5],
  [83,"Bi","Bismuth",2.02,209.0,"post-tm",14,5],
  [84,"Po","Polonium",2.00,209,"metalloid",15,5],
  [85,"At","Astatine",2.20,210,"nonmetal",16,5],
  [86,"Rn","Radon",0,222,"noble",17,5],
  // Period 7 (col 2 empty — Ac in actinide row)
  [87,"Fr","Francium",0.70,223,"alkali",0,6],
  [88,"Ra","Radium",0.90,226,"alkaline",1,6],
  [104,"Rf","Rutherfordium",0,267,"tm",3,6],
  [105,"Db","Dubnium",0,268,"tm",4,6],
  [106,"Sg","Seaborgium",0,269,"tm",5,6],
  [107,"Bh","Bohrium",0,270,"tm",6,6],
  [108,"Hs","Hassium",0,277,"tm",7,6],
  [109,"Mt","Meitnerium",0,278,"tm",8,6],
  [110,"Ds","Darmstadtium",0,281,"tm",9,6],
  [111,"Rg","Roentgenium",0,282,"tm",10,6],
  [112,"Cn","Copernicium",0,285,"tm",11,6],
  [113,"Nh","Nihonium",0,286,"post-tm",12,6],
  [114,"Fl","Flerovium",0,289,"post-tm",13,6],
  [115,"Mc","Moscovium",0,290,"post-tm",14,6],
  [116,"Lv","Livermorium",0,293,"post-tm",15,6],
  [117,"Ts","Tennessine",0,294,"nonmetal",16,6],
  [118,"Og","Oganesson",0,294,"noble",17,6],
  // Lanthanides (row 8, cols 2-16)
  [57,"La","Lanthanum",1.10,138.9,"lanthanide",2,8],
  [58,"Ce","Cerium",1.12,140.1,"lanthanide",3,8],
  [59,"Pr","Praseodymium",1.13,140.9,"lanthanide",4,8],
  [60,"Nd","Neodymium",1.14,144.2,"lanthanide",5,8],
  [61,"Pm","Promethium",1.13,145,"lanthanide",6,8],
  [62,"Sm","Samarium",1.17,150.4,"lanthanide",7,8],
  [63,"Eu","Europium",1.20,152.0,"lanthanide",8,8],
  [64,"Gd","Gadolinium",1.20,157.3,"lanthanide",9,8],
  [65,"Tb","Terbium",1.10,158.9,"lanthanide",10,8],
  [66,"Dy","Dysprosium",1.22,162.5,"lanthanide",11,8],
  [67,"Ho","Holmium",1.23,164.9,"lanthanide",12,8],
  [68,"Er","Erbium",1.24,167.3,"lanthanide",13,8],
  [69,"Tm","Thulium",1.25,168.9,"lanthanide",14,8],
  [70,"Yb","Ytterbium",1.10,173.0,"lanthanide",15,8],
  [71,"Lu","Lutetium",1.27,175.0,"lanthanide",16,8],
  // Actinides (row 9, cols 2-16)
  [89,"Ac","Actinium",1.10,227,"actinide",2,9],
  [90,"Th","Thorium",1.30,232.0,"actinide",3,9],
  [91,"Pa","Protactinium",1.50,231.0,"actinide",4,9],
  [92,"U","Uranium",1.38,238.0,"actinide",5,9],
  [93,"Np","Neptunium",1.36,237,"actinide",6,9],
  [94,"Pu","Plutonium",1.28,244,"actinide",7,9],
  [95,"Am","Americium",1.30,243,"actinide",8,9],
  [96,"Cm","Curium",1.30,247,"actinide",9,9],
  [97,"Bk","Berkelium",1.30,247,"actinide",10,9],
  [98,"Cf","Californium",1.30,251,"actinide",11,9],
  [99,"Es","Einsteinium",1.30,252,"actinide",12,9],
  [100,"Fm","Fermium",1.30,257,"actinide",13,9],
  [101,"Md","Mendelevium",1.30,258,"actinide",14,9],
  [102,"No","Nobelium",1.30,259,"actinide",15,9],
  [103,"Lr","Lawrencium",1.30,266,"actinide",16,9],
];

// Empirical atomic radii (pm)
const ATOMIC_RADIUS: Record<number, number> = {
  1:25,2:120,3:145,4:105,5:85,6:70,7:65,8:60,9:50,10:160,
  11:180,12:150,13:125,14:110,15:100,16:100,17:100,18:71,
  19:220,20:180,21:160,22:140,23:135,24:140,25:140,26:140,27:135,28:135,29:135,30:135,
  31:130,32:125,33:115,34:115,35:115,36:88,
  37:235,38:200,39:180,40:155,41:145,42:145,43:135,44:130,45:135,46:140,47:160,48:155,
  49:155,50:145,51:145,52:140,53:140,54:108,
  55:260,56:215,72:155,73:145,74:135,75:135,76:130,77:135,78:135,79:135,80:150,81:190,82:180,83:160,84:190,85:150,86:120,
  87:260,88:215,104:150,105:150,106:150,107:150,108:150,109:150,110:150,111:150,112:150,113:150,114:150,115:150,116:150,117:150,118:150,
  57:195,58:185,59:185,60:185,61:185,62:185,63:185,64:180,65:175,66:175,67:175,68:175,69:175,70:175,71:175,
  89:195,90:180,91:180,92:175,93:175,94:175,95:175,96:175,97:175,98:175,99:175,100:175,101:175,102:175,103:175,
};

const ELEMENTS: El[] = RAW.map(([z,s,n,en,m,cat,col,row]) => ({
  z, s, n, en, m, cat, col, row, relevant: RELEVANT.has(s),
  rad: ATOMIC_RADIUS[z] ?? 150,
}));

const RAD_MIN = 25;
const RAD_MAX = 260;
function scatterR(el: El, cs: number): number {
  const t = (el.rad - RAD_MIN) / (RAD_MAX - RAD_MIN);
  return (cs / 2) * (0.55 + t * 0.9);
}

// ─── Color config ───────────────────────────────────────────────────────
const CAT_COLORS: Record<Cat, { bg: string; stroke: string; text: string; fill: string }> = {
  alkali:     { bg: "#fff7ed", stroke: "#fb923c", text: "#c2410c", fill: "#f97316" },
  alkaline:   { bg: "#fffbeb", stroke: "#fbbf24", text: "#b45309", fill: "#f59e0b" },
  tm:         { bg: "#eff6ff", stroke: "#60a5fa", text: "#1d4ed8", fill: "#3b82f6" },
  "post-tm":  { bg: "#f5f3ff", stroke: "#a78bfa", text: "#6d28d9", fill: "#8b5cf6" },
  metalloid:  { bg: "#ecfdf5", stroke: "#34d399", text: "#047857", fill: "#10b981" },
  nonmetal:   { bg: "#fdf4ff", stroke: "#e879f9", text: "#a21caf", fill: "#d946ef" },
  noble:      { bg: "#f1f5f9", stroke: "#cbd5e1", text: "#94a3b8", fill: "#64748b" },
  lanthanide: { bg: "#f0fdfa", stroke: "#2dd4bf", text: "#0f766e", fill: "#14b8a6" },
  actinide:   { bg: "#fef2f2", stroke: "#fca5a5", text: "#b91c1c", fill: "#ef4444" },
};

const CAT_LABELS: Record<string, string> = {
  alkali:     "Alkali metal",
  alkaline:   "Alkaline earth",
  tm:         "Transition metal",
  "post-tm":  "Post-transition metal",
  metalloid:  "Metalloid",
  nonmetal:   "Nonmetal",
  lanthanide: "Lanthanide",
  actinide:   "Actinide",
  noble:      "Noble gas",
};

const LEGEND_GROUPS: Cat[] = [
  "alkali","alkaline","tm","post-tm","metalloid","nonmetal","lanthanide","noble",
];

// ─── Layout ─────────────────────────────────────────────────────────────
const N_COLS    = 18;
const PAD       = 16;
const GAP       = 2;
const SCATTER_H = 560;
const LERP       = 0.14;
const SHAPE_LERP = 0.22;
const INACTIVE_ALPHA = 0.15;

// ─── Scatter angle / spread per group ───────────────────────────────────
const GROUP_ANGLE: Record<string, number> = {
  nonmetal:   -Math.PI * 0.50,
  metalloid:  -Math.PI * 0.15,
  "post-tm":   Math.PI * 0.15,
  lanthanide:  Math.PI * 0.50,
  alkaline:    Math.PI * 0.82,
  alkali:     -Math.PI * 0.82,
  tm:          Math.PI,
  actinide:    Math.PI * 0.35,
  noble:      -Math.PI * 0.90,
};
const GROUP_SPREAD: Record<string, number> = {
  nonmetal:  0.70, metalloid: 0.55, "post-tm": 0.60,
  lanthanide: 1.10, alkaline: 0.40, alkali: 0.30,
  tm: 2.20, actinide: 0.50, noble: 0.30,
};

// ─── Helpers ────────────────────────────────────────────────────────────
function gridYFn(row: number, step: number): number {
  if (row <= 6) return PAD + row * step;
  return PAD + (row - 0.5) * step;
}

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

// ─── Scatter computation ────────────────────────────────────────────────
function computeScatter(
  anchorIdx: number, cx: number, cy: number,
  cs: number, w: number, h: number,
): { pts: Vec2[]; enScale: number } {
  const anchor = ELEMENTS[anchorIdx];
  const aR     = scatterR(anchor, cs);
  const maxR   = Math.min(w / 2 - cs * 0.8, h / 2 - cs * 0.8);
  const minR   = Math.max(aR + cs * 0.6, 50);

  // Compute actual max ΔEN so distance is truly proportional
  let maxDEN = 0;
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant || el.en === 0) return;
    maxDEN = Math.max(maxDEN, Math.abs(anchor.en - el.en));
  });
  if (maxDEN < 0.1) maxDEN = 1;
  const enScale = maxDEN * 1.05;

  const byGroup: Record<string, number[]> = {};
  ELEMENTS.forEach((el, i) => {
    if (i === anchorIdx || !el.relevant) return;
    (byGroup[el.cat] ??= []).push(i);
  });

  const pts: Vec2[] = ELEMENTS.map(() => ({ x: cx, y: cy }));

  for (const [group, idxs] of Object.entries(byGroup)) {
    const base   = GROUP_ANGLE[group]  ?? 0;
    const spread = GROUP_SPREAD[group] ?? 0.5;
    idxs.sort((a, b) =>
      Math.abs(anchor.en - ELEMENTS[a].en) - Math.abs(anchor.en - ELEMENTS[b].en),
    );
    idxs.forEach((idx, k) => {
      const dEN = Math.abs(anchor.en - ELEMENTS[idx].en);
      const r   = minR + Math.sqrt(Math.min(dEN / enScale, 1)) * (maxR - minR);
      const off = idxs.length === 1 ? 0 : spread * (k / (idxs.length - 1) - 0.5);
      const a   = base + off;
      pts[idx]  = { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r };
    });
  }

  // Collision resolution — anchor stays fixed, pushes others away
  for (let iter = 0; iter < 300; iter++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++) {
      if (i !== anchorIdx && !ELEMENTS[i].relevant) continue;
      const rI = scatterR(ELEMENTS[i], cs);
      for (let j = i + 1; j < pts.length; j++) {
        if (j !== anchorIdx && !ELEMENTS[j].relevant) continue;
        const rJ = scatterR(ELEMENTS[j], cs);
        const minD = rI + rJ + 8;
        const dx = pts[j].x - pts[i].x;
        const dy = pts[j].y - pts[i].y;
        const d  = Math.hypot(dx, dy);
        if (d < minD && d > 0.001) {
          const push = (minD - d) / 2 + 0.5;
          const nx = dx / d, ny = dy / d;
          if (i === anchorIdx) {
            pts[j].x += nx * push * 2; pts[j].y += ny * push * 2;
          } else if (j === anchorIdx) {
            pts[i].x -= nx * push * 2; pts[i].y -= ny * push * 2;
          } else {
            pts[i].x -= nx * push; pts[i].y -= ny * push;
            pts[j].x += nx * push; pts[j].y += ny * push;
          }
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  pts.forEach((p, i) => {
    if (i === anchorIdx || !ELEMENTS[i].relevant) return;
    const margin = scatterR(ELEMENTS[i], cs) + 6;
    p.x = Math.max(margin, Math.min(w - margin, p.x));
    p.y = Math.max(margin, Math.min(h - margin, p.y));
  });

  return { pts, enScale };
}

// ─── Component ──────────────────────────────────────────────────────────
interface ElementMapProps {
  onGenerate?: (elA: string, elB: string, n: number) => void;
  isGenerating?: boolean;
}

export function ElementMap({ onGenerate, isGenerating }: ElementMapProps = {}) {
  const wrapRef    = useRef<HTMLDivElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const compoundPanelRef = useRef<HTMLDivElement>(null);

  const posRef     = useRef<Vec2[]>([]);
  const tgtRef     = useRef<Vec2[]>([]);
  const opacityRef = useRef<number[]>([]);
  const cellRef    = useRef(0);
  const cwRef      = useRef(0);
  const chRef      = useRef(0);
  const rafRef     = useRef(0);
  const selARef    = useRef<number | null>(null);
  const selBRef    = useRef<number | null>(null);
  const hoverRef   = useRef<number | null>(null);
  const animRef    = useRef(false);
  const binaryCountsRef = useRef<Record<string, { total: number; stable: number }>>({});
  const binaryAbortRef  = useRef<AbortController | null>(null);
  const enScaleRef = useRef(2.5);
  const shapeRef   = useRef<number[]>([]);

  const [selectedA, setSelectedA] = useState<number | null>(null);
  const [selectedB, setSelectedB] = useState<number | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [nCandidates, setNCandidates] = useState(12);

  // MP phase data
  interface MpPhase {
    id: string; formula: string; spacegroup: string; crystal_system: string;
    e_above_hull: number; formation_energy: number; density: number;
    n_sites: number; volume: number; is_stable: boolean;
    a: number | null; b: number | null; c: number | null;
    alpha: number | null; beta: number | null; gamma: number | null;
  }
  const [mpPhases, setMpPhases]     = useState<MpPhase[] | null>(null);
  const [mpLoading, setMpLoading]   = useState(false);
  const [mpError, setMpError]       = useState<string | null>(null);
  const mpAbortRef = useRef<AbortController | null>(null);

  const mode = selectedA !== null && selectedB !== null
    ? "compound" as const
    : selectedA !== null ? "scatter" as const : "grid" as const;

  const stablePhases = useMemo(
    () => mpPhases?.filter(p => p.is_stable) ?? [],
    [mpPhases],
  );

  // ── canvas helpers ────────────────────────────────────────────────────

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
    return ELEMENTS.map(el => ({
      x: PAD + el.col * step + cs / 2,
      y: gridYFn(el.row, step) + cs / 2,
    }));
  }, []);

  // ── helpers for current mode ──────────────────────────────────────────

  function modeRef(): "grid" | "scatter" | "compound" {
    if (selARef.current !== null && selBRef.current !== null) return "compound";
    if (selARef.current !== null) return "scatter";
    return "grid";
  }

  function opaTarget(el: El, _i: number, m: "grid"|"scatter"|"compound"): number {
    if (m === "compound" || m === "scatter") return el.relevant ? 1.0 : 0;
    return el.relevant ? 1.0 : INACTIVE_ALPHA;
  }

  // ── draw ──────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w    = cwRef.current;
    const h    = chRef.current;
    const cs   = cellRef.current;
    const selA = selARef.current;
    const selB = selBRef.current;
    const hov  = hoverRef.current;
    const m    = modeRef();
    const font = "ui-sans-serif,system-ui,-apple-system,sans-serif";

    ctx.clearRect(0, 0, w, h);

    // ── Radial EN-compatibility background (scatter only) ──────────
    if (m === "scatter" && selA !== null) {
      const anchorPos = posRef.current[selA];
      if (anchorPos) {
        const enScale = enScaleRef.current;
        const maxR = Math.min(w / 2 - cs, h / 2 - cs);
        const aR = scatterR(ELEMENTS[selA], cs);
        const minR = Math.max(aR * 2 + 20, 80);

        // EN ring guides at meaningful intervals
        ctx.save();
        ctx.setLineDash([3, 6]);
        ctx.lineWidth = 1;

        const labelAngle = -Math.PI * 0.67;
        const ringStep = enScale > 2 ? 0.5 : enScale > 1 ? 0.25 : 0.1;
        for (let dEN = ringStep; dEN <= enScale; dEN += ringStep) {
          const r = minR + (dEN / enScale) * (maxR - minR);
          if (r > maxR + 20) break;

          ctx.strokeStyle = "rgba(209,213,219,0.35)";
          ctx.beginPath();
          ctx.arc(anchorPos.x, anchorPos.y, r, 0, Math.PI * 2);
          ctx.stroke();

          ctx.save();
          ctx.setLineDash([]);
          ctx.fillStyle = "rgba(156,163,175,0.55)";
          ctx.font = `600 10px ${font}`;
          const lx = anchorPos.x + Math.cos(labelAngle) * r;
          const ly = anchorPos.y + Math.sin(labelAngle) * r;
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(`ΔEN ${dEN.toFixed(dEN < 0.5 ? 2 : 1)}`, lx, ly - 4);
          ctx.restore();
        }
        ctx.restore();

        const anchor = ELEMENTS[selA];
        const anchorVR = scatterR(anchor, cs);
        ctx.fillStyle = "rgba(156,163,175,0.7)";
        ctx.font = `500 10px ${font}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(`EN ${anchor.en.toFixed(2)}`, anchorPos.x, anchorPos.y + anchorVR + 4);
      }
    }

    // ── Connection lines (scatter / compound, binary compound counts) ─
    if ((m === "scatter" || m === "compound") && selA !== null) {
      const aPos = posRef.current[selA];
      const counts = binaryCountsRef.current;
      if (aPos && Object.keys(counts).length > 0) {
        ctx.save();
        ctx.lineWidth = 1.4;
        ELEMENTS.forEach((el, i) => {
          if (i === selA || !el.relevant) return;
          const p = posRef.current[i];
          if (!p || opacityRef.current[i] < 0.05) return;
          const cnt = counts[el.s];
          if (!cnt || cnt.total === 0) return;

          const dx = p.x - aPos.x, dy = p.y - aPos.y;
          const len = Math.hypot(dx, dy);
          if (len < 1) return;

          const nLines = Math.min(Math.ceil(cnt.total / 2), 5);
          const nx = -dy / len, ny = dx / len;
          const spreadPx = Math.min(nLines * 2, 10);
          const aR = scatterR(ELEMENTS[selA], cs);
          const eR = scatterR(el, cs);
          const f0 = aR / len, f1 = 1 - eR / len;

          ctx.strokeStyle = "rgba(148,163,184,0.25)";

          for (let k = 0; k < nLines; k++) {
            const off = nLines === 1 ? 0 : spreadPx * (k / (nLines - 1) - 0.5);
            ctx.beginPath();
            ctx.moveTo(aPos.x + dx * f0 + nx * off, aPos.y + dy * f0 + ny * off);
            ctx.lineTo(aPos.x + dx * f1 + nx * off, aPos.y + dy * f1 + ny * off);
            ctx.stroke();
          }
        });
        ctx.restore();
      }
    }

    // Lanthanide / actinide indicators in grid mode
    if (m === "grid") {
      const step = cs + GAP;
      const indX = PAD + 2 * step + cs / 2;
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.font = `600 ${Math.max(6, Math.round(cs * 0.18))}px ${font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = CAT_COLORS.lanthanide.fill;
      ctx.fillText("57-71", indX, gridYFn(5, step) + cs / 2);
      ctx.fillStyle = CAT_COLORS.actinide.fill;
      ctx.fillText("89-103", indX, gridYFn(6, step) + cs / 2);
      ctx.restore();
    }

    // ── elements (3 passes: bg → normal → hovered) ─────────────────────
    // Unified draw: interpolates shape between square (grid) and circle
    // (scatter) using shapeRef (0 = square, 1 = circle).
    const drawEl = (i: number) => {
      const el  = ELEMENTS[i];
      const pos = posRef.current[i];
      const opa = opacityRef.current[i];
      if (!pos || opa < 0.01) return;

      const isAnchor   = i === selA;
      const isSelected = i === selB;
      const isHovered  = i === hov && el.relevant;
      const hScale     = (isHovered || isSelected) ? 1.14 : 1.0;
      const c          = CAT_COLORS[el.cat];
      const shape      = shapeRef.current[i] ?? 0;

      // Interpolate size: grid square ↔ scatter circle
      const gridSize    = cs * hScale;
      const scatterDiam = scatterR(el, cs) * 2 * hScale;
      const size        = gridSize + (scatterDiam - gridSize) * shape;
      const bRadius     = 5 + (size / 2 - 5) * shape;

      ctx.save();
      ctx.globalAlpha = opa;

      const bx = pos.x - size / 2;
      const by = pos.y - size / 2;
      drawRoundRect(ctx, bx, by, size, size, bRadius);

      const isScatter = shape > 0.5;

      if (isAnchor) {
        if (isScatter) {
          ctx.fillStyle = c.fill;
          ctx.fill();
          ctx.strokeStyle = "#3b82f6";
          ctx.lineWidth = 3;
          ctx.stroke();
        } else {
          ctx.fillStyle = "#3b82f6";
          ctx.fill();
        }
        ctx.fillStyle = "#fff";
      } else if (isSelected) {
        ctx.fillStyle = c.fill;
        ctx.fill();
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = "#fff";
      } else if (isHovered) {
        ctx.fillStyle = c.fill;
        ctx.fill();
        ctx.fillStyle = "#fff";
      } else {
        ctx.fillStyle = c.bg;
        ctx.fill();
        ctx.strokeStyle = c.stroke;
        ctx.lineWidth = el.relevant ? 1.5 : 0.8;
        ctx.stroke();
        ctx.fillStyle = c.text;
      }

      const scatterFont = Math.max(9, Math.round(scatterDiam * 0.35));
      const gridFont    = Math.max(8, Math.round(gridSize * 0.32));
      const fontSize    = Math.round(gridFont + (scatterFont - gridFont) * shape);
      ctx.font = `600 ${fontSize}px ${font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(el.s, pos.x, pos.y);

      ctx.restore();
    };

    ELEMENTS.forEach((el, i) => { if (!el.relevant && i !== hov) drawEl(i); });
    ELEMENTS.forEach((el, i) => { if (el.relevant && i !== hov) drawEl(i); });
    if (hov !== null) drawEl(hov);

    if (tooltipRef.current && hov !== null && posRef.current[hov]) {
      const p = posRef.current[hov];
      const shape = shapeRef.current[hov] ?? 0;
      const hovR = cs / 2 + (scatterR(ELEMENTS[hov], cs) - cs / 2) * shape;
      tooltipRef.current.style.transform =
        `translate(${p.x}px, ${p.y - hovR - 6}px)`;
    }
  }, []);

  // ── animation loop ────────────────────────────────────────────────────

  const tick = useCallback(() => {
    const m = modeRef();
    let done = true;

    posRef.current.forEach((p, i) => {
      const t = tgtRef.current[i];
      if (!t) return;
      const dx = t.x - p.x, dy = t.y - p.y;
      const dist = Math.hypot(dx, dy);
      if (dist > 0.3) {
        const speed = LERP + Math.min(dist / 800, 0.08);
        p.x += dx * speed;
        p.y += dy * speed;
        done = false;
      } else { p.x = t.x; p.y = t.y; }
    });

    opacityRef.current.forEach((o, i) => {
      const el = ELEMENTS[i];
      const target = opaTarget(el, i, m);
      const diff = target - o;
      if (Math.abs(diff) > 0.005) {
        const fast = (m !== "grid" && target === 0);
        opacityRef.current[i] = o + diff * (fast ? 0.32 : LERP);
        done = false;
      } else {
        opacityRef.current[i] = target;
      }
    });

    // Morph shape: 0 = square (grid), 1 = circle (scatter/compound)
    const shapeTarget = (m === "scatter" || m === "compound") ? 1 : 0;
    shapeRef.current.forEach((s, i) => {
      const diff = shapeTarget - s;
      if (Math.abs(diff) > 0.005) {
        shapeRef.current[i] = s + diff * SHAPE_LERP;
        done = false;
      } else {
        shapeRef.current[i] = shapeTarget;
      }
    });

    draw();

    if (!done) {
      animRef.current = true;
      rafRef.current = requestAnimationFrame(tick);
    } else {
      animRef.current = false;
    }
  }, [draw]);

  const startAnim = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    animRef.current = true;
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  // ── layout transitions ────────────────────────────────────────────────

  const goGrid = useCallback((w: number) => {
    const cs   = Math.floor((w - PAD * 2) / N_COLS);
    const step = cs + GAP;
    const h    = gridYFn(9, step) + cs + PAD;
    cellRef.current = cs;
    resizeCanvas(w, h);
    tgtRef.current = gridTargets(w, cs);
  }, [resizeCanvas, gridTargets]);

  const goScatter = useCallback((w: number, anchorIdx: number) => {
    resizeCanvas(w, SCATTER_H);
    const result = computeScatter(
      anchorIdx, w / 2, SCATTER_H / 2, cellRef.current, w, SCATTER_H,
    );
    tgtRef.current = result.pts;
    enScaleRef.current = result.enScale;
  }, [resizeCanvas]);

  // ── reset helper ──────────────────────────────────────────────────────

  const resetToGrid = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    selARef.current = null;
    selBRef.current = null;
    setSelectedA(null);
    setSelectedB(null);
    hoverRef.current = null;
    setHoveredIdx(null);
    setMpPhases(null);
    setMpError(null);
    mpAbortRef.current?.abort();
    binaryCountsRef.current = {};
    binaryAbortRef.current?.abort();
    goGrid(wrap.clientWidth);
    startAnim();
  }, [goGrid, startAnim]);

  // ── MP API fetch ────────────────────────────────────────────────────

  const fetchMpPhases = useCallback(async (symA: string, symB: string) => {
    mpAbortRef.current?.abort();
    const ctrl = new AbortController();
    mpAbortRef.current = ctrl;
    setMpPhases(null);
    setMpError(null);
    setMpLoading(true);
    try {
      const res = await fetch(
        `/api/mp-phases?a=${encodeURIComponent(symA)}&b=${encodeURIComponent(symB)}`,
        { signal: ctrl.signal },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
      }
      const json = await res.json() as { phases: MpPhase[] };
      if (!ctrl.signal.aborted) setMpPhases(json.phases);
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setMpError(e instanceof Error ? e.message : "Failed to fetch");
      }
    } finally {
      if (!ctrl.signal.aborted) setMpLoading(false);
    }
  }, []);

  // ── MP binary counts fetch ──────────────────────────────────────────

  const fetchBinaryCounts = useCallback(async (symbol: string) => {
    binaryAbortRef.current?.abort();
    const ctrl = new AbortController();
    binaryAbortRef.current = ctrl;
    binaryCountsRef.current = {};
    try {
      const res = await fetch(
        `/api/mp-binary-counts?el=${encodeURIComponent(symbol)}`,
        { signal: ctrl.signal },
      );
      if (res.ok) {
        const json = (await res.json()) as {
          counts: Record<string, { total: number; stable: number }>;
        };
        if (!ctrl.signal.aborted) {
          binaryCountsRef.current = json.counts;
          draw();
        }
      }
    } catch { /* ignore abort / network errors */ }
  }, [draw]);

  // ── event handlers ────────────────────────────────────────────────────

  const onCanvasClick = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    const wrap   = wrapRef.current;
    if (!canvas || !wrap) return;
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const cs   = cellRef.current;
    const m    = modeRef();
    const w    = wrap.clientWidth;

    let hitIdx = -1;
    for (let i = 0; i < ELEMENTS.length; i++) {
      if (!ELEMENTS[i].relevant) continue;
      if ((m === "scatter" || m === "compound") && opacityRef.current[i] < 0.1) continue;
      const p = posRef.current[i];
      if (!p) continue;
      if (m === "scatter" || m === "compound") {
        const hitR = scatterR(ELEMENTS[i], cs) + 3;
        if (Math.hypot(mx - p.x, my - p.y) <= hitR) { hitIdx = i; break; }
      } else {
        const half = cs / 2 + 3;
        if (Math.abs(mx - p.x) <= half && Math.abs(my - p.y) <= half) { hitIdx = i; break; }
      }
    }

    if (m === "grid") {
      if (hitIdx === -1) return;
      selARef.current = hitIdx;
      setSelectedA(hitIdx);
      goScatter(w, hitIdx);
      startAnim();
      fetchBinaryCounts(ELEMENTS[hitIdx].s);
    } else if (m === "scatter" || m === "compound") {
      if (hitIdx === selARef.current) {
        // Click anchor again — reset to grid
        selARef.current = null;
        selBRef.current = null;
        setSelectedA(null);
        setSelectedB(null);
        setMpPhases(null);
        binaryCountsRef.current = {};
        binaryAbortRef.current?.abort();
        goGrid(w);
        startAnim();
      } else if (hitIdx === -1 && m === "compound") {
        // Click empty space in compound mode — deselect B, stay in scatter
        selBRef.current = null;
        setSelectedB(null);
        setMpPhases(null);
        draw();
      } else if (hitIdx >= 0 && hitIdx !== selBRef.current) {
        // Select (or change) element B
        selBRef.current = hitIdx;
        setSelectedB(hitIdx);
        fetchMpPhases(ELEMENTS[selARef.current!].s, ELEMENTS[hitIdx].s);
        draw();
        setTimeout(() => {
          compoundPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 100);
      }
    }
  }, [goGrid, goScatter, startAnim, draw, fetchBinaryCounts, fetchMpPhases]);

  const prevHoverRef = useRef<number | null>(null);

  const onMouseMove = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const m = modeRef();

    if (animRef.current) {
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        prevHoverRef.current = null;
        setHoveredIdx(null);
      }
      canvas.style.cursor = "default";
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const cs   = cellRef.current;

    let hitIdx = -1;
    for (let i = 0; i < ELEMENTS.length; i++) {
      if (!ELEMENTS[i].relevant) continue;
      if ((m === "scatter" || m === "compound") && opacityRef.current[i] < 0.1) continue;
      const p = posRef.current[i];
      if (!p) continue;
      if (m === "scatter" || m === "compound") {
        const hitR = scatterR(ELEMENTS[i], cs) + 3;
        if (Math.hypot(mx - p.x, my - p.y) <= hitR) { hitIdx = i; break; }
      } else {
        const half = cs / 2 + 3;
        if (Math.abs(mx - p.x) <= half && Math.abs(my - p.y) <= half) { hitIdx = i; break; }
      }
    }

    const newHov = hitIdx >= 0 ? hitIdx : null;
    hoverRef.current = newHov;
    canvas.style.cursor = newHov !== null ? "pointer" : "default";

    if (newHov !== prevHoverRef.current) {
      prevHoverRef.current = newHov;
      setHoveredIdx(newHov);
    }

    draw();
  }, [draw]);

  const onMouseLeave = useCallback(() => {
    hoverRef.current = null;
    prevHoverRef.current = null;
    setHoveredIdx(null);
    draw();
  }, [draw]);

  const onResize = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const w = wrap.clientWidth;
    if (w === 0) return;
    const m = modeRef();
    if ((m === "compound" || m === "scatter") && selARef.current !== null)
      goScatter(w, selARef.current);
    else goGrid(w);
    posRef.current = tgtRef.current.map(p => ({ ...p }));
    const targetShape = (m === "scatter" || m === "compound") ? 1 : 0;
    ELEMENTS.forEach((el, i) => {
      opacityRef.current[i] = opaTarget(el, i, m);
      shapeRef.current[i] = targetShape;
    });
    draw();
  }, [goGrid, goScatter, draw]);

  // ── mount ─────────────────────────────────────────────────────────────

  useEffect(() => {
    const wrap   = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const w    = wrap.clientWidth;
    const cs   = Math.floor((w - PAD * 2) / N_COLS);
    const step = cs + GAP;
    const h    = gridYFn(9, step) + cs + PAD;
    cellRef.current = cs;
    resizeCanvas(w, h);

    const tgts = gridTargets(w, cs);
    tgtRef.current = tgts;
    posRef.current = tgts.map(p => ({ ...p }));
    opacityRef.current = ELEMENTS.map(el => el.relevant ? 1.0 : INACTIVE_ALPHA);
    shapeRef.current = ELEMENTS.map(() => 0);
    draw();

    canvas.addEventListener("click",      onCanvasClick);
    canvas.addEventListener("mousemove",  onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    const ro = new ResizeObserver(onResize);
    ro.observe(wrap);

    return () => {
      canvas.removeEventListener("click",      onCanvasClick);
      canvas.removeEventListener("mousemove",  onMouseMove);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      ro.disconnect();
      cancelAnimationFrame(rafRef.current);
    };
  }, [resizeCanvas, gridTargets, draw, onCanvasClick, onMouseMove, onMouseLeave, onResize]);

  // ── render ────────────────────────────────────────────────────────────

  const elA = selectedA !== null ? ELEMENTS[selectedA] : null;
  const elB = selectedB !== null ? ELEMENTS[selectedB] : null;
  const hoveredEl = hoveredIdx !== null ? ELEMENTS[hoveredIdx] : null;

  return (
    <section id="periodic-table" className="mx-auto max-w-6xl px-4 pb-10 pt-4 sm:px-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold tracking-[-0.02em] text-[var(--foreground)]">
          Element Space
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          {mode === "compound" && elA && elB
            ? `${elA.n}–${elB.n} selected — see details below. Click another element to change, or ${elA.s} to reset.`
            : mode === "scatter" && elA
              ? `${elA.n} selected — pick a second element to explore the binary compound, or click ${elA.s} to reset.`
              : "Highlighted elements are microelectronics-relevant. Click one to anchor it and explore electronegativity space."}
        </p>
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-[var(--shadow)]">
        <div ref={wrapRef} className="relative">
          <canvas ref={canvasRef} style={{ display: "block" }} />

          {/* Hover tooltip (grid / scatter only) */}
          <div
            ref={tooltipRef}
            className="pointer-events-none absolute top-0 left-0 z-10"
            style={{
              opacity: hoveredEl ? 1 : 0,
              transition: "opacity 120ms ease",
            }}
          >
            <div className="-translate-x-1/2 -translate-y-full pb-2">
              <div className="rounded-lg bg-gray-900/90 backdrop-blur-sm px-3 py-2 shadow-lg text-white whitespace-nowrap">
                {hoveredEl && (
                  <>
                    <div className="flex items-baseline gap-2">
                      <span className="text-base font-bold leading-tight">{hoveredEl.s}</span>
                      <span className="text-[10px] text-gray-400">{hoveredEl.z}</span>
                    </div>
                    <div className="text-[11px] text-gray-300 leading-tight">{hoveredEl.n}</div>
                    <div className="text-[10px] text-gray-400 mt-0.5 leading-tight">
                      {hoveredEl.en > 0 ? `EN ${hoveredEl.en.toFixed(2)}` : "EN —"}
                      {" · "}
                      {hoveredEl.m.toFixed(1)} u
                    </div>
                    {elA && hoveredIdx !== selectedA && hoveredEl.en > 0 && elA.en > 0 && (
                      <div className="text-[10px] text-blue-300 mt-0.5 leading-tight">
                        ΔEN {Math.abs(elA.en - hoveredEl.en).toFixed(2)}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ── Compound pair builder ──────────────────────────────────── */}
        {(mode === "scatter" || mode === "compound") && elA && (
          <div
            className={`flex items-center justify-center py-5 ${mode !== "grid" ? "border-t border-gray-100" : ""}`}
            onClick={undefined}
            style={{ cursor: undefined }}
          >
            {/* Element A */}
            <div className="flex flex-col items-center gap-1.5">
              <div
                className="w-[88px] h-[88px] rounded-full flex flex-col items-center justify-center shadow-sm"
                style={{
                  background: CAT_COLORS[elA.cat].fill,
                  border: `2.5px solid ${CAT_COLORS[elA.cat].stroke}`,
                }}
              >
                <span className="text-white text-xl font-bold leading-tight">{elA.s}</span>
                <span className="text-white/70 text-[10px] leading-tight">{elA.n}</span>
              </div>
              <span className="text-[10px] text-gray-400">{CAT_LABELS[elA.cat]}</span>
            </div>

            {/* Connecting line */}
            <div className="relative mx-4 w-20 flex items-center">
              <div className="w-full border-t-2 border-dashed border-gray-300" />
              {elB && elA.en > 0 && elB.en > 0 && (
                <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-medium text-gray-500 whitespace-nowrap">
                  ΔEN {Math.abs(elA.en - elB.en).toFixed(2)}
                </span>
              )}
            </div>

            {/* Element B or placeholder */}
            <div className="flex flex-col items-center gap-1.5">
              {elB ? (
                <>
                  <div
                    className="w-[88px] h-[88px] rounded-full flex flex-col items-center justify-center shadow-sm"
                    style={{
                      background: CAT_COLORS[elB.cat].fill,
                      border: `2.5px solid ${CAT_COLORS[elB.cat].stroke}`,
                    }}
                  >
                    <span className="text-white text-xl font-bold leading-tight">{elB.s}</span>
                    <span className="text-white/70 text-[10px] leading-tight">{elB.n}</span>
                  </div>
                  <span className="text-[10px] text-gray-400">{CAT_LABELS[elB.cat]}</span>
                </>
              ) : (
                <>
                  <div className="w-[88px] h-[88px] rounded-full border-2 border-dashed border-gray-300 flex flex-col items-center justify-center">
                    <svg className="w-5 h-5 text-gray-300 mb-0.5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                    </svg>
                    <span className="text-[10px] text-gray-400 leading-tight">Select 2nd</span>
                  </div>
                  <span className="text-[10px] text-transparent select-none">placeholder</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Compound info panel ──────────────────────────────────────── */}
      {mode === "compound" && elA && elB && (
        <div ref={compoundPanelRef} className="mt-5 rounded-2xl border border-[var(--border)] bg-white p-6 shadow-[var(--shadow)]">
          <div className="flex items-start justify-between mb-5">
            <div>
              <h3 className="text-lg font-semibold text-[var(--foreground)]">
                {elA.s}–{elB.s} Binary System
              </h3>
              <p className="mt-0.5 text-sm text-[var(--muted)]">
                Exploring the {elA.n}–{elB.n} composition space
              </p>
            </div>
            <button
              onClick={() => { selBRef.current = null; setSelectedB(null); setMpPhases(null); draw(); }}
              className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted)] hover:bg-gray-50 transition-colors"
            >
              ← Back to scatter
            </button>
          </div>

          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Electronegativity</div>
              <div className="text-sm font-semibold text-[var(--foreground)]">
                ΔEN = {Math.abs(elA.en - elB.en).toFixed(2)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {elA.s} ({elA.en.toFixed(2)}) · {elB.s} ({elB.en.toFixed(2)})
              </div>
            </div>

            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Element Categories</div>
              <div className="flex flex-wrap gap-1 mt-1">
                <span className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                  style={{ background: CAT_COLORS[elA.cat].bg, color: CAT_COLORS[elA.cat].text, border: `1px solid ${CAT_COLORS[elA.cat].stroke}` }}>
                  {CAT_LABELS[elA.cat]}
                </span>
                <span className="inline-block rounded px-1.5 py-0.5 text-xs font-medium"
                  style={{ background: CAT_COLORS[elB.cat].bg, color: CAT_COLORS[elB.cat].text, border: `1px solid ${CAT_COLORS[elB.cat].stroke}` }}>
                  {CAT_LABELS[elB.cat]}
                </span>
              </div>
            </div>

            <div className="rounded-lg bg-gray-50 p-4">
              <div className="text-xs font-medium text-gray-500 mb-1">Known Phases (MP)</div>
              {mpLoading && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
                  Loading…
                </div>
              )}
              {mpError && <div className="text-xs text-red-500">{mpError}</div>}
              {mpPhases && !mpLoading && (
                <div className="text-sm font-semibold text-[var(--foreground)]">
                  {mpPhases.length} phase{mpPhases.length !== 1 ? "s" : ""}
                  {stablePhases.length > 0 && (
                    <span className="ml-1 text-xs font-normal text-emerald-600">
                      ({stablePhases.length} stable)
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* MP phases table */}
          {mpPhases && mpPhases.length > 0 && !mpLoading && (
            <div className="mb-5 overflow-x-auto rounded-lg border border-gray-200">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-gray-200 bg-gray-50 text-gray-500">
                    <th className="px-3 py-2 font-medium">Formula</th>
                    <th className="px-3 py-2 font-medium">Space Group</th>
                    <th className="px-3 py-2 font-medium">Crystal System</th>
                    <th className="px-3 py-2 font-medium text-right">E<sub>hull</sub> (eV/at)</th>
                    <th className="px-3 py-2 font-medium text-right">E<sub>form</sub> (eV/at)</th>
                    <th className="px-3 py-2 font-medium text-right">Density (g/cm³)</th>
                    <th className="px-3 py-2 font-medium text-right">a (Å)</th>
                    <th className="px-3 py-2 font-medium text-right">b (Å)</th>
                    <th className="px-3 py-2 font-medium text-right">c (Å)</th>
                    <th className="px-3 py-2 font-medium text-center">Stable</th>
                    <th className="px-3 py-2 font-medium">MP ID</th>
                  </tr>
                </thead>
                <tbody>
                  {mpPhases.slice(0, 20).map((p) => (
                    <tr key={p.id} className="border-b border-gray-100 hover:bg-gray-50/60">
                      <td className="px-3 py-2 font-medium text-[var(--foreground)]">{p.formula}</td>
                      <td className="px-3 py-2 text-gray-600">{p.spacegroup}</td>
                      <td className="px-3 py-2 text-gray-600 capitalize">{p.crystal_system}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.e_above_hull === "number" ? p.e_above_hull.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.formation_energy === "number" ? p.formation_energy.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.density === "number" ? p.density.toFixed(2) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.a === "number" ? p.a.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.b === "number" ? p.b.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                        {typeof p.c === "number" ? p.c.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-center">
                        {p.is_stable
                          ? <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" title="Stable" />
                          : <span className="inline-block h-2 w-2 rounded-full bg-gray-300" title="Metastable" />}
                      </td>
                      <td className="px-3 py-2">
                        <a href={`https://next-gen.materialsproject.org/materials/${p.id}`}
                          target="_blank" rel="noopener noreferrer"
                          className="text-blue-600 hover:underline">{p.id}</a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {mpPhases.length > 20 && (
                <div className="px-3 py-2 text-xs text-gray-400 bg-gray-50 border-t border-gray-200">
                  Showing 20 of {mpPhases.length} phases
                </div>
              )}
            </div>
          )}

          {mpPhases && mpPhases.length === 0 && !mpLoading && (
            <div className="mb-5 rounded-lg border border-dashed border-gray-300 p-4 text-center text-sm text-gray-400">
              No known phases found in Materials Project for {elA.s}–{elB.s}
            </div>
          )}

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-gray-600">
              Candidates
              <input
                type="number"
                min={10}
                max={200}
                step={10}
                value={nCandidates}
                onChange={(e) => setNCandidates(Number(e.target.value) || 50)}
                className="h-10 w-20 rounded-lg border border-gray-200 bg-gray-50 px-3 text-sm text-gray-900 tabular-nums"
              />
            </label>
            <button
              type="button"
              disabled={isGenerating}
              onClick={() => onGenerate?.(elA.s, elB.s, nCandidates)}
              className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 active:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isGenerating ? (
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714a2.25 2.25 0 0 0 .659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 0 1-1.59.659H9.06a2.25 2.25 0 0 1-1.591-.659L5 14.5m14 0H5" />
                </svg>
              )}
              {isGenerating ? "Generating…" : `Generate New Crystal Structure Candidates for ${elA.s}–${elB.s}`}
            </button>
          </div>
        </div>
      )}

      {/* Legend (hide in compound mode) */}
      {mode !== "compound" && (
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1">
          {LEGEND_GROUPS.map(cat => {
            const c = CAT_COLORS[cat];
            return (
              <div key={cat} className="flex items-center gap-1.5 text-xs text-slate-500">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm"
                  style={{ background: c.bg, border: `1.5px solid ${c.stroke}` }}
                />
                {CAT_LABELS[cat]}
              </div>
            );
          })}
          {elA && (
            <span className="ml-auto text-xs text-[var(--muted)]">
              Distance&nbsp;∝&nbsp;|ΔEN| from {elA.s}&nbsp;(EN&nbsp;=&nbsp;{elA.en})
            </span>
          )}
        </div>
      )}
    </section>
  );
}
